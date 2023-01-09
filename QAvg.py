from tqdm import tqdm
from myutils import *
import matplotlib.pyplot as plt
import argparse
import seaborn
import os

parser = argparse.ArgumentParser()
parser.add_argument("--nS", default=5, type=int)
parser.add_argument("--nA", default=5, type=int)
parser.add_argument("--gamma", default=0.9, type=float)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--lr_decay", default=0.95, type=float)
parser.add_argument("--n", default=5, type=int)
parser.add_argument("--iter", default=128)
args = parser.parse_args()

nS, nA, gamma = args.nS, args.nA, args.gamma
ntrain = args.n
ntest = 4*args.n
d0 = np.ones(nS)/nS

epochs = 16000

E_list = (1, 2, 4, 8, 16, 32)
kappa_list = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
E_size = len(E_list) + 1
kappa_size = len(kappa_list)

data = [[],[],[],[],[],[],[]]
iterations = [4,8,16,32,64,128,256,512]

Seed = 101

save_dir = './QAvg_randomMDP'

def experiment():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.random.seed(Seed)
    seeds = np.random.randint(low=0, high=1000000, size=epochs)

    total_center = np.zeros((epochs, kappa_size, E_size, args.iter))

    for count in tqdm(range(epochs)):
        seed = seeds[count]
        R = np.random.uniform(low=0, high=1, size=(nS, nA))
        Q_init = np.random.uniform(size=(nS, nA))
        mix_Ps = generate_mix_Ps_Orth(n=ntrain + ntest, nS=nS, nA=nA, kappa_list=kappa_list, seed=seed)

        for kappa in range(kappa_size):
            Ps = mix_Ps[kappa]
            center_P = [Ps[0]]
            train_P = Ps[:ntrain]
            test_P = Ps[ntrain:]

            MEObj_list = np.zeros((E_size, args.iter))
            test_MEObj_list = np.zeros((E_size, args.iter))
            test_center = np.zeros((E_size, args.iter))

            # QAvg with different Es
            for E_num in range(E_size-1):
                E = E_list[E_num]
                Q = Q_init.copy()
                lr = args.lr
                for e in range(args.iter // E):
                    # If the decrease is too small or negative, reduce the lr
                    if e > 2 and (MEObj_list[E_num][(e - 1)*E] - MEObj_list[E_num][(e - 2)*E]) < MEObj_list[E_num][(e - 2)*E]*1e-3:
                        lr = lr*args.lr_decay
                    V = QtoV(Q)
                    pi = QtoPolicy(Q)
                    center_MEObj = evaluate(pi, R, center_P, d0, gamma)
                    MEObj = evaluate(pi, R, train_P, d0, gamma)
                    MEObj_test = evaluate(pi, R, test_P, d0, gamma)
                    for t in range(E):
                        MEObj_list[E_num][e*E + t] = MEObj
                        test_MEObj_list[E_num][e*E + t] = MEObj_test
                        test_center[E_num][e*E + t] = center_MEObj
                    Qs = []
                    for i in range(ntrain):
                        Vi = V.copy()
                        for _ in range(E):
                            delta_Vi, delta_pi = value_iteration(Vi, train_P[i], R, gamma)
                            Vi = (1-lr)*Vi + lr*delta_Vi
                        Qi = VtoQ(Vi, train_P[i], R, gamma)
                        Qs.append(Qi)
                    Q = sum(Qs)/ntrain

            # Baseline: Train every agent separately, i.e. do not merge
            lr = args.lr
            Q = Q_init.copy()
            Qs = [Q.copy() for _ in range(ntrain)]
            Q_avg = Q
            for e in range(args.iter):
                if e > 2 and (MEObj_list[-1][e-1] - MEObj_list[-1][e-2]) < MEObj_list[-1][e-2]*1e-3:
                    lr = lr*args.lr_decay
                pi_avg = QtoPolicy(Q_avg)
                MEObj = evaluate(pi_avg, R, train_P, d0, gamma)
                MEObj_test = evaluate(pi_avg, R, test_P, d0, gamma)
                center_MEObj = evaluate(pi_avg, R, center_P, d0, gamma)

                MEObj_list[-1][e] = MEObj
                test_MEObj_list[-1][e] = MEObj_test
                test_center[-1][e] = center_MEObj

                for i in range(ntrain):
                    Qi = Qs[i]
                    Vi = QtoV(Qi)
                    delta_Vi, delta_pi = value_iteration(Vi, train_P[i], R, gamma)
                    Vi = (1-lr)*Vi + lr*delta_Vi
                    Qi = VtoQ(Vi, train_P[i], R, gamma)
                    Qs[i] = Qi
                Q_avg = sum(Qs)/ntrain
                best_MEObj = -inf
                avg_MEObj = 0
                avg_MEObj_test = 0
                MEObj_tests = []
                for i in range(ntrain):
                    Qi = Qs[i]
                    pi_i = QtoPolicy(Qi)
                    MEObj_i = evaluate(pi_i, R, train_P, d0, gamma)
                    avg_MEObj += MEObj_i
                    if MEObj_i > best_MEObj:
                        best_MEObj = MEObj_i
                    # Test
                    MEObj_i_test = evaluate(pi_i, R, test_P, d0, gamma)
                    avg_MEObj_test += MEObj_i_test
                    MEObj_tests.append(MEObj_i_test)
                    

            test_center = test_center - 0.5/(1 - gamma)
            total_center[count][kappa] = test_center
    np.save(save_dir + '/test_center.npy', total_center)

def report():
    coff = np.sqrt(epochs)
    total_center = np.load(save_dir + '/test_center.npy')
    for E_num in range(E_size):
        print('-----------------------------------------------------------------------------')
        if E_num == E_size - 1:
            print('E = Inf:')
        else:
            print(f'E = {E_list[E_num]}:')
        center = total_center[:, :, E_num, :]
        mean = []
        standard = []
        for kappa in range(kappa_size):
            mean.append(np.average(center[:, kappa, :], axis=0))
            standard.append(np.std(center[:, kappa, :], axis=0) / coff)
            for i in range(len(mean[kappa])):
                    # use to print the table that shows the objective values of FedRL at different iterations during the training of QAvgs with different E
                    if kappa == 2 and E_num != E_size - 1 and (i+1) in iterations:
                        data[E_num].append(mean[kappa][i])
        for kappa in range(kappa_size):
            print(f'Center kappa = {kappa_list[kappa]}' + f':  Mean = {mean[kappa][-1]}'.ljust(30) + f' Std = {standard[kappa][-1]}'.ljust(30))

def print_table(data):
    seaborn.set()
    np.random.seed(0)
    f,ax = plt.subplots(figsize=(10,6))
    seaborn.heatmap(data,vmin=0,vmax=1,cmap='YlOrRd',annot=True,linewidths=2,cbar=None)
    
    ax.set_xlabel('Iteration')
    ax.xaxis.tick_top()
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    x_scale_ls = range(6)
    x_index_ls = ['4','8','16','32','64','128']
    plt.xticks(x_scale_ls,x_index_ls)
    y_scale_ls = range(6)
    y_index_ls = ['E = 1','E = 2','E = 4','E = 8','E = 16','E = 32']
    plt.yticks(y_scale_ls,y_index_ls)
    
    plt.show()

experiment()
report()
print_table(data)