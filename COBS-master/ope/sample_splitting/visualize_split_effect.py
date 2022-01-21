import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import loadtxt

x_axis = [8,16,32,64,128]
random_seeds = [37,66,100,1024,1330]

nosplit_data = []
split_data = []

# # IS split effect
# # Read in different random runs
# for seed in random_seeds:
#     temp = np.loadtxt("./COBS-master/ope/sample_splitting/plot_arrays/" + str(seed) + "_nosplit_paper_8-128_IS_FQE.csv", delimiter=",", unpack=False)
#     nosplit_data.append(temp)
#     temp = np.loadtxt("./COBS-master/ope/sample_splitting/plot_arrays/" + str(seed) + "_split_paper_8-128_IS_FQE.csv", delimiter=",", unpack=False)
#     split_data.append(temp)

# IS_paper_nosplit = np.mean(nosplit_data, axis=0)
# IS_paper_split = np.mean(split_data, axis=0)
# IS_paper_nosplit_var = np.std(nosplit_data, axis=0) / math.sqrt(len(random_seeds))
# IS_paper_split_var = np.std(split_data, axis=0) / math.sqrt(len(random_seeds))

# plt.figure(0)
# #plt.figure(figsize=(5, 4))
# plt.errorbar(x_axis, IS_paper_split, yerr=IS_paper_split_var,label='split IS', linestyle='-')
# plt.errorbar(x_axis, IS_paper_nosplit, yerr=IS_paper_nosplit_var, label='no split IS', linestyle='-.')
# plt.xlabel('Number of trajectories')
# plt.ylabel('MSE')
# plt.legend()
# plt.savefig('IS_paper_comparison')
# plt.close(0)

nosplit_data_AM = []
nosplit_data_DR = []
nosplit_data_WDR = []
nosplit_data_MAGIC = []
nosplit_data_SDR = []
split_data_AM = []
split_data_DR = []
split_data_WDR = []
split_data_MAGIC = []
split_data_SDR = []

models = ['FQE', 'Retrace', 'Tree-Backup', 'Q^pi(lambda)', 'Q-Reg', 'MRDR', 'MBased']
# HM split effect
count = 0
for model in models:
    for seed in random_seeds:
        nosplit_data_AM.append(np.loadtxt("./COBS-master/ope/sample_splitting/plot_arrays/" + str(seed) + "_nosplit_paper_8-128_AM_" + model + ".csv", delimiter=",", unpack=False))
        split_data_AM.append(np.loadtxt("./COBS-master/ope/sample_splitting/plot_arrays/" + str(seed) + "_split_paper_8-128_AM_" + model + ".csv", delimiter=",", unpack=False))
        nosplit_data_DR.append(np.loadtxt("./COBS-master/ope/sample_splitting/plot_arrays/" + str(seed) + "_nosplit_paper_8-128_DR_" + model + ".csv", delimiter=",", unpack=False))
        split_data_DR.append(np.loadtxt("./COBS-master/ope/sample_splitting/plot_arrays/" + str(seed) + "_split_paper_8-128_DR_" + model + ".csv", delimiter=",", unpack=False))
        nosplit_data_WDR.append(np.loadtxt("./COBS-master/ope/sample_splitting/plot_arrays/" + str(seed) + "_nosplit_paper_8-128_WDR_" + model + ".csv", delimiter=",", unpack=False))
        split_data_WDR.append(np.loadtxt("./COBS-master/ope/sample_splitting/plot_arrays/" + str(seed) + "_split_paper_8-128_WDR_" + model + ".csv", delimiter=",", unpack=False))
        nosplit_data_MAGIC.append(np.loadtxt("./COBS-master/ope/sample_splitting/plot_arrays/" + str(seed) + "_nosplit_paper_8-128_MAGIC_" + model + ".csv", delimiter=",", unpack=False))
        split_data_MAGIC.append(np.loadtxt("./COBS-master/ope/sample_splitting/plot_arrays/" + str(seed) + "_split_paper_8-128_MAGIC_" + model + ".csv", delimiter=",", unpack=False))
        nosplit_data_SDR.append(np.loadtxt("./COBS-master/ope/sample_splitting/plot_arrays/" + str(seed) + "_nosplit_paper_8-128_SDR_" + model + ".csv", delimiter=",", unpack=False))
        split_data_SDR.append(np.loadtxt("./COBS-master/ope/sample_splitting/plot_arrays/" + str(seed) + "_split_paper_8-128_SDR_" + model + ".csv", delimiter=",", unpack=False))
    paper_nosplit_DM = np.mean(nosplit_data_AM, axis=0)
    paper_split_DM = np.mean(split_data_AM, axis=0)
    paper_nosplit_DM_var = np.std(nosplit_data_AM, axis=0) / math.sqrt(len(random_seeds))
    paper_split_DM_var = np.std(split_data_AM, axis=0) / math.sqrt(len(random_seeds))
    paper_nosplit_DR = np.mean(nosplit_data_DR, axis=0)
    paper_split_DR = np.mean(split_data_DR, axis=0)
    paper_nosplit_DR_var = np.std(nosplit_data_DR, axis=0) / math.sqrt(len(random_seeds))
    paper_split_DR_var = np.std(split_data_DR, axis=0) / math.sqrt(len(random_seeds))
    paper_nosplit_WDR = np.mean(nosplit_data_WDR, axis=0)
    paper_split_WDR = np.mean(split_data_WDR, axis=0)
    paper_nosplit_WDR_var = np.std(nosplit_data_WDR, axis=0) / math.sqrt(len(random_seeds))
    paper_split_WDR_var = np.std(split_data_WDR, axis=0) / math.sqrt(len(random_seeds))
    paper_nosplit_MAGIC = np.mean(nosplit_data_MAGIC, axis=0)
    paper_split_MAGIC = np.mean(split_data_MAGIC, axis=0)
    paper_nosplit_MAGIC_var = np.std(nosplit_data_MAGIC, axis=0) / math.sqrt(len(random_seeds))
    paper_split_MAGIC_var = np.std(split_data_MAGIC, axis=0) / math.sqrt(len(random_seeds))
    paper_nosplit_SDR = np.mean(nosplit_data_SDR, axis=0)
    paper_split_SDR = np.mean(split_data_SDR, axis=0)
    paper_nosplit_SDR_var = np.std(nosplit_data_SDR, axis=0) / math.sqrt(len(random_seeds))
    paper_split_SDR_var = np.std(split_data_SDR, axis=0) / math.sqrt(len(random_seeds))

    # paper_nosplit_DR = pd.read_csv("./COBS-master/ope/sample_splitting/plot_arrays/" + random_seed + "nosplit_paper_8-128_DR_" + model + ".csv", header=None)
    # paper_nosplit_WDR = pd.read_csv("./COBS-master/ope/sample_splitting/plot_arrays/" + random_seed + "nosplit_paper_8-128_WDR_" + model + ".csv", header=None)
    # paper_nosplit_MAGIC = pd.read_csv("./COBS-master/ope/sample_splitting/plot_arrays/" + random_seed + "nosplit_paper_8-128_MAGIC_" + model + ".csv", header=None)
    # paper_nosplit_SDR = pd.read_csv("./COBS-master/ope/sample_splitting/plot_arrays/" + random_seed + "nosplit_paper_8-128_SDR_" + model + ".csv", header=None)
    # paper_split_DR = pd.read_csv("./COBS-master/ope/sample_splitting/plot_arrays/" + random_seed + "split_paper_8-128_DR_" + model + ".csv", header=None)
    # paper_split_WDR = pd.read_csv("./COBS-master/ope/sample_splitting/plot_arrays/" + random_seed + "split_paper_8-128_WDR_" + model + ".csv", header=None)
    # paper_split_MAGIC = pd.read_csv("./COBS-master/ope/sample_splitting/plot_arrays/" + random_seed + "split_paper_8-128_MAGIC_" + model + ".csv", header=None)
    # paper_split_SDR = pd.read_csv("./COBS-master/ope/sample_splitting/plot_arrays/" + random_seed + "split_paper_8-128_SDR_" + model + ".csv", header=None)
    
    plt.figure(count)
    plt.title(model)
    #plt.figure(figsize=(5, 4))
    # plot
    # plt.errorbar(x_axis, paper_nosplit_DM, yerr=paper_nosplit_DM_var, label= 'nosplit DM', color='forestgreen', linestyle='--')
    # plt.errorbar(x_axis, paper_nosplit_DR, yerr=paper_nosplit_DR_var, label= 'nosplit DR', color='saddlebrown', linestyle=':')
    # plt.errorbar(x_axis, paper_nosplit_WDR, yerr=paper_nosplit_WDR_var, label= 'nosplit WDR', color='purple', linestyle='-.')
    plt.errorbar(x_axis, paper_nosplit_MAGIC, yerr=paper_nosplit_MAGIC_var, label= 'nosplit MAGIC', color='firebrick', linestyle='solid')
    # plt.errorbar(x_axis, paper_split_DM, yerr=paper_split_DM_var, label= 'split DM', color='limegreen', linestyle='--')
    # plt.errorbar(x_axis, paper_split_DR, yerr=paper_split_DR_var, label= 'split DR', color='gray', linestyle=':')
    # plt.errorbar(x_axis, paper_split_WDR, yerr=paper_split_WDR_var, label= 'split WDR', color='mediumorchid', linestyle='-.')
    plt.errorbar(x_axis, paper_split_MAGIC, yerr=paper_split_MAGIC_var, label= 'split MAGIC', color='salmon', linestyle='solid')

    plt.ylim(top=1)  # adjust the top leaving bottom unchanged
    plt.ylim(bottom=0)  # adjust the top leaving bottom unchanged
    plt.xlabel('Number of trajectories')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig('paper_comparison_8-128_' + model)
    plt.close(count)
    count += 1