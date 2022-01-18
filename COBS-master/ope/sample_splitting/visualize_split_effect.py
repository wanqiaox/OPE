import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import loadtxt

x_axis = [8,16,32,64,128]

# IS split effect
IS_toy_nosplit = pd.read_csv("./COBS-master/ope/sample_splitting/plot_arrays/nosplit_toy_8-128_IS_FQE.csv", header=None)
IS_toy_split = pd.read_csv("./COBS-master/ope/sample_splitting/plot_arrays/split_toy_8-128_IS_FQE.csv", header=None)

# plt.figure(0)
# #plt.figure(figsize=(5, 4))
# plt.plot(x_axis, IS_toy_split, label='split IS', linestyle='-')
# plt.plot(x_axis, IS_toy_nosplit, label='no split IS', linestyle='-.')

# plt.xlabel('Number of trajectories')
# plt.ylabel('MSE')
# plt.legend()
# plt.savefig('IS_toy_comparison')
# plt.close(0)

models = ['FQE', 'Retrace', 'Tree-Backup', 'Q^pi(lambda)', 'Q-Reg', 'MRDR', 'MBased']
# HM split effect
count = 0
for model in models:
    toy_nosplit_AM = pd.read_csv("./COBS-master/ope/sample_splitting/plot_arrays/nosplit_toy_8-128_AM_" + model + ".csv", header=None)
    toy_nosplit_DR = pd.read_csv("./COBS-master/ope/sample_splitting/plot_arrays/nosplit_toy_8-128_DR_" + model + ".csv", header=None)
    toy_nosplit_WDR = pd.read_csv("./COBS-master/ope/sample_splitting/plot_arrays/nosplit_toy_8-128_WDR_" + model + ".csv", header=None)
    toy_nosplit_MAGIC = pd.read_csv("./COBS-master/ope/sample_splitting/plot_arrays/nosplit_toy_8-128_MAGIC_" + model + ".csv", header=None)
    toy_nosplit_SDR = pd.read_csv("./COBS-master/ope/sample_splitting/plot_arrays/nosplit_toy_8-128_SDR_" + model + ".csv", header=None)
    toy_split_AM = pd.read_csv("./COBS-master/ope/sample_splitting/plot_arrays/split_toy_8-128_AM_" + model + ".csv", header=None)
    toy_split_DR = pd.read_csv("./COBS-master/ope/sample_splitting/plot_arrays/split_toy_8-128_DR_" + model + ".csv", header=None)
    toy_split_WDR = pd.read_csv("./COBS-master/ope/sample_splitting/plot_arrays/split_toy_8-128_WDR_" + model + ".csv", header=None)
    toy_split_MAGIC = pd.read_csv("./COBS-master/ope/sample_splitting/plot_arrays/split_toy_8-128_MAGIC_" + model + ".csv", header=None)
    toy_split_SDR = pd.read_csv("./COBS-master/ope/sample_splitting/plot_arrays/split_toy_8-128_SDR_" + model + ".csv", header=None)
    
    plt.figure(count)
    plt.title(model)
    #plt.figure(figsize=(5, 4))
    # plot
    # plt.plot(x_axis, toy_nosplit_AM, label= 'nosplit AM', color='forestgreen', linestyle='--')
    plt.plot(x_axis, toy_nosplit_DR, label= 'nosplit DR', color='saddlebrown', linestyle=':')
    # plt.plot(x_axis, toy_nosplit_WDR, label= 'nosplit WDR', color='purple', linestyle='-.')
    # plt.plot(x_axis, toy_nosplit_MAGIC, label= 'nosplit MAGIC', color='firebrick', linestyle='solid')
    # plt.plot(x_axis, toy_split_AM, label= 'split AM', color='limegreen', linestyle='--')
    plt.plot(x_axis, toy_split_AM, label= 'split DR', color='gray', linestyle=':')
    # plt.plot(x_axis, toy_split_AM, label= 'split WDR', color='mediumorchid', linestyle='-.')
    # plt.plot(x_axis, toy_split_AM, label= 'split MAGIC', color='salmon', linestyle='solid')

    plt.xlabel('Number of trajectories')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig('toy_comparison_8-128_' + model)
    plt.close(count)
    count += 1