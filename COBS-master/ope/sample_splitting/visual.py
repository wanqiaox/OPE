import matplotlib.pyplot as plt
import numpy as np

def visualize(dic, x_axis):
    assert(len(dic) == len(x_axis))

    # Change to 'split' if splitting
    is_split = 'realnosplit'
    
    # Choose one model among: FQE, Retrace, Tree-Backup, Q^pi(lambda), Q-Reg, MRDR, MBased, IS
    models = ['FQE', 'Retrace', 'Tree-Backup', 'Q^pi(lambda)', 'Q-Reg', 'MRDR', 'MBased']
    #model = 'FQE'
    count = 0
    for model in models:
        AM = []
        DR = []
        WDR = []
        MAGIC = []
        SDR = []
        IS = []
        plt.figure(count)
        #plt.figure(figsize=(5, 4))
        for num, item in enumerate(dic):
            for key, val in item.items():
                if key.find('AM ' + model) != -1:
                    AM.append(val[1])
                elif key == 'DR ' + model:
                    DR.append(val[1])
                elif key.find('WDR ' + model) != -1:
                    WDR.append(val[1])
                elif key.find('MAGIC ' + model) != -1:
                    MAGIC.append(val[1])
                elif key.find('SDR ' + model) != -1:
                    SDR.append(val[1])
                elif key.find('IS', 0, 2) != -1:
                    IS.append(val[1])
        x_axis = np.asarray(x_axis)
        AM = np.asarray(AM)
        DR = np.asarray(DR)
        WDR = np.asarray(WDR)
        MAGIC = np.asarray(MAGIC)
        SDR = np.asarray(SDR)
        IS = np.asarray(IS)
        # Save the results vector for plotting both split and non-split
        np.savetxt("./COBS-master/ope/sample_splitting/plot_arrays/" + is_split + "_paper_8-128_AM_" + model + ".csv", AM, delimiter=",")
        np.savetxt("./COBS-master/ope/sample_splitting/plot_arrays/" + is_split + "_paper_8-128_DR_" + model + ".csv", DR, delimiter=",")
        np.savetxt("./COBS-master/ope/sample_splitting/plot_arrays/" + is_split + "_paper_8-128_WDR_" + model + ".csv", WDR, delimiter=",")
        np.savetxt("./COBS-master/ope/sample_splitting/plot_arrays/" + is_split + "_paper_8-128_MAGIC_" + model + ".csv", MAGIC, delimiter=",")
        np.savetxt("./COBS-master/ope/sample_splitting/plot_arrays/" + is_split + "_paper_8-128_SDR_" + model + ".csv", SDR, delimiter=",")
        np.savetxt("./COBS-master/ope/sample_splitting/plot_arrays/" + is_split + "_paper_8-128_IS_" + model + ".csv", IS, delimiter=",")
        # plot
        plt.plot(x_axis, AM, label= is_split + ' AM ' + model)
        plt.plot(x_axis, DR, label= is_split + ' DR ' + model)
        plt.plot(x_axis, WDR, label= is_split + ' WDR ' + model)
        plt.plot(x_axis, MAGIC, label= is_split + ' MAGIC ' + model)
        plt.plot(x_axis, SDR, label= is_split + ' SDR ' + model)
        plt.plot(x_axis, IS, label= is_split + ' IS ' + model)
        plt.xlabel('Number of trajectories')
        plt.ylabel('MSE')
        plt.legend()
        plt.savefig(is_split + '_paper_8-128_' + model)
        plt.close(count)
        count += 1