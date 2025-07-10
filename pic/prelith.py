import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import set_option
import torch
set_option("display.max_rows", 10)
pd.options.mode.chained_assignment = None
# filename = 'facies_vectors.csv'
training_data = pd.read_csv('')
training_data  = training_data .dropna()
n= training_data[training_data['Well Name'] == '']


file_path_gat_cdan = "wa_wb_gat_cdan.txt"
file_path_gat_dann = "wa_wb_gat_dann.txt"
file_path_gat_mdd = "wa_wb_gat_mdd.txt"
file_path_dgcn_daal = "wa_wb_dgcn_daac.txt"

def read_file(file_name):
    with open(file_name, "r") as file:
        data_list = []
        for line in file:
            data_list.append(int(line.strip()))
        return data_list


pre_log_gat_cdan =torch.tensor(read_file(file_path_gat_cdan))[1381:]
pre_log_gat_dann =torch.tensor(read_file(file_path_gat_dann))[1381:]
pre_log_gat_mdd =torch.tensor(read_file(file_path_gat_mdd))[1381:]
pre_log_dgcn_daal =torch.tensor(read_file(file_path_dgcn_daal))[1381:]



print("trainningdata:",n.shape)
print("prelog_gat:",pre_log_gat_cdan.shape)
print("prelog_mdd:",pre_log_gat_mdd.shape)
print("prelog_daal:",pre_log_dgcn_daal.shape)



# print(pre_log)
# pre_log = random_tensor = torch.randint(0, 9, size=(2783,))
# print("pre:",pre_log)



training_data = training_data[training_data['Well Name'] == 'CROSS H CATTLE']

print(training_data.shape)
training_data['Well Name'] = training_data['Well Name'].astype('category')
# training_data['Formation'] = training_data['Formation'].astype('category')
training_data['Well Name'].unique()
# print(training_data['Well Name'])
# print(training_data['Formation'])
# 1=sandstone  2=c_siltstone   3=f_siltstone
# 4=marine_silt_shale 5=mudstone 6=wackestone 7=dolomite
# 8=packstone 9=bafflestone
# facies_colors = ['#F4D03F', '#F5B041', '#DC7633', '#6E2C00',
#                  '#1B4F72', '#2E86C1', '#AED6F1', '#A569BD', '#196F3D']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00',
       '#1B4F72','#FF0000', '#AED6F1', '#A569BD', '#196F3D']

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D','PS', 'BS']
# facies_color_map is a dictionary that maps facies labels
# to their respective colors
facies_color_map = {}
for ind, label in enumerate(facies_labels):
    facies_color_map[label] = facies_colors[ind]

def make_facies_log_plot(logs,pre_log_gat_cdan,pre_log_gat_dann,pre_log_gat_mdd,pre_log_dgcn_daal,facies_colors):

    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
        facies_colors[0:len(facies_colors)], 'indexed')

    ztop = logs.Depth.min()
    zbot = logs.Depth.max()

    cluster = np.repeat(np.expand_dims(logs['Facies'].values, 1), 100, 1)
    pre_cluster_gat_cdan= np.repeat(np.expand_dims(pre_log_gat_cdan+1, 1), 100, 1)
    pre_cluster_gat_dann = np.repeat(np.expand_dims(pre_log_gat_dann+1, 1), 100, 1)
    pre_cluster_gat_mdd = np.repeat(np.expand_dims(pre_log_gat_mdd+1, 1), 100, 1)
    pre_cluster_dgcn_daal = np.repeat(np.expand_dims(pre_log_dgcn_daal+1, 1), 100, 1)

    f, ax = plt.subplots(nrows=1, ncols=10, figsize=(14, 12))
    ax[0].plot(logs.GR, logs.Depth, '-g')
    ax[1].plot(logs.ILD_log10, logs.Depth, '-')
    ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='0.5')
    ax[3].plot(logs.PHIND, logs.Depth, '-', color='r')
    ax[4].plot(logs.PE, logs.Depth, '-', color='black')
    im = ax[5].imshow(cluster, interpolation='none', aspect='auto',
                      cmap=cmap_facies, vmin=1, vmax=9)
    ax[6].imshow(pre_cluster_gat_cdan, interpolation='none', aspect='auto',
                      cmap=cmap_facies, vmin=1, vmax=9)
    ax[7].imshow(pre_cluster_gat_dann, interpolation='none', aspect='auto',
                 cmap=cmap_facies, vmin=1, vmax=9)
    ax[8].imshow(pre_cluster_gat_mdd, interpolation='none', aspect='auto',
                 cmap=cmap_facies, vmin=1, vmax=9)
    ax[9].imshow(pre_cluster_gat_cdan, interpolation='none', aspect='auto',
                 cmap=cmap_facies, vmin=1, vmax=9)
    divider = make_axes_locatable(ax[9])


    tick_positions = np.linspace(0.2, 18, num=len(facies_labels))

    cax = divider.append_axes("right", size="30%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ticks=tick_positions, shrink=0.8)
    cbar.ax.set_yticklabels(facies_labels, rotation=90, va='center')
    # cbar.set_label("Facies Labels", fontsize=14)
    cbar.ax.tick_params(size=0)
    cbar.ax.tick_params(axis='y', which='major', labelsize=10)

    # cbar = plt.colorbar(im, cax=cax)
    cbar = plt.colorbar(im, cax=cax, ticks=range(1, 10))
    cbar.ax.set_yticklabels(facies_labels)
    cbar.set_label((17 * ' ').join([' SS ', 'CSiS', 'FSiS',
                                    'SiSh', ' MS ', ' WS ', ' D  ',
                                    ' PS ', ' BS ']))
    # cbar.set_ticks(range(0, 1));

    cbar.set_ticklabels('')

    for i in range(len(ax) - 5):
        ax[i].set_ylim(ztop, zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)

    for i in range(6, 10):
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].set_xticks([])
        ax[i].set_yticks([])


    ax[0].set_xlabel("GR", rotation=45)
    ax[0].set_xlim(logs.GR.min(), logs.GR.max())
    ax[1].set_xlabel("ILD_log10", rotation=45)
    ax[1].set_xlim(logs.ILD_log10.min(), logs.ILD_log10.max())
    ax[2].set_xlabel("DeltaPHI", rotation=45)
    ax[2].set_xlim(logs.DeltaPHI.min(), logs.DeltaPHI.max())
    ax[3].set_xlabel("PHIND", rotation=45)
    ax[3].set_xlim(logs.PHIND.min(), logs.PHIND.max())
    ax[4].set_xlabel("PE", rotation=45)
    ax[4].set_xlim(logs.PE.min(), logs.PE.max())
    ax[5].set_xlabel('Facies', rotation=45)

    ax[6].set_xlabel('GAT+CDAN', rotation=45)
    ax[7].set_xlabel('GAT+DANN', rotation=45)
    ax[8].set_xlabel('GAT+MDD', rotation=45)
    ax[9].set_xlabel('DGCN-DAAC', rotation=45)


    ax[1].set_yticklabels([]);
    ax[2].set_yticklabels([]);
    ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]);
    ax[5].set_yticklabels([])
    ax[5].set_xticklabels([])
    ax[6].set_yticklabels([])
    ax[6].set_xticklabels([])
    ax[7].set_yticklabels([])
    ax[8].set_yticklabels([])
    ax[9].set_yticklabels([])

    ax[1].yaxis.set_ticks([])
    ax[2].yaxis.set_ticks([])
    ax[3].yaxis.set_ticks([])
    ax[4].yaxis.set_ticks([])
    ax[5].yaxis.set_ticks([])
    ax[6].yaxis.set_ticks([])
    ax[7].yaxis.set_ticks([])
    ax[8].yaxis.set_ticks([])
    ax[9].yaxis.set_ticks([])







    f.suptitle('Well: Well1', fontsize=14, y=0.94)

make_facies_log_plot(
    training_data[training_data['Well Name'] ==''],
    pre_log_gat_cdan,pre_log_gat_dann,pre_log_gat_mdd,pre_log_dgcn_daal,
    facies_colors)
plt.savefig("Well1.jpg", dpi=600, format="jpg")
plt.show()
