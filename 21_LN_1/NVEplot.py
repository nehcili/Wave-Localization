import matplotlib.pyplot as plt
import os
from numpy import loadtxt
from iodata import load_data_set_
from benchmark import weyl
import numpy as np

def plot_data_set(data_set_name, index, data_folder_path="data"):
    file_name = os.path.join(data_folder_path, data_set_name + "_" + str(index) + "_nve.txt")
    nve = loadtxt(file_name, delimiter=',')

    d = {}
    info_file = os.path.join(data_folder_path, data_set_name + "_" + str(index) + "_info.txt")
    with open(info_file) as f:
        line = f.readline()
        while line:
            k, v = line.split(':', 1)
            d[k] = v
            line = f.readline()

    #print(nve[:,0].shape, nve[:,1].shape)
    fig=plt.figure(figsize=(18, 16), dpi= 50, facecolor='w', edgecolor='k')
    plt.scatter(nve[:,0], nve[:,1], label="Eigenvalue counting")


    #print(d)
    subtitle = ""
    for k in d:
        if k != 'potential type':
            subtitle += k + ":" + d[k].rstrip('\n') + '; '

    plt.title(subtitle)
    plt.suptitle(d['potential type'], fontsize=14, fontweight='bold')
    plt.xlabel("Energy")
    plt.ylabel("Eigenvalue Count")
    plt.show()


def plot_model_pred(model, data_set_name, index, data_folder_path="data"):
    p, E, target = load_data_set_(data_folder_path, data_set_name, [index])

    d = {}
    info_file = os.path.join(data_folder_path, data_set_name + "_" + str(index) + "_info.txt")
    with open(info_file) as f:
        line = f.readline()
        while line:
            k, v = line.split(':', 1)
            d[k] = v
            line = f.readline()

    #print(nve[:,0].shape, nve[:,1].shape)
    fig=plt.figure(figsize=(18, 16), dpi= 50, facecolor='w', edgecolor='k')
    plt.scatter(E, target, color="blue", label="True Eigenvalue counting")
    weyl_base = weyl((p,E)).numpy().astype(np.float32)
    #print('weyl shape', weyl_base.shape)
    plt.scatter(E, weyl_base, color="gray", label="Prediction by weyl's law")

    pred = model((p,E))
    #print('pred shape', pred.shape)
    plt.scatter(E, pred, color="orange", label="Prediction by {}".format(model.name))


    #print(d)
    subtitle = ""
    for k in d:
        if k != 'potential type':
            subtitle += k + ":" + d[k].rstrip('\n') + '; '

    plt.title(subtitle)
    plt.suptitle(d['potential type'], fontsize=14, fontweight='bold')
    plt.xlabel("Energy")
    plt.ylabel("Eigenvalue Count")
    plt.show()
