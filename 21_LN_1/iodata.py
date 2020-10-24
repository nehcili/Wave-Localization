############################################
# Libraries
############################################

# I/O
import numpy as np
from numpy import savetxt, loadtxt
import os


############################################
# I/O management
############################################
# saves
# pinfo = "{data_name}_info.txt" see log.md
# pds = "{data_name}_potential.txt" see log.md
# nvs = "{data_name}_nve.txt" see log.md
# data_folder_path = string
# data_set_name = string
# no rvalue
def save_data_set(pinfo: str, pds: np.ndarray, nve: np.ndarray, \
    data_folder_path: str, data_set_name: str, astype32=True) -> str:
    index = 0
    while os.path.isfile(os.path.join(data_folder_path, data_set_name + "_" + str(index) + "_potential.txt")):
        index += 1

    file_name = os.path.join(data_folder_path, data_set_name + "_" + str(index))
    if astype32:
        savetxt(file_name + "_potential.txt", pds.astype(np.float32), delimiter=',')
        savetxt(file_name + "_nve.txt", nve.astype(np.float32), delimiter=',')
    else:
        savetxt(file_name + "_potential.txt", pds, delimiter=',')
        savetxt(file_name + "_nve.txt", nve, delimiter=',')
    with open(file_name + "_info.txt", 'w') as f:
        f.write(pinfo)

    return file_name

# loads from "{data_set_name}_{counter}_..." for counter in range(start, end)
# start and end: counter for the data set
# rvalue:
#   batch_size = number of (E, N_V(E)) pairs in each file whose counter is in range(start, end)
#   potentials: batch_size x domain_size x 2, np.ndarray
#   E: batch_size, np.ndarray
#   target: batch_size, np.ndarray
def load_data_set_(data_folder_path: str, data_set_name: str, data_idx: list) -> tuple:
    potentials = []
    nve = []
    for index in data_idx:
        #print(data_set_name)
        file_name = os.path.join(data_folder_path, data_set_name + "_" + str(index))

        #print(file_name)
        nve.append(loadtxt(file_name + "_nve.txt", delimiter=','))

        potential = loadtxt(file_name + "_potential.txt", delimiter=',')
        shape = potential.shape
        potential = potential.reshape(1, shape[0], shape[1])
        potential = np.repeat(potential, nve[-1].shape[0], axis=0)
        potentials.append(potential)


    potentials = np.concatenate(potentials, axis=0)
    nve = np.concatenate(nve, axis=0)
    E = nve[:,0:1]
    target = nve[:,1:2]
    #print('iodata', E.shape, target.shape)

    return (potentials, E, target)

# load_data_set
# Overloaded function
# input:
#   file_name_prefix : str = the full path to the file upto indices of the file and not including "_"
#       e.g. data/[data_set_name]/train/discrete_10000
#   data_idx : suffix of the data name i.e. discrete_10000_{data_idx}_nve.txt
#   shuffle : bool : is the returned data shuffled
def load_data_set(file_name_prefix: str, data_idx: list, shuffle=False) -> tuple:
    potentials = []
    nve = []
    for index in data_idx:
        #print(data_set_name)
        file_name = file_name_prefix + "_" + str(index)
        nve.append(loadtxt(file_name + "_nve.txt", delimiter=','))

        potential = loadtxt(file_name + "_potential.txt", delimiter=',')
        shape = potential.shape
        potential = potential.reshape(1, shape[0], shape[1])
        potential = np.repeat(potential, nve[-1].shape[0], axis=0)
        potentials.append(potential)

    potentials = np.concatenate(potentials, axis=0)
    nve = np.concatenate(nve, axis=0)
    E = nve[:,0:1]
    target = nve[:,1:2]
    #print('iodata', E.shape, target.shape)

    if shuffle:
        idx = np.arange(E.shape[0])
        np.random.shuffle(idx)
        return ((potentials[idx], E[idx]), target[idx])
    else:
        return ((potentials, E), target)




# np.random.seed(12)
# n = 10
# # p = np.random.rand(n)
# # H = make_hamiltonian(p)
# # print(eigenvalue_count(H-np.eye(n)*0.5))
# # print(type(H))
#
# nve = np.array([[x/10, x] for x in range(1, 11)])
#
# for i in range(5):
#     save_data_set("stuff", np.random.rand(n, 2), nve, "data", "test")
#
# potentials, E, target = load_data_set("data", "test", 0, end=5)
#
# print(potentials.shape)
# print(E.shape)
# print(target.shape)
# print(E)
# print(target)
#print(H)
# print(np.finfo(np.float64))
#
# for i in range(9):
#     lu, d, perm = ldl(H-np.eye(n)*(2*i/10))
#     #print(H)
#     #print(d)
#     print(eigenvalue_count(H-np.eye(n)*(2*i/4)))
#
# lu, d, perm = ldl(H-np.eye(n)*4)
# print(d)

#print(p)
#u = compute_u(p)
#print(u)
#
# fig=plt.figure(figsize=(18, 16), dpi= 50, facecolor='w', edgecolor='k')
# plt.plot(p, color='blue')
# plt.plot(1/u, color='red')
# plt.show()
