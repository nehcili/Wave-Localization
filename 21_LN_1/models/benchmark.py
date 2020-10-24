
#################################################
# WARNING:
# Must also update the file benchmark.py
# in the folder "models"
# the file benchmark.py in the main
# directory is the newest version!
#################################################

import tensorflow as tf
import numpy as np
import os

#################################################
# constants
#################################################


# input:
#     p: potential domain_size
#     E: batch_size
#     here the batch_size is number of potential x number of eigenvalues
def weyl_single(p: np.ndarray, E: np.float32):
    p = 2*np.sqrt(np.maximum(E-p, 0.0))
    return np.sum(p)/2/np.pi

# This is the overload whose input machines those of ML models
# input:
#   input: tuple of potential and E where
#   potential is batch_size x domain_size x channel_size
#   E is batch_size x 1
#   pot_type: 0 = V and 1 = W
# rtype:
#   batch_size x 1
def weyl(input, pot_type=1):
    p, E = input

    p = p[:,:,pot_type]
    #print('pshape 0', p.shape)
    #print(E.shape)
    p = 2*tf.math.sqrt(tf.math.maximum(E-p, 0.0))
    #print('pshape 1', p.shape)
    p = tf.reduce_sum(p, axis=1)/2/np.pi
    #print('pshape 2', p.shape)
    return p[:, tf.newaxis]



# input:
#     p: domain_size or 1 x domain_size x 1
#     evs: batch_size
#     here the batch_size is number of potential x number of eigenvalues
def boxCount_single(p: np.ndarray, E: np.float32, c1=0.35, c2=1.25):
    if len(p.shape) == 1:
        p = p[np.newaxis,:, np.newaxis]

    E /= c2
    if E <= 1/p.shape[1]**2:
        E = 1/p.shape[1]**2

    box_size = int(np.round(max(1, 1/np.sqrt(E))))
    #print(p.shape)

    p = (E-p)
    p = tf.nn.max_pool1d(p, ksize=box_size, strides=box_size, padding='SAME')
    #p = p[0,:,0]

    return c1*np.sum(p >= 0)


# This is the overload whose input machines those of ML models
# input:
#   input: tuple of potential and E where
#   potential is batch_size x domain_size x channel_size, batch_size = number of potential x number of EV per potential
#   E is batch_size
#   pot_type: 0 = V and 1 = W
# rtype:
#   batch_size
def boxCount(input, pot_type=1, c1=0.35, c2=1.25):
    potentials, E = input
    t = np.empty(len(E))
    for i in range(len(t)):
        t[i] = boxCount_single(potentials[i:i+1,:,pot_type:pot_type+1], E[i], c1=c1, c2=c2)

    return t

class boxCounter(object):
    def __init__(self, c1=0.35, c2=1.25):
        self.c1 = c1
        self.c2 = c2

    def call(self, input, pot_type=1, c1=None, c2=None):
        if not c1:
            c1 = self.c1
        if not c2:
            c2 = self.c2

        return boxCount(input, pot_type=pot_type, c1=c1, c2=c2)

    # Input
    #   x: 1d np array > 0
    #   y: 1d np array
    # rtype float32
    def loss(self, x, y):
        x = x[y>0]
        y = y[y>0]
        # y = y[x>0]
        # x = x[x>0]

        if len(y) == 0 or len(x) == 0:
            return float('inf')

        xyavg = np.sum(np.log(x/y))/len(y)
        yxavg = np.sum(np.log(y/x))/len(y)

        return np.sum( (np.log(y/x)-yxavg)**2 )/len(y) + np.sum( (np.log(x/y)-xyavg)**2 )/len(y)
        #+ np.sum(np.log(y)**2)/len(y)
        # +


    # input:
    #   input: tuple of potential and E where
    #   potential is batch_size x domain_size x channel_size, batch_size = number of potential x number of EV per potential
    #   E is batch_size
    #   target: batch_size
    #   loss: function whose argument is target/prediction
    #   batch_size: int; how many to do SGD
    #   range: list of 2 element; range for c2
    #   shuffle: bool shuffled training?
    def fit(self, input, target,
            loss=None, batch_size=20, epoch=1, lr=0.1, decay=0.001,
            c2=None, shuffle=True, verbose=True):
        if not loss:
            loss = self.loss
        if not c2:
            c2 = self.c2


        potentials, E = input
        potentials = potentials[target > 0]
        E = E[target>0]
        target = target[target >0]
        # # Debug
        # old_p = potentials.copy()
        # old_E = E.copy()
        #
        # x = boxCount((potentials, E), c1=1, c2=1.25)
        # print('bad i', np.any(x==10))
        # for i in range(len(E)):
        #     if boxCount((potentials[i:i+1], E[i:i+1]), c1=1, c2=1.25) == 10:
        #         print('found bad i', i)
        #         print(boxCount((potentials[i:i+1], E[i:i+1]), c1=1, c2=1.25))
        # print()
        #
        #
        # idx = (target>0)*(x>0)
        # old_idx = idx.copy()
        # tpossitive = target>0
        # print('idx and tpossitive', all(idx == tpossitive))
        # print('same p?', np.all(old_p == potentials))
        # print('all x', all(x[idx] > 0))
        # print('all target', all(target[idx]>0))
        #
        # print('sum target and x', np.sum(target>0), np.sum(x>0), np.sum(idx))
        # print('first check', np.sum(target[idx]/x[idx])/len(x[idx]))
        # print('both > 0', x[idx].shape)
        # print('same idx', all(idx == old_idx))
        # # p = potentials[target > 0]
        # # e = E[target>0]
        # # t = target[target >0]
        # # p = potentials[idx]
        # # e = E[idx]
        # # t = target[idx]
        # p = old_p[idx]
        # e = old_E[idx]
        # t = target[idx]
        # print('all t', all(t>0))
        # print('e', e.shape)
        # # debug
        # y = boxCount((p, e), c1=1, c2=1.25)
        # print('second all y', all(y>0))
        # print('commute', np.all(x[idx] == y))
        # print(sorted(x))
        # print(boxCount((potentials[idx][31:32], (E[idx])[31:32]), c1=1, c2=1.25))
        # print((potentials[idx][31:32]).shape)
        #
        # print(y[31])
        # print('first check, again', np.sum(target[idx]/x[idx])/len(x[idx]))
        # #idx = (x>0)
        # #print('idx all', all(idx), all(target>0))
        # #print(x.shape, target.shape, (x>0).shape)
        # #t = t[idx]
        # #x = x[idx]
        # print('second check', self.c2, np.sum(t/y)/len(y))
        # x = boxCount((potentials, E), c1=1, c2=1.25)
        # print('first check, again again', np.sum(target[idx]/x[idx])/len(x[idx]))
        #
        #
        #
        # print(t.shape)
        #
        # return


        # min_loss = float('inf')
        # cur_loss = loss(target, self.call(input, c1=1, c2=c2))

        if verbose:
            print("Training DFM boxCounter for c1 and c2.")

        for i in range(epoch):
            if shuffle:
                idx = np.random.permutation(len(E))
                potentials = potentials[idx]
                E = E[idx]
                target = target[idx]

            cum_loss = 0
            for j in range(0, len(E), batch_size):
                t = target[j:j+batch_size]
                p = potentials[j:j+batch_size,:,:]
                e = E[j:j+batch_size]

                cur_loss = loss(t, self.call((p,e), c1=1, c2=c2))
                loss_p = loss(t, self.call((p,e), c1=1, c2=c2+lr))
                loss_m = loss(t, self.call((p,e), c1=1, c2=c2-lr))

                if cur_loss <= loss_p and cur_loss <= loss_m:
                    # continue
                    c2 += (np.random.rand()-0.5) * lr**4
                    cum_loss += cur_loss
                    #cur_loss = loss(t, self.call((p,e), c1=1, c2=c2))
                elif loss_p <= loss_m:
                    c2 += lr
                    cum_loss = +loss_p
                else:
                    c2 -= lr
                    cum_loss = +loss_m

                lr *= (1-decay)

            if verbose:
                print("Epoch {}: \tloss: {}".format(i, cum_loss/((len(E)+batch_size)//batch_size)))

        x = self.call((potentials, E), c1=1, c2=c2)
        #print(x)
        # x = x[target>0]
        # target = target[target>0]
        target = target[x>0]
        x = x[x>0]

        self.c1 = np.sum(target/x)/len(x) #+ 1/np.sum(x/target)/len(x))/2
        self.c2 = c2

    def save(self,name, path="models"):
        name = os.path.join(path, name)
        with open(name, 'w') as f:
            f.write('{},{}'.format(self.c1, self.c2))

    @classmethod
    def load_model(self, name, path="models"):
        name = os.path.join(path, name)
        with open(name, 'r') as f:
            s = f.read()

        s = s.split(',')
        return float(s[0]), float(s[1])



# # test
# import iodata
# #p2, E2, t2 = iodata.load_data_set('data', 'discrete_1000', 2)
# #p3, E3, t3 = iodata.load_data_set('data', 'discrete_1000', 3)
# p, E, t = iodata.load_data_set('data', 'discrete_1000', 0,20)
#
# #print('correct load', np.all(np.concatenate([t2,t3]) == t))
#
# x = (p, E)
# BC = boxCounter(c1=0.35, c2=1.25)
# #BC.fit((p2, E2), t2, lr=0.01, decay=0.01, batch_size=100, epoch=0)
# #BC.fit((p3, E3), t3, lr=0.01, decay=0.01, batch_size=100, epoch=0)
# BC.fit(x, t, lr=0.01, decay=0.01, batch_size=500, epoch=20)
# print('BC c1 c1', BC.c1, BC.c2)
#
#
#
# w = weyl(x)[:,:,1]
# b0 = boxCount(x, c1=0.35, c2=1.25)
# b = BC.call(x)
# idx = (t>0)*(b>0)
# idx0 = (t>0)*(b0>0)
# print('final b check', np.sum(t[idx]/b[idx])/np.sum(idx))
# print('final b0 check', np.sum(t[idx0]/b0[idx0])/np.sum(idx0))
#
# #print(b0)
#
# # Plot
# import matplotlib.pyplot as plt
# fig=plt.figure(figsize=(18, 16), dpi= 50, facecolor='w', edgecolor='k')
# plt.scatter(E, t, color='blue')
# #plt.scatter(E, w, color='red')
# plt.scatter(E, b, color='green')
# plt.scatter(E, b0, color='gray')
# plt.show()
#
#
#
# fig=plt.figure(figsize=(18, 16), dpi= 50, facecolor='w', edgecolor='k')
# #plt.scatter(E, t, color='blue')
# #plt.scatter(E, w, color='red')
# plt.scatter(E, t/b, color='green')
# plt.scatter(E, t/b0, color='gray')
# plt.show()
#
# BC.save('discrete_1000_boxCounter_0')
# BC.load_model('discrete_1000_boxCounter_0')
