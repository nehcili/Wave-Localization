#import numpy as np

# tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from benchmark import weyl

class cnn_landscape(Model):
    def __init__(self, params):
        name = params['model_name']
        super(cnn_landscape, self).__init__(name=name)

        # choice of potential
        self.pchoice = params['pchoice']

        # residual
        self.res = keras.Sequential(name='residual')
        self.res.add(layers.Conv1D(self.pchoice[1]-self.pchoice[0], 20, strides=1, padding='same'))
        self.res.add(layers.ELU())
        self.res.add(layers.Conv1D(self.pchoice[1]-self.pchoice[0], 20, strides=1, padding='same'))
        self.res.add(layers.LeakyReLU())
        self.res.add(layers.Dropout(rate=0.1))

        # embedding layers
        self.emb = keras.Sequential(name='embedding')
        self.emb.add(layers.BatchNormalization())
        self.emb.add(layers.Conv1D(5, 50, strides=5, padding='same'))
        self.emb.add(layers.ELU())
        self.emb.add(layers.Dropout(rate=0.1))
        self.emb.add(layers.Conv1D(5, 50, strides=5, padding='same'))
        self.emb.add(layers.LeakyReLU())
        self.emb.add(layers.Dropout(rate=0.1))

        # reduction of input
        self.conv1 = layers.Conv1D(5, 50, strides=25, padding='same')


        self.weighters = keras.Sequential(name='integral_weight')
        self.weighters.add(layers.Conv1D(10, 20, strides=10, padding='same'))
        self.weighters.add(layers.ELU())
        self.weighters.add(layers.Dropout(rate=0.1))
        self.weighters.add(layers.Conv1D(20, 20, strides=10, padding='same'))
        self.weighters.add(layers.LeakyReLU())
        self.weighters.add(layers.Dropout(rate=0.1))
        self.weighters.add(layers.Conv1D(30, 4, strides=4, padding='same'))
        self.weighters.add(layers.Flatten())

        self.out_layer1 = keras.Sequential(name='out_layer1')
        self.out_layer1.add(layers.Dense(30))
        self.out_layer1.add(layers.LeakyReLU())
        self.out_layer1.add(layers.Dense(30))
        self.out_layer1.add(layers.LeakyReLU())

        self.out_layer2 = keras.Sequential(name='out_layer2')
        self.out_layer2.add(layers.Dense(10))
        self.out_layer2.add(layers.LeakyReLU())
        self.out_layer2.add(layers.Dense(1))

    def call(self, input, training=False):
        weyl_base = weyl(input)

        p, E = input

        # selecting the right potential
        p = p[:,:,self.pchoice[0]:self.pchoice[1]]

        # reshape E to have the same shape as p
        resized_E = (E[:,:,tf.newaxis]+0*p)
        resized_E = resized_E[:,:,:1]

        # pE = p and E
        pE = tf.concat([p, resized_E], axis=2)

        # compute features - this is the F(dW, W) in front of \int f_{FD}'
        p = self.res(p)+p
        p = self.emb(p, training=training)

        # conv down potential
        pE = self.conv1(pE)

        # features as weight * conved down potential = F(dW, W) \int f_{FD}''
        #print(p.shape, pE.shape)
        p = pE*p

        # weighted sum = integration
        p = self.weighters(p, training=training)

        pp = self.out_layer1(p)
        p = self.out_layer2(pp)+p # residual layer

        #print(p.shape)

        return tf.nn.softplus(weyl_base + p)



def get_model(params):
    return cnn_landscape(params)

##############################################
# Training params template
##############################################

# params = {
#     # model params
#     "model_class" : "cnn_dense_0",
#     "model_name" : "cnn_dense_0",
#     "model_folder_path" : "models",
#     "pchoice" : [1,2], # [0,1] = V, [1,2] = W, [0,2] = [V,W]
#
#     # data params
#     "ds_folder" : "data",
#     "ds_name" : "discrete_10000",
#     "train_idx" : list(range(100)), # np 1D array
#     "val_idx" : list(range(50)),
#
#     # io/checkpointing
#     "ckpt_filename" : "cp-{epoch:04d}-{val_loss:.2f}.ckpt",
#     "save_freq" : 'epoch',
#     "save_best_only" : False, # only save if val_loss improves
#
#     # training params
#     "epochs" : 100,
#     "train_batch_size" : 25,
#     "val_batch_size" : 25,
#     "shuffle" : True, # shuffle per epoch
#     "loss_description" : "mean squared error",
#     "loss" : "mse",
#     "optimizer_description" : "adam",
#     "learning_rate" : 0.0001, # have to write this in the learning_rate of optimizer
#     "clipnorm" : 0.2,
#     "optimizer" : tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=0.2), # have to write this in "learning_rate"
#     "verbose" : 2,
#
#     # mlflow content
#     # server_uri. Initiate by
#     # mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0
#     "server_uri" : "http://0.0.0.0:5000",
#     "every_n_iter" : 1 # ml logs every every_n_iter epoch
# }
