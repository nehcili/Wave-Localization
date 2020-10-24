#import numpy as np

# tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from benchmark import weyl

class cnn_dense_0(Model):
    def __init__(self, params):
        super(cnn_dense_0, self).__init__(name=params['model_name'])

        self.pchoice = params['pchoice']

        self.internal = tf.keras.Sequential(name=params['model_name']+'_layers')
        self.internal.add(layers.BatchNormalization())
        self.internal.add(layers.Conv1D(4, 20, strides=10, padding='same'))
        self.internal.add(layers.ELU())
        self.internal.add(layers.Dropout(0.1))

        self.internal.add(layers.BatchNormalization())
        self.internal.add(layers.Conv1D(8, 20, strides=10, padding='same'))
        self.internal.add(layers.ELU(alpha=2.0))
        self.internal.add(layers.Dropout(0.1))

        self.internal.add(layers.BatchNormalization())
        self.internal.add(layers.Conv1D(12, 20, strides=10, padding='same'))
        self.internal.add(layers.ELU(alpha=4.0))
        self.internal.add(layers.Dropout(0.1))

        self.internal.add(layers.BatchNormalization())
        self.internal.add(layers.Conv1D(32, 10, strides=10, padding='same'))
        self.internal.add(layers.ELU(alpha=8.0))
        self.internal.add(layers.Dropout(0.1))

        self.internal.add(layers.Flatten())

        self.internal.add(layers.Dense(32))
        self.internal.add(layers.ELU(alpha=10.0))
        self.internal.add(layers.Dropout(0.05))

        self.internal.add(layers.Dense(16))
        self.internal.add(layers.ELU(alpha=10.0))

        self.internal.add(layers.Dense(1))

    def call(self, input):
        weyl_base = weyl(input)

        p, E = input


        # selecting the right potential
        p = p[:,:,self.pchoice[0]:self.pchoice[1]]
        E = E[:,:,tf.newaxis]
        E = 0.0*p+E
        p = tf.concat([p, E], axis=2)
        p = self.internal(p)

        return tf.nn.softplus(weyl_base + p)



def get_model(params):
    return cnn_dense_0(params)

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
#     "train_idx" : list(range(500)), # np 1D array
#     "val_idx" : list(range(50)),
#
#     # io/checkpointing
#     "ckpt_filename" : "cp-{epoch:04d}-{val_loss:.2f}.ckpt",
#     "save_freq" : 'epoch',
#     "save_best_only" : False, # only save if val_loss improves
#
#     # training params
#     "epochs" : 100,
#     "train_batch_size" : 50,
#     "val_batch_size" : 50,
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
