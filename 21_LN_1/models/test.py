#import numpy as np

# tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from benchmark import weyl

class test(Model):
    def __init__(self, params):
        super(test, self).__init__(name=params['model_name'])

        self.pchoice = params['pchoice']
        self.out_layer = layers.Dense(1)

    def call(self, input):
        weyl_base = weyl(input)

        p, E = input


        # selecting the right potential
        p = p[:,:,self.pchoice[0]]
        p = p-E
        p = self.out_layer(p)

        return tf.nn.softplus(weyl_base + p)



def get_model(params):
    return test(params)


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
