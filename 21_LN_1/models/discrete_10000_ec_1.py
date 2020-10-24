#import numpy as np

# tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from benchmark import weyl

class discrete_10000_ec_1(Model):
    def __init__(self, params):
        name = params['model_name']
        super(discrete_10000_ec_1, self).__init__(name=name)

        # float
        dropout_rate = 0.1

        # list with 2 element: start and end
        # [0:1] = V
        # [1:2] = W
        # [0:2] =  V and W
        self.pchoice = params['pchoice']

        self.decider = keras.Sequential()
        self.decider.add(layers.Conv1D(6, 2, strides=1, padding='same'))
        self.decider.add(layers.ELU())
        self.decider.add(layers.Dropout(rate=dropout_rate))
        self.decider.add(layers.Conv1D(7, 4, strides=1, padding='same'))
        self.decider.add(layers.ELU(alpha=2.0))
        self.decider.add(layers.Dropout(rate=dropout_rate))
        self.decider.add(layers.Conv1D(8, 6, strides=1, padding='same'))

        self.counters = []
        for i in range(1,8):
            model = keras.Sequential()
            model.add(layers.Conv1D(8, 2**i, 2**(i-1), padding='same'))
            model.add(layers.ELU(alpha=4.0))
            model.add(layers.Dropout(rate=dropout_rate))
            pool_size = (10000 + 2**(i-1)-1) // 2**(i-1)
            model.add(layers.AveragePooling1D(pool_size=pool_size))
            model.add(layers.LeakyReLU())
            self.counters.append(model)

        self.conv_down = layers.Dense(1)

        self.weighters = keras.Sequential()
        self.weighters.add(layers.BatchNormalization())
        self.weighters.add(layers.Dense(21))
        self.weighters.add(layers.ELU(alpha=8.0))
        self.weighters.add(layers.Dropout(rate=dropout_rate))
        self.weighters.add(layers.Dense(14))
        self.weighters.add(layers.LeakyReLU())
        self.weighters.add(layers.Dropout(rate=dropout_rate))
        self.weighters.add(layers.Dense(7))
        self.weighters.add(layers.Activation('softmax'))

        self.out_layer = layers.Dense(1)

    def call(self, input, training=False):
        weyl_base = weyl(input)

        p, E = input

        # selecting the right potential
        p = p[:,:,self.pchoice[0]:self.pchoice[1]]

        # reshape E to have the same shape as p
        resized_E = (E[:,:,tf.newaxis]+0*p)
        resized_E = resized_E[:,:,:1]

        p = tf.concat([p, resized_E], axis=2)
        #print(0, p.shape)
        p = self.decider(p, training=training)
        #print(1, p.shape)
        out = [self.conv_down(model(p, training=training)[:,0,:]) for model in self.counters]
        p = tf.concat(out, axis=1)
        #print(2, p.shape)

        #print(p.shape)
        box_size = tf.minimum(1/tf.math.sqrt(E), 10000)
        E = tf.concat([E, box_size], axis=1)
        #print(3, E.shape)
        E = self.weighters(E, training=training)
        #print(4, E.shape)

        #print(E.shape)
        p = p*E
        #print(5, p.shape)

        p = self.out_layer(p)
        return tf.nn.softplus(weyl_base + p)/2



def get_model(params):
    return discrete_10000_ec_1(params)

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
