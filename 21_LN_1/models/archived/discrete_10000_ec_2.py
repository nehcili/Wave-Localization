#import numpy as np

# tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
#from benchmark import weyl

class discrete_10000_ec_2(Model):
    def __init__(self, params):
        name = params['model_name']
        super(discrete_10000_ec_2, self).__init__(name=name)

        # float
        dropout_rate = params['dropout_rate']

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
        self.decider.add(layers.LeakyReLU())
        self.decider.add(layers.Dropout(rate=dropout_rate))
        self.decider.add(layers.Conv1D(8, 6, strides=1, padding='same'))

        self.counters = []
        for i in range(1,8):
            model = keras.Sequential()
            model.add(layers.Conv1D(8, 2**i, 2**(i-1), padding='same'))
            model.add(layers.ELU())
            model.add(layers.Dropout(rate=dropout_rate))
            pool_size = (10000 + 2**(i-1)-1) // 2**(i-1)
            model.add(layers.LeakyReLU())
            self.counters.append(model)

        self.conv_down = layers.Dense(1)

        self.weighters = keras.Sequential()
        self.weighters.add(layers.BatchNormalization())
        self.weighters.add(layers.Dense(21))
        self.weighters.add(layers.ELU())
        self.weighters.add(layers.Dropout(rate=dropout_rate))
        self.weighters.add(layers.Dense(14))
        self.weighters.add(layers.LeakyReLU())
        self.weighters.add(layers.Dropout(rate=dropout_rate))
        self.weighters.add(layers.Dense(7))
        self.weighters.add(layers.Activation('softmax'))

        self.out_layer = layers.Dense(1)

    def call(self, input, training=False):
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
        out = [self.conv_down(tf.math.reduce_sum(model(p, training=training), axis=1)) for model in self.counters]
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
        return tf.nn.softplus(p)



def create_model(params, data_idx=[0]):
    data_folder_path = params["data_folder_path"]
    model_name = params["model_name"]
    data_set_name = params["data_set_name"]

    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import iodata

    #print()
    p, E, _ = iodata.load_data_set(data_folder_path, data_set_name, data_idx)
    input = (p, E)
    model = discrete_10000_ec_2(params)
    output = model(input)
    return model

##############################################
# Training params template
##############################################

# train_params = {
#     # model params
#     "model_class" : "discrete_10000_ec_2",
#     "model_name" : "ec2_0",
#     "model_folder_path" : "models",
#     "dropout_rate" : 0.1,
#     "pchoice" : [1,2], # [0,1] = V, [1,2] = W, [0,2] = [V,W]
#     "model_description" : "not weyl assisted"
#
#     # data params
#     "data_folder_path" : "data",
#     "data_set_name" : "discrete_10000",
#     "data_idx" : list(range(700)), # np 1D array
#
#     # training params
#     "meta_epoch" : 5,
#     "epochs" : 2,
#     "batch_size" : 125,
#     "validation_split" : 0,
#     "shuffle" : True, # shuffle per epoch
#     "loss_description" : "mean squared error",
#     "loss" : "mse",
#     "optimizer_description" : "adam",
#     "learning_rate" : 0.01, # have to write this in the learning_rate of optimizer
#     "clipnorm" : 10000,
#     "optimizer" : tf.keras.optimizers.Adam(learning_rate=0.01, clipnorm=10000), # have to write this in "learning_rate"
#     "data_set_cap" : 20, # avoid RAM outage, 20 is a good number of my horrible computer
#     "max_to_keep" : 100, # used in CheckpointManager, None = keep all
#     "verbose" : 2,
#     "blow_up_multiplier" : 5,
#
#     # mlflow content
#     # server_uri. Initiate by
#     # mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0
#     "server_uri" : "http://0.0.0.0:5000",
# }

# model = create_model('tf_save_load_test', 'discrete_10000')
# model.save('models/tf_save_load_test')


# model = create_model("test")
# print(model.summary())
#
# model.save_weights("models/discrete_10000_EC0/test")
# model.load_weights("models/discrete_10000_EC0/test")
