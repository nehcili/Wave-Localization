import numpy as np

# tensorflow
import tensorflow as tf
from tensorflow.keras import layers, Model
from benchmark import weyl


class testModel(Model):
    def __init__(self):
        super(testModel, self).__init__(name='')

        self.convs = []
        self.flatten = tf.keras.layers.Flatten()

        for i in range(3):
            self.convs.append(layers.Conv1D(3*(i+1), 10, strides=1, padding='same'))
            self.convs.append(layers.LeakyReLU(alpha=0.01))

    def call(self, input):
        potentials, E = input
        potentials = potentials - E[:,:,tf.newaxis]


        for i in range(6):
            potentials = self.convs[i](potentials)

        potentials = self.flatten(potentials)
        potentials = tf.math.reduce_sum(potentials, axis=1)
        #print(potentials.shape)

        return potentials + weyl(input)

pinput = tf.keras.Input(shape=(10000, 2))
Einput = tf.keras.Input(shape=(1))
input = (pinput, Einput)

model = testModel()
output = model(input)

model.save("models/discrete_10000_testModel")
