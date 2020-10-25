import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
import numpy as np
from numpy import loadtxt, savetxt
import matplotlib.pyplot as plt
# import mlflow


# test data gen
# C to F conversion
# return shape
#   x: batch_size x 1
#   y: batch_size
def gen(batch_size):
    x = 100*np.random.rand(batch_size)-50
    y = 9*x/5+32+ 10*np.random.standard_normal(batch_size)
    return x[:,np.newaxis], y

# IO data
def save(fname, x, y):
    savetxt('x_' + fname + '.csv', x, delimiter=',')
    savetxt('y_' + fname + '.csv', y, delimiter=',')

def load(fname):
    x = loadtxt('x_' + fname + '.csv', delimiter=',')
    y = loadtxt('y_' + fname + '.csv', delimiter=',')
    return x[:, np.newaxis], y

def gen_save(fname, batch_size):
    x, y = gen(batch_size)
    save(fname, x, y)


# training
def train(model, epochs, batch_size, server_uri, ftrain, fval):
    # date
    x_train, y_train = load(ftrain)
    x_val, y_val = load(fval)

    # building the model before fit
    model(x_val)

    # ml flow
    # remote_server_uri = server_uri  # set to your server URI
    # mlflow.set_tracking_uri(remote_server_uri)
    # mlflow.tensorflow.autolog(every_n_iter=1)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mse')
    history = model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(x_val, y_val))

    return model, history

def graph(model):
    x, y = gen(500)
    fig=plt.figure(figsize=(18, 16), dpi= 50, facecolor='w', edgecolor='k')
    plt.scatter(x, y, color="blue", label="True CF conversion")
    y_pred = model(x)
    plt.scatter(x, y_pred, color="orange", label="Predicted CF conversion")
    plt.show()

# generate data
gen_save('train', 10000)
gen_save('val', 200)

# model
# hello world model
model = Sequential(name='docker_test_model')
model.add(layers.Dense(1))

# mlflow:
# mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0
model, history = train(
    model,
    30,
    200,
    "http://0.0.0.0:5000",
    'train',
    'val'
)
model.save('docker_test_model.h5')
model = tf.keras.models.load_model('docker_test_model.h5')
graph(model)
