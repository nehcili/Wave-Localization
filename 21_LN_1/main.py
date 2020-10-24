from train import train
import iomodel
import tensorflow as tf


###############################################################
# Input variables
###############################################################
#def percent_mse(y_true, y_pred):
#    return tf.reduce_mean((y_true-y_pred)**2/(0.1*y_true**2 +1))

# lr schedule
def lr_schedule(epoch, lr):
    if epoch < 10:
        return 0.01
    elif epoch < 20:
        return 0.001
    elif epoch > 45:
        return lr * 1.5
    else:
        return lr * 0.8


params = {
    # model params
    "model_class" : "cnn_dense",
    "model_name" : "cnn_dense",
    "model_folder_path" : "models",
    "pchoice" : [0,1], # [0,1] = V, [1,2] = W, [0,2] = [V,W]

    # data params
    "ds_folder" : "data",
    "ds_name" : "discrete_10000",
    "train_idx" : list(range(500)), # np 1D array
    "val_idx" : list(range(50)),

    # io/checkpointing
    "ckpt_filename" : "cp-{epoch:04d}-{val_loss:.2f}.ckpt",
    "save_freq" : 'epoch',
    "save_best_only" : False, # only save if val_loss improves

    # lr schedule
    'lr_schedule' : lr_schedule,

    # training params
    "epochs" : 50,
    "train_batch_size" : 25,
    "val_batch_size" : 25,
    "shuffle" : True, # shuffle per epoch
    "loss_description" : "mean squared error",
    "loss" : "mse",
    "optimizer_description" : "adam",
    "learning_rate" : 0.001, # have to write this in the learning_rate of optimizer
    "clipnorm" : 0.1,
    "optimizer" : tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=0.1), # have to write this in "learning_rate"
    "verbose" : 2,

    # mlflow content
    # server_uri. Initiate by
    # mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0
    "server_uri" : "http://0.0.0.0:5000",
    "every_n_iter" : 1 # ml logs every every_n_iter epoch
}


###############################################################
# Training
###############################################################

# cnn_landscape
# print("\n CNN landscape\n")
# print("\n# V potential\n")
# params['pchoice'] = [0,1]
# params['model_name'] = 'cnn_landscape_V'
# model = iomodel.create_model(params)
# model.summary()
# train(model, params)
#
# print("\n# W potential\n")
# params['pchoice'] = [1,2]
# params['model_name'] = 'cnn_landscape_W'
# model = iomodel.create_model(params)
# model.summary()
# train(model, params)
#
# print("\n# V and W potential\n")
# params['pchoice'] = [0,2]
# params['model_name'] = 'cnn_landscape_VW'
# model = iomodel.create_model(params)
# model.summary()
# train(model, params)

# discrete_10000_ec_2
print("\n# model class discrete_10000_ec_2\n")
print("\n# VW potential\n")
params['pchoice'] = [0,1]
params['model_class'] = 'discrete_10000_ec_2'
params['model_name'] = 'discrete_10000_ec_2_VW'
model = iomodel.create_model(params)
model.summary()
train(model, params)


# cnn_dense
def lrs(epoch, lr):
    if epoch < 10:
        return 0.001
    elif epoch < 20:
        return 0.0001
    elif epoch > 45:
        return lr * 1.5
    else:
        return lr * 0.8

params['lr_schedule'] = lrs

print('\n# model class cnn_dense\n')
print("\n# V potential\n")
params['model_class'] = 'cnn_dense_0'
params['pchoice'] = [0,1]
params['model_name'] = 'cnn_dense_0_V'
model = iomodel.create_model(params)
model.summary()
train(model, params)



print("\n# W potential\n")
params['pchoice'] = [1,2]
params['model_name'] = 'cnn_dense_0_W'
model = iomodel.create_model(params)
model.summary()
train(model, params)

print("\n# V and W potential\n")
params['pchoice'] = [0,2]
params['model_name'] = 'cnn_dense_0_VW'
model = iomodel.create_model(params)
model.summary()
train(model, params)



###############################################################
# Visual Analysis
###############################################################
visualize = False
if visualize:
    import NVEplot as nplt
    import numpy as np
    #status = checkpoint.restore(manager.latest_checkpoint)

    for i in np.random.randint(0, high=499, size=10):
        nplt.plot_model_pred(model, "discrete_10000", i, data_folder_path='data/discrete_10000/train/')

###############################################################
# Debug
###############################################################

#from iodata import load_data_set
#from benchmark import weyl
#import os



# potentials, E, target = load_data_set(
#     "data",
#     "discrete_10000_uniform",
#     [48]
# )

# model_path = os.path.join(train_params["model_folder_path"], train_params["model_name"], "")
# model = create_model(train_params["model_name"], train_params["data_set_name"])
# checkpoint = tf.train.Checkpoint(optimizer=train_params["optimizer"], model=model)
# manager = tf.train.CheckpointManager(checkpoint, directory=model_path, max_to_keep=train_params["max_to_keep"])
# status = checkpoint.restore(manager.latest_checkpoint)

#model = tf.keras.models.load_model('models/tf_save_load_test')


#
# Plot
# import matplotlib.pyplot as plt
# import numpy as np
# # weyl prediction
#
# #potentials *= 0
# w = weyl((potentials, E), pot_type=1).numpy()
# # model prediction
# #print(E.shape)
# #print(potentials.shape)
# b = model([potentials, E]).numpy()
#
# #print(w.shape, b.shape)
# # #print(type(w), type(b))
# # print(potentials.shape)
# # print('W min:', min(potentials[0,:,1]))
# # print(potentials[0,:100,1])
#
# fig=plt.figure(figsize=(18, 16), dpi= 50, facecolor='w', edgecolor='k')
# #plt.plot(list(range(100)), potentials[0,:100,0], color='red')
# #plt.plot(list(range(100)), potentials[0,:100,1], color='blue')
# plt.scatter(E, w, color='gray')
# plt.scatter(E, target, color='red')
# plt.scatter(E, np.maximum(0, w+b), color='green')
# plt.show()
