############################################
# Libraries
############################################
# io
from iodata import load_data_set
import os
from models.benchmark import weyl
import mlflow

# tensorflow
import tensorflow as tf

# numpy
import numpy as np

############################################
# Data Generator
############################################

class NVEGenerator(tf.keras.utils.Sequence):
    def __init__(
            self,
            data_folder_path: str,
            data_set_name: str,
            data_idx: list,
            meta_batch_size: int,
            transformer=None,
            shuffle=True,
        ):
        super(NVEGenerator, self).__init__()
        self.data_idx = data_idx
        self.file_name_prefix = os.path.join(data_folder_path, data_set_name)
        self.meta_batch_size = meta_batch_size
        self.shuffle = shuffle

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.data_idx)

    def __len__(self):
        return int(np.ceil(len(self.data_idx) / float(self.meta_batch_size)))

    def __getitem__(self, idx):
        idx = self.data_idx[idx*self.meta_batch_size: (idx+1)*self.meta_batch_size]
        return load_data_set(self.file_name_prefix, idx, shuffle=self.shuffle)

# Testing
# G = NVEGenerator('data', 'discrete_10000', list(range(100)), 20)
#
# for x, y in G:
#     p, E = x
#     print(p.shape, E.shape, y.shape)
#     print(y[:10])


############################################
# Training
############################################

 # __init__(
 #        self,
 #        data_folder_path: str,
 #        data_set_name: str,
 #        data_idx: list,
 #        meta_batch_size: int,
 #        transformer=None,
 #        shuffle=True
 #    ):
def train(model, params):
    # mlflow logging
    # mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0
    remote_server_uri = params["server_uri"]  # set to your server URI
    mlflow.set_tracking_uri(remote_server_uri)  # or set the MLFLOW_TRACKING_URI in the env

    exp_name = params["model_name"]
    mlflow.set_experiment(exp_name)

    # auto logging
    mlflow.tensorflow.autolog(every_n_iter=params['every_n_iter'])

    print("# mlflow initialized")


    # Load model weights
    ckpt_path = os.path.join(params["model_folder_path"], params["model_name"], params["ckpt_filename"])
    ckpt_dir = os.path.dirname(ckpt_path)
    latest = tf.train.latest_checkpoint(ckpt_dir)
    if not latest:
        model.save_weights(ckpt_path.format(epoch=0, val_loss=0))
        latest = tf.train.latest_checkpoint(ckpt_dir)
    else:
        model.load_weights(latest)

    print("# Model loaded from: {}".format(latest))


    # Initializing train and test data set generator
    train_ds_path = os.path.join(params['ds_folder'], params['ds_name'], "train")
    val_ds_path = os.path.join(params['ds_folder'], params['ds_name'], "validation")

    # deciding a full load or use NVEGenerator API to load
    # data on the fly in pieces (to avoid memory overflow)
    if params["full_load"]:
        train_ds_prefix = os.path.join(train_ds_path, params['ds_name'])
        train_ds = tf.data.Dataset.from_tensor_slices(
            load_data_set(train_ds_prefix, params['train_idx'], shuffle=False))
        train_ds = train_ds.batch(params['train_batch_size'])

        val_ds_prefix = os.path.join(val_ds_path, params['ds_name'])
        val_ds = tf.data.Dataset.from_tensor_slices(
            load_data_set(val_ds_prefix, params['val_idx'], shuffle=False))
        val_ds = val_ds.batch(params['val_batch_size'])
    else:
        train_ds = NVEGenerator(train_ds_path, params['ds_name'],
            params['train_idx'], meta_batch_size=params['train_batch_size'],
            shuffle= params['shuffle'])
        val_ds = NVEGenerator(val_ds_path, params['ds_name'],
            params['val_idx'], meta_batch_size=params['val_batch_size'],
            shuffle= params['shuffle'])


    print("# Training and validation generator initialized")


    # Compiling model
    model.compile(
        optimizer=params['optimizer'],
        loss=params['loss'],
        #metrics=params['metrics']
    )

    print("# Model compiled")

    # callbacks
    # check point
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path,
        verbose=1,
        save_weights_only=True,
        mode='min',
        save_freq=params['save_freq'], # saves weights every save_freq times a ds generator is called
        monitor='val_loss',
        save_best_only=params['save_best_only']
    )
    # lr schedule
    callbacks_list = [cp_callback]
    if 'lr_schedule' in params:
        callbacks_list.append(tf.keras.callbacks.LearningRateScheduler(
                params['lr_schedule'], verbose=0
            )
        )

    print("# Checkpoint callback initialized")


    # latest epoch
    latest_epoch = latest.find('cp-')
    latest_epoch = int(latest[latest_epoch+3:latest_epoch+7])+1

    print("# continue training from epoch", latest_epoch)

    # Training
    # Note:
    # do not specify batch_size if data is generator or dataset
    # shuffle is ignored if data set is a generator. shuffle by epoch end.
    history = model.fit(
        x=train_ds,
        epochs=params['epochs'],
        verbose=params["verbose"],
        callbacks=callbacks_list,
        validation_data=val_ds,
        shuffle=True,
        initial_epoch=latest_epoch,
    )

    return model, history


















# ############################################
# # Old Training Functions
# ############################################
#
#
# # Helper functions
# # it is assumed that end - start <= 5
# # input:
# #   model : tf.keras.Model
# #   train_params : dict - training parameters
# #   model_folder_path : string : path of folder that contains the model
# #   data_folder_path : string : path of the folder that contains data
# def train_model_on_batch(model, train_params, model_folder_path="models", data_folder_path="data"):
#     if len(train_params["data_idx"]) > train_params["data_set_cap"]:
#         print("Too many datasets specified. Might not be able to load into memory.")
#         return None, None
#
#     potentials, E, target = load_data_set(
#         data_folder_path,
#         train_params['data_set_name'],
#         train_params['data_idx']
#     )
#
#     x = (potentials, E)
#     #print('x shape', potentials.shape, E.shape, target.shape)
#     y = target # - weyl(x)
#     #print(y)
#     #print(target.shape, weyl(x).shape)
#     #print('target shape', y.shape)
#
#     model.compile(
#         optimizer=train_params['optimizer'],
#         loss=train_params['loss']
#     )
#
#     history = model.fit(
#         x=x,
#         y=y,
#         epochs=train_params['epochs'],
#         batch_size=train_params['batch_size'],
#         validation_split=train_params['validation_split'],
#         verbose=train_params["verbose"],
#         shuffle=True,
#     )
#
#     return model, history
#
# def train(model, params):
#     # loading model
#     model_path = os.path.join(params["model_folder_path"], params["model_name"], "")
#     data_path = os.path.join(params["data_folder_path"], params["data_set_name"])
#
#     checkpoint = tf.train.Checkpoint(optimizer=params["optimizer"], model=model)
#     manager = tf.train.CheckpointManager(checkpoint, directory=model_path, max_to_keep=params["max_to_keep"])
#     status = checkpoint.restore(manager.latest_checkpoint)
#
#     if params['meta_epoch'] == 0:
#         return
#
#     # mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0
#     remote_server_uri = params["server_uri"]  # set to your server URI
#     mlflow.set_tracking_uri(remote_server_uri)  # or set the MLFLOW_TRACKING_URI in the env
#
#     exp_name = params["model_name"]
#     mlflow.set_experiment(exp_name)
#
#     # log data set range
#     mlflow.log_param('data_set_range', [params['data_idx'][0], params['data_idx'][-1]+1])
#
#     # auto logging
#     # disabled, too verbose
#     if params['autolog']:
#         mlflow.tensorflow.autolog()
#
#     # initialize parameter to pass to train_model_on_batch
#     data_idx_size = (len(params["data_idx"]) + params["data_set_cap"] - 1)//params["data_set_cap"]
#     train_param_individual = params.copy()
#     data_idx = train_param_individual["data_idx"].copy()
#
#
#     # initialize epoch training variables
#     prev_loss = float('inf')
#     for i in range(params["meta_epoch"]):
#         print("# Meta epoch {}/{}: ".format(i, params["meta_epoch"]))
#         data_idx = np.random.permutation(data_idx)
#
#
#         for j in range(data_idx_size):
#             print("## Meta epoch {}.{}/{}.{}".format(i, j, params["meta_epoch"], data_idx_size))
#             train_param_individual["data_idx"] = data_idx[j*params["data_set_cap"]:(j+1)*params["data_set_cap"]]
#             model, history = train_model_on_batch(model,
#                     train_param_individual,
#                     model_folder_path="models",
#                     data_folder_path="data"
#                 )
#
#             # if blow up
#             cur_loss = np.average(history.history['loss'])
#             if cur_loss > train_param_individual['blow_up_multiplier']*prev_loss:
#                 print("Warning: loss exploded at {} at {}x the previous loss".format(cur_loss, \
#                     train_param_individual['blow_up_multiplier']))
#                 ckpt = manager.latest_checkpoint
#                 print("Model is retored to previous ckpt: {}".format(ckpt))
#                 status = checkpoint.restore(ckpt)
#                 continue
#
#             # if no blow pu, update prev_loss
#             prev_loss = cur_loss
#
#             # save weights
#             manager.save()
#
#             # logging
#             # parameters
#             # delete the data_idx array, too long for logging purpose
#             del train_param_individual['data_idx']
#             mlflow.log_params(train_param_individual)
#
#             # artifacts
#             ckpt = manager.latest_checkpoint
#             print("Check point saved: {}".format(ckpt))
#             mlflow.set_tag("epoch {}.{}/{}.{}".format(i, j, params["meta_epoch"], data_idx_size), ckpt)
#             mlflow.log_artifact(ckpt + ".index")
#             mlflow.log_artifact(ckpt + ".data-00000-of-00001")
#
#             # training results
#             mlflow.log_metric(
#                 train_param_individual['loss_description'],
#                 cur_loss
#             )
#
#
#         print("# Meta epoch {} training finished.\n".format(i))


# # train model on large data set
# # train_model only allows at most 20 individual data sets
# # model : tf.keras.Model
# #   train_params : dict - training parameters (data_idx can be > 20)
# #   model_folder_path : string : path of folder that contains the model
# #   data_folder_path : string : path of the folder that contains data
# def train_model(model, train_params, model_folder_path="models", data_folder_path="data", data_set_cap=20):
#     data_idx_size = (len(train_params["data_idx"]) + data_set_cap - 1)//20
#     train_param_individual = train_params.copy()
#     data_idx = train_param_individual["data_idx"].copy()
#
#
#     for i in range(train_params["meta_epoch"]):
#         print("# Meta epoch {}: ".format(i))
#         data_idx = np.random.permutation(data_idx)
#
#         for j in range(data_idx_size):
#             print("## Meta epoch {}.{}".format(i, j))
#             train_param_individual["data_idx"] = data_idx[j:j+data_set_cap]
#             model, history = train_model_on_batch(model,
#                   train_param_individual, model_folder_path="models", data_folder_path="data")
#
#         print("# Meta epoch {} training finished.\n".format(i))
#
# def train_from_weights(model_init, model_name, train_params,
#       model_folder_path="models", data_folder_path="data", data_set_cap=20):
#     model = model_init()
#     weights = os.path.join(model_folder_path, model_name)
#     model.load_weights(weights)
#
#     train_model(model,
#                 train_params,
#                 model_folder_path=model_folder_path,
#                 data_folder_path=data_folder_path,
#                 data_set_cap=data_set_cap)
#
# def train_from_saved(model_name, train_params, model_folder_path="models", data_folder_path="data", data_set_cap=20):
#     model = tf.keras.load_model(os.join(model_folder_path, model_name))
#
#     model, history = train_model(model,
#                 train_params,
#                 model_folder_path=model_folder_path,
#                 data_folder_path=data_folder_path,
#                 data_set_cap=data_set_cap)




#
# # testing
# # hyperparameters
# # these are logged
# train_params = {
#     "data_set_name" : "discrete_1000",
#     "data_index_start" : 0,
#     "data_index_end" : 17,
#     "epochs" : 10,
#     "batch_size" : 180,
#     "validation_split" : 0.1,
#     "shuffle" : True, # shuffle per epoch
#     "loss_description" : "mean squared error",
#     "loss" : "mse",
#     "optimizer_description" : "adam",
#     "optimizer" : "adam",
# }
#
# from benchmark import weyl
# potentials, E, target = load_data_set(
#     "data",
#     train_params['data_set_name'],
#     train_params['data_index_start'],
#     train_params['data_index_end']
# )
# w = weyl((potentials, E))
# print('w shape', w.shape)
#
# print(potentials.shape, E.shape)
# model, history = train("discrete_1000_testModel", train_params)
#
# b = model([potentials, E])
# print('b shape', b.shape)
# # Plot
# import matplotlib.pyplot as plt
# fig=plt.figure(figsize=(18, 16), dpi= 50, facecolor='w', edgecolor='k')
# plt.scatter(E, target, color='red')
# plt.scatter(E, w, color='blue')
# plt.scatter(E, b, color='green')
# plt.show()
#
# model.save("models/discrete_1000_testModel")
