import os
from iodata import load_data_set_

def create_model(params):
    train_ds_path = os.path.join(params['ds_folder'], params['ds_name'], "train")

    import importlib
    model_file =  params["model_folder_path"] + '.' + params["model_class"]
    model_getter = importlib.import_module(model_file).get_model
    model = model_getter(params)

    p, E, _ = load_data_set_(train_ds_path, params['ds_name'], [0])

    model((p, E))

    model_save_path = os.path.join(params["model_folder_path"], params["model_name"], params["model_name"])
    if not os.path.isfile(model_save_path):
        model.save(model_save_path)

    return model

def save_model(model, params):
    model_path = os.path.join(params["model_folder_path"], params["model_name"])
    model.save(model_path)

def load_model(params):
    import tensorflow as tf

    model_path = os.path.join(params["model_folder_path"], params["model_name"])
    model = tf.keras.models.load_model(model_path)

    return model
