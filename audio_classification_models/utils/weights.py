import os
import tensorflow as tf

def load_pretrain(model, url, fname=None):
    "Download weights from google drive using url then load weights to the model"
    local_path = tf.keras.utils.get_file(fname, origin=url)
    model.load_weights(local_path, by_name=True,skip_mismatch=True)