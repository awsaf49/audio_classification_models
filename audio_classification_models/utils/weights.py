import gdown
import os

def load_pretrain(model, url, fname):
    "Download weights from google drive using url then load weights to the model"
    base_dir = '~/.keras/datasets/'
    local_path = os.path.join(base_dir, fname)
    os.makedirs(base_dir, exist_ok=True)
    if not os.path.isfile(local_path):
        gdown.download(url, local_path, quiet=False, fuzzy=True)
    model.load_weights(local_path, by_name=True,skip_mismatch=True)