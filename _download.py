#
# Import Libraries
#
# %%
import os
import urllib
import tarfile

#
# Function to Download Model
#


def _download(DOWNLOAD_BASE, MODEL_FILE):
    '''
    Download model
    Args:
    DOWNLOAD_BASE: base url of the download model
    MODEL_FILE: the name, including extension, of the download model
    '''
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())
