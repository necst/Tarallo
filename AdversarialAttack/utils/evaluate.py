from config.config_paths import general_paths
import os
import torch
from utils.model import *
from utils.helper import *
from utils.attack import *


# Function to set the needed paths
def set_paths():
    global DIR
    global model_path

    DIR = general_paths['dir']
    model_path = os.path.join(DIR, general_paths['model_name']) # name of the model to load


def evaluate(original_sequence):
    global model
    # Set the paths
    set_paths()

    # Extract the info from the data
    original_names = original_sequence[0]
    original_behaviors = original_sequence[1]
    original_import = original_sequence[2]
    original_sample_name = original_sequence[3]
    original_called = original_sequence[4]
    

    # Load the model
    model = torch.load(model_path)

    # Cast the model to the new defined class
    model.__class__ = CustomNet

    # Restore the proper dimensionality
    original_names = np.expand_dims(original_names, axis=0)
    original_behaviors = np.expand_dims(original_behaviors, axis=0)

    # Decode and filter the list of imported APIs
    original_import_decoded  = convert_to_string(original_import)
    original_import_filtered = helper_func(original_import_decoded)

    # Check if the sample is already classified as not malicious
    prediction = model.pred_given_split(torch.tensor(original_names).to(device), torch.tensor(original_behaviors).to(device))

    return (original_sample_name, prediction)