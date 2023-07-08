import requests
import os
import json
import numpy as np
import math
from termcolor import cprint
from tqdm import tqdm
from pwn import *
from ChainFramework.Utils.peAnalyzer import retrieve_imported_api_list_from_dir


HEADERS = {"Authorization": "REDACTED"}
URL = "http://localhost:8090/tasks/create/submit"


# Function to read and send a limited number of files in one call
def files_handler(file_names, files_dir):
    file_names = np.array(file_names) # numpy array of PE file names
    n_chunks = math.ceil(len(file_names)/50) # number of chuncks (size: 50 elements) in which split the array
    file_names_splitted = np.array_split(file_names, n_chunks) # split the array in n_chuncks (size: 50 elements)
    for file_names_cut in tqdm(file_names_splitted):
        files = [("files", open(os.path.join(files_dir, file_name), "rb")) for file_name in file_names_cut]
        submit(files)

# Submit the files
def submit(files):
    global tasks
    global errors

    done = False
    while not done:
        try:
            r = requests.post(URL, files=files, headers=HEADERS) # post the submit request
        except requests.exceptions.ConnectionError:
            cprint("Cuckoo is not running!", "red", attrs=['bold'])
            return
        
        if not r.ok:
            cprint("Submit failed, trying again", "red", attrs=['bold'])
        else:
            cprint("Submit OK", "green", attrs=['bold'])
            done = True
    tasks += r.json()["task_ids"]
    errors += r.json()["errors"]

# Write the IDs and the errors in files
def writer(tasks, errors, tasks_file, errors_file):
    # If the path already exists, delete the folder to remove old files from some previous run
    if os.path.exists(tasks_file):
        os.remove(tasks_file)
    if os.path.exists(errors_file):
        os.remove(errors_file)
    
    # Write the task ids and error ids in two different files
    if len(tasks) > 0:
        with open(tasks_file, 'w', encoding='utf-8') as f:
            json.dump(tasks, f)
    if len(errors) > 0:
         with open(errors_file, 'w', encoding='utf-8') as f:
            json.dump(errors, f)


def submitter(n_subm, files_dir, tasks_file, errors_file, extracted_apis_output_dir=None, file_names=None):  
    global tasks
    global errors

    tasks = [] # list of successed submssion ids
    errors = [] # list of failed submission ids

    # Retrieve the file names to submit
    if file_names is None:
        file_names = os.listdir(files_dir) [:]

    # Submit each file N_SUBM times
    file_names_repeated = file_names * n_subm

    # Manage the submission
    files_handler(file_names_repeated, files_dir)

    # Write the ids in files
    writer(tasks, errors, tasks_file, errors_file)

    # First retrive the api list from the files in files_dir
    # now works in parallel
    if extracted_apis_output_dir is not None:
        print("Retrieving the imported API list from the files...")
        retrieve_imported_api_list_from_dir(files_dir, file_names, extracted_apis_output_dir)