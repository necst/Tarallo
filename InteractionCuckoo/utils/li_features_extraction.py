import os
import json
import numpy as np
from tqdm import tqdm
import requests
from config.config_general import general
import datetime


# Open and read the file with the task ids written by the submitter
def read_task_ids(tasks_file):
    try:
        with open(tasks_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(e)

def retrieve_api_list_from_sample(sample_filename, suffix = '_imported_apis'):
    path = os.path.join(imported_apis_folder, sample_filename + suffix)
    if not os.path.exists(path):
        print("File {} not found".format(path))
        return []
    with open(path, 'rb') as f:
        api_list = f.read().splitlines()
    return api_list

def api_extraction(r):
    """
    This function extracts the recorded api calls from the json report and stores it in the global variable report_list
    """
    global report_list

    api_num = []

    try:
        report_dict = {}
        call_list = []
        load_dict = r.json()
        if 'behavior' not in load_dict:
            return
        name = load_dict['target']['file']['name']
        report_dict['md5'] = load_dict['target']['file']['md5']
        for process in load_dict['behavior']['processes']:
            if len(process['calls']) != 0:
                for call in process['calls']:
                    call_list.append(call['api'])
                    # Statistics API
                    if call['api'] not in api_num:
                        api_num.append(call['api'])
        report_dict['call_list'] = call_list
        report_dict['sample_name'] = name 
        report_dict['imported_apis'] = retrieve_api_list_from_sample(name, '_imported_apis')
        report_dict['called_apis'] = retrieve_api_list_from_sample(name, '_called_apis')

        if report_dict['imported_apis'] != []:
            report_list.append(report_dict)

    except KeyError as e:
        if e.args[0] == "behavior":
            print(f"The PE {name} cannot be executed by cuckoo. No api calls information")
        elif e.args[0] == "target":
             print(f"The PE name was not recorded by Cuckoo. No api calls extracted")
        else:
            print(f"KeyError - reason {e}")
    except Exception as e:
        print(f"Another exception occured. Reason: {e}")


# Helper function to transform a tuple of behaviors in a tuple of ids
def tuple_to_id(tpl):
    return np.array([behavior2id[tpl[0]], behavior2id[tpl[1]], behavior2id[tpl[2]], behavior2id[tpl[3]]])


# Helper function to transform a sequence of API names into a sequence
# of behavior ids
def behaviors_from_name(api_names_sequence):
    
    # From api name to behavior
    api_id_sequence = np.array(list(map(lambda x: word2behavior[x], api_names_sequence)))

    # From behavior to behavior id
    behaviors_id_sequence = np.array(list(map(tuple_to_id, api_id_sequence)))
  
    return behaviors_id_sequence


# Helper function to transform a sequence of API id names into a sequence
# of behavior ids
def behaviors_from_id_name(api_id_names_sequence):
    
    # From api id to api name
    api_names_sequence = list(map(lambda x: id2word[x], api_id_names_sequence))
    
    return behaviors_from_name(api_names_sequence)

def id_from_name(api_names_sequence):
    # From api name to id
    api_id_sequence = np.array(list(map(lambda x: word2id[x], api_names_sequence)))
    return api_id_sequence

def li_features_extraction(tasks_file, output_path, imported_apis_path):
    global word2id
    global id2word
    global behavior2id
    global word2behavior
    global imported_apis_folder
    global report_list

    report_list = []  # List of dictionaries with the extracted features

    imported_apis_folder = imported_apis_path

    tesi_cwd = os.environ['TESI_CWD']
    dict_dir = os.path.join(tesi_cwd, 'Tesi/Cuckoo/multi_submissions/static')
        
    # Load all the needed dicts for the conversions
    word2id = np.load(os.path.join(dict_dir,'word2id.npz'), allow_pickle=True)
    word2id = word2id['word2id'][()]
    id2word = dict(map(reversed, word2id.items())) #reverse map
    behavior2id = np.load(os.path.join(dict_dir,'behavior2id.npz'), allow_pickle=True)
    behavior2id = behavior2id['behavior2id'][()]
    word2behavior = np.load(os.path.join(dict_dir,'word2behavior.npz'), allow_pickle=True)
    word2behavior = word2behavior['word2behavior'][()]

    # Read the task ids from file
    task_ids = read_task_ids(tasks_file)

    for task_id in tqdm(task_ids):
        current_url = general["webserver_url"] + str(task_id)
        
        request_retry_num = 10
        while request_retry_num > 0:
            try:
                # Get request to cuckoo's REST API
                r = requests.get(current_url, headers=general["header"])
            except Exception as e:
                print(e)
                r = None
            if r is not None and r.status_code == 200:
                break
            else:
                print("Error in request. Retrying...")
                request_retry_num -= 1
        api_extraction(r)
    
    # Remove the imported APIs that are not in the set analyzed by Cuckoo
    report_list_filtered = []
    for rep_dict in tqdm(report_list):
        rep_dict_filtered = {}
        rep_dict_filtered['md5'] = rep_dict['md5']    
        rep_dict_filtered['call_list'] = [a for a in rep_dict['call_list'] if a in word2id.keys()]
        rep_dict_filtered['imported_apis'] = rep_dict['imported_apis']
        report_list_filtered.append( rep_dict_filtered)
    
    FIX_LEN = 1000
    
    x_name      = np.zeros((0, FIX_LEN),   dtype=np.int64)
    x_semantic  = np.zeros((0, FIX_LEN*4), dtype=np.int64)

    for rep_dict in tqdm(report_list_filtered):
        
        if len(rep_dict['call_list']) < FIX_LEN:
        
            # Pad the sequence with the _PAD_ token
            rep_dict['call_list'] = rep_dict['call_list'] + ['_PAD_'] * (FIX_LEN - len(rep_dict['call_list']))
        
        elif len(rep_dict['call_list']) > FIX_LEN:
        
            # Truncate the sequence
            rep_dict['call_list'] = rep_dict['call_list'][:FIX_LEN]

        id_for_apis  = id_from_name(rep_dict['call_list'])
        
        # For a single sample, 1000 api call
        id_for_apis  = np.reshape(id_for_apis, (1000, 1))
        id_for_apis  = np.transpose(id_for_apis)

        # Stacking all the samples
        x_name       = np.vstack((x_name, id_for_apis))

        # For a single sample, 1000 api call * 4 behaviors
        behaviors = behaviors_from_name(rep_dict['call_list'][:FIX_LEN])
        
        # Flatten the array
        sem_for_apis = behaviors.flatten()

        sem_for_apis = np.expand_dims(sem_for_apis, axis=0)
        
        # Stacking all the samples
        x_semantic   = np.vstack((x_semantic, sem_for_apis))

    # Number of samples
    assert x_name.shape[0] == x_semantic.shape[0]
    
    # Number of API calls
    assert x_name.shape[1] == FIX_LEN
    
    # Number of API calls * 4 behaviors
    assert x_semantic.shape[1]  == FIX_LEN * 4

    # This assumes that we are only passing malware samples (label = 1) !!
    now = datetime.datetime.now() # get the current date and time
    filename = "li_model_extracted_features_" + now.strftime("%Y-%m-%d_%H-%M-%S")
    file_path = os.path.join(output_path, filename)
    np.savez(file_path,
              x_name = x_name,
              x_semantic = x_semantic,
              y = np.ones((x_name.shape[0], 1), dtype=np.int64),
              imported_apis = [rep_dict['imported_apis'] for rep_dict in report_list],
              called_apis   = [rep_dict['called_apis'] for rep_dict in report_list],
              sample_names  = [rep_dict['sample_name'] for rep_dict in report_list]
            )