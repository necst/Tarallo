from distutils.command.config import config
import requests
import json
import os
from pwn import *
from tqdm import tqdm
import shutil
import ast
from config.config_general import general
from utils.exception import *

HEADERS = {"Authorization": "REDACTED"}
WORK_SUBDIR = "CuckooResults"


URL = "http://localhost:8090/tasks/report/"


# Open and read the file with the task ids written by the submitter
def read_task_ids(tasks_file):
    try:
        with open(tasks_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(e)


# Parse the response of the REST API and add the new info to the summary dict
"""
Structure of the output (single PE)
[
    (process1_name, ( (api1_name, (arg1, arg2, arg3, ...) ), (api2_name, (arg1, arg2, arg3, ...) ), ... ) ), 
    (process2_name, ( (api1_name, (arg1, arg2, arg3, ...) ), (api2_name, (arg1, arg2, arg3, ...) ), ... ) ),
    ...
]
"""
def parse_response_comparison(r, api_calls_path):
    global examined_pes

    try:
        json_response = r.json()

        pe_name = json_response["target"]["file"]["name"] # recorded PE name
        processes = json_response["behavior"]["processes"] # list of recorded processes

        # The Cuckoo's recording includes a lot of information. Most of the recorded elements
        # are not needed for our purposes. Let filter out this information
        processes_apis = [] # list of filtered api calls for all processes
        examined_processes = [] # list of already examined process names
        for process in processes:
            process_name = process['process_name'] # recorded process name
            calls = process["calls"] # recorded api calls for that process
            api_calls = [] # list of filtered api calls for that process

            # Remove random process names
            if 'tmp' in process_name or 'TMP' in process_name:
                process_name = "tmp"

            # If there are already other analyzed processes with the same name, then add an incremental
            # number to the process name
            if process_name in examined_processes:
                examined_processes.append(process_name)
                n_occurrences =  len(list(filter(lambda x: x == process_name, examined_processes)))
                process_name += f'({n_occurrences-1})'
            else:
                examined_processes.append(process_name)
            
            # If we are extracting APIs from one of the original executions, we discard some of the last
            # calls to compensate the overhead introduced by the injections
            if global_mode == 'comparison_original':
                new_start = int(len(calls) * 0.20)
                new_finish = int(len(calls) * 0.90)
                calls = calls[:new_finish]

            for call in calls:
                # We keep the api name and the list of arguments
                api_name = call["api"]
                
                """if api_name == "HttpSendRequestA":
                    print("\n\n\n\n\n\n\n\n")
                    print(call["arguments"])
                    print("\n\n\n\n\n\n\n")"""

                api_arguments = []
                # Ignore the argument specified in the config file and the ones with value that contain a
                # tmp value
                for argument in call["arguments"]:
                    if argument not in general["ignored_arguments"]:
                        # An argument might be a list or a dict. We want pure strings
                        argument_value = str(call["arguments"][argument]).lower()
                        if "temp" not in argument_value and "tmp" not in argument_value:
                            api_arguments.append(argument_value)
                        else:
                            api_arguments.append('')
                # We want sorted arguments for coherence
                api_arguments = sorted(set(api_arguments))


                api_calls.append((api_name, tuple(api_arguments)))
        
            processes_apis.append((process_name, tuple(api_calls)))

        # Transform the api calls sequence in a set of api calls
        processes_apis = set(processes_apis)

        # Check the mode and define the combination strategy
        if global_mode == 'comparison_original':
            combine = my_set_intersection
        elif global_mode == 'comparison_patched':
            combine = my_set_union
        else:
            raise WrongCombinationMode('The combination mode does not exist')

        # If there is already an api calls set for the current PE name, then we overwrite
        # that set with the intersection between the old one and the new one. Otherwise, we 
        # simply save the current api calls set
        if pe_name in examined_pes:
            # Open the old api calls set from file
            with open(os.path.join(api_calls_path, pe_name), 'r', encoding='utf-8') as f:
                old_processes_apis = ast.literal_eval(json.load(f))
            
            # Combination between the old and the new api calls sets
            new_processes_apis = combine(old_processes_apis, processes_apis)

            # Overwrite the old api calls set file
            write_extracted_apis(os.path.join(api_calls_path, pe_name), str(new_processes_apis))
        else:
            # Mark the PE name as examined
            examined_pes.append(pe_name)
            # Write the api calls set in a file with filename equal to the PE name
            write_extracted_apis(os.path.join(api_calls_path, pe_name), str(processes_apis))
    
    except KeyError as e:
        if e.args[0] == "behavior":
            print(f"The PE {pe_name} cannot be executed by cuckoo. No api calls information")
        elif e.args[0] == "target":
             print(f"The PE name was not recorded by Cuckoo. No api calls extracted")
        else:
            print(f"KeyError - reason {e}")
    except Exception as e:
        print(f"Another exception occured. Reason: {e}")

def parse_response_attack(r, api_calls_path):
    call_list = []
    json_response = r.json()

    try:
        pe_name = json_response["target"]["file"]["name"] # recorded PE name
        pe_name = ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
            
        for process in json_response['behavior']['processes']:
            if len(process['calls']) != 0:
                for call in process['calls']:
                    call_list.append(call['api'])

        # Write the api calls set in a file with filename equal to the PE name
        write_extracted_apis(os.path.join(api_calls_path, pe_name), call_list)

    except KeyError as e:
        if e.args[0] == "behavior":
            print(f"The PE {pe_name} cannot be executed by cuckoo. No api calls information")
        elif e.args[0] == "target":
             print(f"The PE name was not recorded by Cuckoo. No api calls extracted")
        else:
            print(f"KeyError - reason {e}")
    except Exception as e:
        print(f"Another exception occured. Reason: {e}")



# Function to iterate over each reported process and perfom the intersection between the 
# correspondant api calls sets
def my_set_intersection(old_processes_apis, current_processes_apis):
    new_processes_apis = []

    # Get the processes names from both the old and new api calls sets
    old_processes_name = set(map(lambda x: x[0], old_processes_apis))
    current_processes_name =  set(map(lambda x: x[0], current_processes_apis))
    processes_name = old_processes_name.intersection(current_processes_name)

    for process_name in processes_name:
        try:
            # Extract from the old PE apis the tuple of a specific process
            old_filtered_processes_apis = list(filter(lambda x: x[0] == process_name, old_processes_apis))

            # Extract from the current PE apis the tuple of a specific process
            current_filtered_processes_apis = list(filter(lambda x: x[0] == process_name, current_processes_apis))

            # New tuple as intersection of the two tuples
            new_api_calls = set(old_filtered_processes_apis[0][1]).intersection(set(current_filtered_processes_apis[0][1]))
        
        except Exception as e:
            print("Wrong results format!")
            exit()
        new_processes_apis.append((process_name, tuple(new_api_calls)))
    return set(new_processes_apis)


# Function to iterate over each reported process and perfom the union between the correspondant 
# api calls sets
def my_set_union(old_processes_apis, current_processes_apis):
    new_processes_apis = []

    # Get the processes names from both the old and new api calls sets
    old_processes_name = list(map(lambda x: x[0], old_processes_apis))
    current_processes_name =  list(map(lambda x: x[0], current_processes_apis))
    processes_name = old_processes_name + current_processes_name

    for process_name in processes_name:

        # Extract from the old PE apis the tuple of a specific process
        old_filtered_processes_apis = list(filter(lambda x: x[0] == process_name, old_processes_apis))

        # Extract from the current PE apis the tuple of a specific process
        current_filtered_processes_apis = list(filter(lambda x: x[0] == process_name, current_processes_apis))

        if len(current_filtered_processes_apis) == 0 and len(old_filtered_processes_apis) == 1:

            # Match found only in the old api calls set
            new_api_calls = set(old_filtered_processes_apis[0][1])
        elif len(current_filtered_processes_apis) == 1 and len(old_filtered_processes_apis) == 0:

            # Match found only in the current api calls set
            new_api_calls = set(current_filtered_processes_apis[0][1])
        elif len(current_filtered_processes_apis) == 1 and len(old_filtered_processes_apis) == 1:
            
            # Match found both in the current and old api calls sets
            new_api_calls = set(old_filtered_processes_apis[0][1]).union(set(current_filtered_processes_apis[0][1]))
        else:
            raise Exception(f"Wrong results format. Single process recorded multiple times!")
        new_processes_apis.append((process_name, tuple(new_api_calls)))
    return set(new_processes_apis)


# Write the api calls information in a file at dest_path
def write_extracted_apis(dest_path, processes_apis):
    with open(dest_path, 'w', encoding='utf-8') as f:
        json.dump(processes_apis, f)


# If the dest_path already exists, delete the folder to remove old files from some previous run and create an 
# empty folder   
def delete_old_results(api_calls_path):
    if os.path.exists(api_calls_path):
        shutil.rmtree(api_calls_path)
    os.mkdir(api_calls_path)


def api_extractor(tasks_file, api_calls_path, mode):
    global examined_pes
    global global_mode 
    
    # Set the gloabl mode
    global_mode = mode

    # list of the already examined PEs
    examined_pes = [] 
    
    # Read the task ids from file
    task_ids = read_task_ids(tasks_file)

    delete_old_results(api_calls_path)

    if global_mode == 'comparison_original' or global_mode == 'comparison_patched':
        parse_response = parse_response_comparison
    elif global_mode == 'attack':
        parse_response = parse_response_attack
    else:
        raise WrongExtractionMode('Wrong API extracion mode')

    for task_id in tqdm(task_ids):
        current_url = URL + str(task_id)
        try:
            # Get request to cuckoo's REST API
            r = requests.get(current_url, headers=HEADERS)
        except Exception as e:
            print(e)
            r = None
        parse_response(r, api_calls_path)