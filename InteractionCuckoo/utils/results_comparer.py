import json
import os
import ast
from termcolor import cprint
from tqdm import tqdm
from config.config_general import general
from config.config_paths import original_pes_paths, general_paths
import logging

# Get only the PE filenames actually patched
def get_filenames(summary_path):
    patched_filenames = [] # list of actually patched PE filenames

    # Open the summary file
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Add to the list the correctly patched PEs
    for item in summary:
        if item['exit_code'] == 0:
            patched_filenames.append(item['filename'])
    
    # Filter the ignored PEs
    ignored_pes = general["ignored_pes"]
    patched_filenames = list(filter(lambda x: x not in ignored_pes, patched_filenames))

    return patched_filenames


# Open and read the file with the results written by the api_extractor
def read_results(result_filename):
    try:
        with open(result_filename, "r") as f:
            return ast.literal_eval(json.load(f)) # recover the set structure
    except Exception as e:
        print(e)


# Function to iterate over each reported process and perfom the difference between the correspondant 
# original and patched api calls sets
def my_set_difference(original_results, patched_results):
    
    logger    = logging.getLogger("cuckoo_comparison")

    sample_name = 'unknown'
    for original_process in original_results:
        if 'VirusShare' in original_process[0]:
            sample_name = original_process[0]

    for original_process in original_results:
        original_process_name = original_process[0]
        original_api_calls = set(original_process[1])

        # Extract from the patched results the tuple of a specific process
        patched_filtered_processes = list(filter(lambda x: x[0] == original_process_name, patched_results))

        if len(patched_filtered_processes) == 0:
            if(filename in  []):
                print('\n', filename)
                print('missed')
                print(original_results)
                print('\n')
                print(patched_results)
                print("\n\n\n")
            logger.error(f"[{sample_name}] Process {original_process} not found in the patched PE")
            return False # Process not found in the patched PE
        elif len(patched_filtered_processes) == 1:
            patched_api_calls = set(patched_filtered_processes[0][1])
        else:
            raise Exception(f"Wrong results format. Single process recorded multiple times in PE {filename}!")

        # Compare the api call sets
        if len(original_api_calls - patched_api_calls) == 0:
            continue # Ok, go to  the next process
        else:
            if(filename in   []):
                print('\n', filename)
                print(original_api_calls - patched_api_calls)
                print("\n\n\n")
                print(patched_api_calls - original_api_calls)
            logger.error(f"[{sample_name}] \n\t Diff: {original_api_calls - patched_api_calls}\n\t Patched: {patched_api_calls}")
            logger.error("*"*100)
            if ('__exception__', ()) in patched_api_calls and len(patched_api_calls) == 1:
                logger.critical(f"[{sample_name}] only exception recorded")
            return False # Some api calls are not exeuted in the patched PE
    return True # All the processes have been checked with positive result


# Comparison logic
def comparer(original_results_filenames, patched_results_filenames, patched_filenames, original_results_path, patched_results_path):
    global filename
    for filename in tqdm(patched_filenames):
        if filename in original_results_filenames and filename in patched_results_filenames:
            # Read the original and patched results
            original_results = read_results(os.path.join(original_results_path, filename))
            patched_results = read_results(os.path.join(patched_results_path, filename))

            # Compare the original and patched api calls sets
            if my_set_difference(original_results, patched_results):
                same_api_calls.append(filename) # Ok, the api calls sets are the same
            else:
                different_api_calls.append(filename) # Damn, the api calls sets are different
        
        elif filename in original_results_filenames and filename not in patched_results_filenames:
            # Cuckoo was able to analyze the original PE but not the patched one
            original_but_modified.append(filename)
        elif filename not in original_results_filenames and filename in patched_results_filenames:
            # Cuckoo was able to analyze the pathced PE but not the original one
            modified_but_original.append(filename)
        else:
            # Cuckoo was not able to analyze both the original and the patched PEs
            uncatched.append(filename)


def results_comparer(original_results_path, patched_results_path, summary_path, comparison_results_path):
    global same_api_calls
    global different_api_calls
    global original_but_modified
    global modified_but_original
    global uncatched
  
    same_api_calls = [] # List of PE filenames that preserve their functionalities
    different_api_calls = [] # List of PE filenames that do NOT preserve their functionalities
    original_but_modified = [] # List of PE filenames for which Cuckoo was able to analyze the original PE but not the patched one
    modified_but_original = [] # List of PE filenames for which Cuckoo was able to analyze the patched PE but not the original one
    uncatched = [] # List of PE filenames for which Cuckoo was not able to analyze both the original and the patched PEs
    
    # List the PE filenames with available results
    if not os.path.exists(original_results_path):
        print("\n")
        cprint("The path to the original PEs api calls sets does not exist", "red")
        return
    original_results_filenames = os.listdir(original_results_path)
    if not os.path.exists(patched_results_path):
        print("\n")
        cprint("The path to the patched PEs api calls sets does not exist", "red")
        return
    patched_results_filenames = os.listdir(patched_results_path)

    # Get the actually patched PE filenames from the patcher summary
    patched_filenames = get_filenames(summary_path)

    # Call the comparer
    comparer(original_results_filenames, patched_results_filenames, patched_filenames, original_results_path, patched_results_path)
    
    # Print the results
    total = len(same_api_calls) + len(different_api_calls)

    print(f'The PEs that preserve their functionalities are {len(same_api_calls)} over {len(same_api_calls) + len(different_api_calls)}. In percentage: {len(same_api_calls)/total*100 if total != 0 else 100}%')
    print(f'The PEs that NOT preserve their functionalities are {len(different_api_calls)} over {len(same_api_calls) + len(different_api_calls)}. In percentage: {len(different_api_calls)/total*100 if total != 0 else 100}%')
    print(f'The PEs that were correctly analized before the injection but not after it are {len(original_but_modified)}')
    print(f'The PEs that were correctly analized after the injection but not before it are {len(modified_but_original)}')
    print(f'The PEs that were not correctly analized by cuckoo are {len(uncatched)}')
    print('\n\n\n')
    print(f'PEs that preserve their functionalities \n {same_api_calls}')
    print('\n\n\n')
    print(f'PEs that NOT preserve their functionalities \n {different_api_calls}')
    print('\n\n\n')
    print(f'PEs correctly analized before but not after \n {original_but_modified}')
    print('\n\n\n')
    print(f'PEs correctly analized after but not before \n {modified_but_original}')
    print('\n\n\n')
    print(f'PEs not correctly analized by cuckoo \n {uncatched}')