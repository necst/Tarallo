import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../') # Compatibility reasons
from pwn import *
from utils.submitter import submitter
from utils.api_extractor import api_extractor
from utils.results_comparer import results_comparer
from utils.cuckoo_api_utils import deleter, cuckoo_status, write_dir_api_lists
from config.config_paths import general_paths, original_pes_paths, patched_pes_paths, comparison_paths
from config.config_general import general
import sys
from termcolor import cprint
from ChainFramework.Utils.peAnalyzer import retrieve_imported_api_list_from_dir
from AdversarialAttack.config.config_paths import general_paths as adv_general_paths

from utils.li_features_extraction import li_features_extraction
from utils.zhang_features_extraction import zhang_features_extraction

DIR = general_paths["dir"]
N_SUBM = general["n_subm"] # number of times each PE will be executed by Cuckoo sandbox

# Print the menu options
def print_menu():
    print("What do you want to do?")
    print("1) Submit original PEs to Cuckoo and extract api calls (stored in the original folder)")
    print("2) Retrieve original api calls from Cuckoo's results for comparison")
    print("3) Submit patched PEs to Cuckoo")
    print("4) Retrieve patched api calls from Cuckoo's results for comparison")
    print("5) Compare the original and patched api calls sets")
    print("6) Show configuration")
    print("7) Delete current tasks (Stop tasks)")
    print("8) Show Cuckoo status")
    print("9) Retrieve original api calls from Cuckoo's results for attack")
    print("10) Retrieve patched api calls from Cuckoo's results for attack")
    print("11) Extract original api calls features from Cuckoo's results for Li's model")
    print("12) Extract patched api calls features from Cuckoo's results for Li's model")
    print("13) Extract original api calls features from Cuckoo's results for Zhang's model")
    print("14) Extract patched api calls features from Cuckoo's results for Zhang's model")
    print("15) Submit only samples that were successfully attacked by adversarial logic")
    print("0) Exit")


# Print divisor between menu prints
def print_divisor():
    print("\n")
    print("-" * 35)
    print("\n")


# Print configuration infomartion
def print_conf_info():
    print("\n")
    cprint("General Configurations", "blue", attrs=['bold'])
    print(f"Work directory: {general_paths['dir']}")
    print(f"Each PE will be executed by Cuckoo {general['n_subm']} times")
    print("\n")
    cprint("Original Files Configuration Paths", "blue", attrs=['bold'])
    print(f"The folder containing the PEs to analyze is: {original_pes_paths['pes_subdir']}")
    print(f"The output folder is: {original_pes_paths['output_subdir']}")
    print(f"The filename to store the successed submssion ids is: {original_pes_paths['tasks_filename']}")
    print(f"The filename to store the failed submssion ids is: {original_pes_paths['errors_filename']}")
    print(f"The filename to store the extracted api calls sets for comparison is: {original_pes_paths['api_calls_subdir_comparison']}")
    print(f"The filename to store the extracted api calls sets for attack is: {original_pes_paths['api_calls_subdir_attack']}")
    print("\n")
    cprint("Patched Files Configuration Paths", "blue", attrs=['bold'])
    print(f"The folder containing the PEs to analyze is: {patched_pes_paths['pes_subdir']}")
    print(f"The output folder is: {patched_pes_paths['output_subdir']}")
    print(f"The filename to store the successed submssion ids is: {patched_pes_paths['tasks_filename']}")
    print(f"The filename to store the failed submssion ids is: {patched_pes_paths['errors_filename']}")
    print(f"The filename to store the extracted api calls sets for comparison is: {patched_pes_paths['api_calls_subdir_comparison']}")
    print(f"The filename to store the extracted api calls sets for attack is: {patched_pes_paths['api_calls_subdir_attack']}")
    print("\n")
    cprint("Api Calls Sets Comparison Configuration Paths", "blue", attrs=['bold'])
    print(f"The path to the summary written by the patcher is: {comparison_paths['summary_path']}")
    print(f"The output folder is: {comparison_paths['output_subdir']}")
    print(f"The filename to store the comparison results is: {comparison_paths['comparison_results_filename']}")



# Set the paths and filenames for original PEs analysis
def set_original_paths():
    global PES_SUBDIR
    global OUTPUT_SUBDIR
    global TASKS_FILENAME
    global ERRORS_FILENAME
    global API_CALLS_SUBDIR_COMPARISON
    global API_CALLS_SUBDIR_ATTACK
    global LI_FEATURES_SUBDIR
    global ZHANG_FEATURES_SUBDIR
    global IMPORTED_APIS_SUBDIR

    PES_SUBDIR = original_pes_paths["pes_subdir"] # subfolder containing the PEs to analyze
    OUTPUT_SUBDIR = original_pes_paths["output_subdir"] # output folder name
    TASKS_FILENAME = original_pes_paths["tasks_filename"] # where store the successed submssion ids
    ERRORS_FILENAME = original_pes_paths["errors_filename"] # where store the failed submssion ids

    # Where store the extracted api calls sets for comparison
    API_CALLS_SUBDIR_COMPARISON = original_pes_paths["api_calls_subdir_comparison"]

    # Where store the extracted api calls sets for attack
    API_CALLS_SUBDIR_ATTACK = original_pes_paths["api_calls_subdir_attack"]

    # Where store the extracted features using Li's method
    LI_FEATURES_SUBDIR =  original_pes_paths["li_features_subdir"]

    # Where store the extracted features using Li's method
    ZHANG_FEATURES_SUBDIR =  original_pes_paths["zhang_features_subdir"]    

    # Where store the extracted imported apis
    IMPORTED_APIS_SUBDIR =  original_pes_paths["imported_api_calls_subdir"]
    set_dirs()


# Set the paths and filenames for patched PEs analysis
def set_patched_paths():
    global PES_SUBDIR
    global OUTPUT_SUBDIR
    global TASKS_FILENAME
    global ERRORS_FILENAME
    global API_CALLS_SUBDIR_COMPARISON
    global API_CALLS_SUBDIR_ATTACK
    global LI_FEATURES_SUBDIR
    global ZHANG_FEATURES_SUBDIR
    global IMPORTED_APIS_SUBDIR

    PES_SUBDIR = patched_pes_paths["pes_subdir"] # subfolder containing the PEs to analyze
    OUTPUT_SUBDIR = patched_pes_paths["output_subdir"] # output folder name
    TASKS_FILENAME = patched_pes_paths["tasks_filename"] # where store the successed submssion ids
    ERRORS_FILENAME = patched_pes_paths["errors_filename"] # where store the failed submssion ids

    # Where store the extracted api calls sets for comparison
    API_CALLS_SUBDIR_COMPARISON = patched_pes_paths["api_calls_subdir_comparison"]

    # Where store the extracted api calls sets
    API_CALLS_SUBDIR_ATTACK = patched_pes_paths["api_calls_subdir_attack"]

    # Where store the extracted features using Li's method
    LI_FEATURES_SUBDIR =  patched_pes_paths["li_features_subdir"]

    # Where store the extracted features using Li's method
    ZHANG_FEATURES_SUBDIR =  patched_pes_paths["zhang_features_subdir"]  

    # Where store the extracted imported apis
    IMPORTED_APIS_SUBDIR =  patched_pes_paths["imported_api_calls_subdir"]
    set_dirs()


# Set the paths and filenames for the comparison
def set_comparison_paths():
    global original_results_path
    global patched_results_path
    global comparison_results_path
    global summary_path
    global imported_apis_path

    # Where the extracted api calls sets are stored
    original_results_path = os.path.join(DIR, original_pes_paths["output_subdir"], original_pes_paths["api_calls_subdir_comparison"])
    patched_results_path = os.path.join(DIR, patched_pes_paths["output_subdir"], patched_pes_paths["api_calls_subdir_comparison"])

    # Path to the output folder
    output_dir = os.path.join(DIR, comparison_paths["output_subdir"]) 
    
    # If the path does not exist, create the correspondant folder
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Where store the comparison results
    comparison_results_path = os.path.join(output_dir, comparison_paths["comparison_results_filename"])

    # Where the patcher summary is stored
    summary_path = comparison_paths["summary_path"]



# Set the needed path variables and create the destination folder
def set_dirs():
    global files_dir
    global output_dir
    global tasks_file
    global errors_file
    global api_calls_path_comparison
    global api_calls_path_attack
    global li_features_subdir
    global zhang_features_subdir
    global zhang_tmp_subdir
    global imported_apis_path

    files_dir  = os.path.join(DIR, PES_SUBDIR) # path to the PEs to analyze
    output_dir = os.path.join(DIR, OUTPUT_SUBDIR) # path to the output folder

    tasks_file = os.path.join(output_dir, TASKS_FILENAME) # path where store the successed submssion ids
    errors_file = os.path.join(output_dir, ERRORS_FILENAME) # path where store the failed submssion ids

    api_calls_path_comparison = os.path.join(output_dir, API_CALLS_SUBDIR_COMPARISON)
    api_calls_path_attack     = os.path.join(output_dir, API_CALLS_SUBDIR_ATTACK)
    li_features_subdir        = os.path.join(output_dir, LI_FEATURES_SUBDIR)
    zhang_features_subdir     = os.path.join(output_dir, ZHANG_FEATURES_SUBDIR)
    zhang_tmp_subdir          = os.path.join(output_dir, general_paths['tmp_zhang_subdir'])
    imported_apis_path        = os.path.join(output_dir, IMPORTED_APIS_SUBDIR)

    # If the paths do not exist, create the correspondant folders
    if not os.path.exists(li_features_subdir):
        os.mkdir(li_features_subdir)
    if not os.path.exists(zhang_features_subdir):
        os.mkdir(zhang_features_subdir)
    if not os.path.exists(zhang_tmp_subdir):
        os.mkdir(zhang_tmp_subdir)
    if not os.path.exists(imported_apis_path):
        os.mkdir(imported_apis_path)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


# Menu managment
def menu():
    while True:
        print_menu()
        choice = input("> ")
        choice = choice.replace('\n', '')
        if choice == '1' or choice == '2' or choice == '9' or choice == '11' or choice == '13' or choice=='15':
            set_original_paths()
            if choice == '1':
                # submit N_SUBM time each PE in files_dir
                submitter(N_SUBM, files_dir, tasks_file, errors_file, imported_apis_path, None)
            elif choice == '2':
                # extract the api information for comparison
                api_extractor(tasks_file, api_calls_path_comparison, mode='comparison_original')
            elif choice == '9':
                # extract the api information for attack
                api_extractor(tasks_file, api_calls_path_attack, mode='attack')
            elif choice == '11':
                # extract the features using Li's method
                li_features_extraction(tasks_file, li_features_subdir, imported_apis_path)
            elif choice == '13':
                # extract the features using Zhang's method
                zhang_features_extraction(tasks_file, zhang_tmp_subdir, zhang_features_subdir)
            elif choice == '15':
                # Submit only samples that were successfully attacked by adversarial logic
                hijacking_data_dir = adv_general_paths['hijacking_data_dir']
                file_names = [f for f in os.listdir(files_dir) if os.path.isfile(os.path.join(hijacking_data_dir, 'hijacking_data_'+f+'.pkl'))]
                print('Number of samples to submit: ', len(file_names))
                input('Press enter to continue...')
                submitter(N_SUBM, files_dir, tasks_file, errors_file, None, file_names)
        elif choice == '3' or choice == '4' or choice == '10' or choice == '12' or choice == '14':
            set_patched_paths()
            if choice == '3':
                # submit N_SUBM time each PE in files_dir
                submitter(N_SUBM, files_dir, tasks_file, errors_file, imported_apis_path, None)
            elif choice == '4':
                # extract the api information for comparison
                api_extractor(tasks_file, api_calls_path_comparison, mode='comparison_patched')
            elif choice == '10':
                # extract the api information for attack
                api_extractor(tasks_file, api_calls_path_attack, mode='attack')
            elif choice == '12':
                # extract the features using Li's method
                li_features_extraction(tasks_file, li_features_subdir, imported_apis_path)
            elif choice == '14':
                # extract the features using Zhang's method
                zhang_features_extraction(tasks_file, zhang_tmp_subdir, zhang_features_subdir)
        elif choice == '5':
            set_comparison_paths()
            set_logger()
            # compare the results
            results_comparer(original_results_path, patched_results_path, summary_path, comparison_results_path)
        elif choice == '6':
            print_conf_info()
        elif choice == '7':
            deleter()
        elif choice == '8':
            cuckoo_status()
        elif choice == '0':
            sys.exit()
        else:
            print("Wrong choice!")
        print_divisor()
            

def set_logger():
    logger    = logging.getLogger("cuckoo_comparison")
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    file_path = os.path.join(general_paths['dir'],original_pes_paths['output_subdir'], "cuckoo_comparison" + time.strftime("%d_%m--%H%M%S")+".log")
    handler   = logging.FileHandler(file_path)        
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)

if __name__ == "__main__":
    menu()