import shutil
from pePatcher import *
import os
import numpy as np
import logging
from multiprocessing import Pool
from tqdm import *
import json
from termcolor import cprint
from config.config_paths import paths
from config.config_general import general
from functools import partial
import struct
import pickle


"""
EXIT CODES:
0 : exe successfully patched
1 : error - impossible to determine the exe machine type
2 : error - the exe has not suffient padding in the text section for
            the hijacking logic and no sufficient padding before the
            first section to add a new header
3 : error - no one of the api calls to inject is imported in the exe
4 : error - no one of the api calls to hijack is imported in the exe
5 : error - no one of the api calls to hijack is actually called and
            used in the exe but at least of them is imported
6 : error - no executable section found in the exe
7 : error - file is not a valid PE
8 : error - impossible to determine if if Dynamic Base flag is set
9 : error - impossible to determine if Relocations Stripped flag is 
            set
10: error - Adversarial attack failed, so we do not have hijacking data for this sample
"""

# Add a new entry in the json summary
def create_result_entry(filename, exit_code, hijacked_calls, injected_calls, bits, added_section):
    entry = {
        "filename" : filename,
        "exit_code": exit_code,
        "hijacked_calls": hijacked_calls,
        "injected_calls": injected_calls,
        "bitness": bits,
        "added_section": added_section,
    }
    return entry

# Extract bitness from PE
# Source: https://stackoverflow.com/questions/1345632/determine-if-an-executable-or-library-is-32-or-64-bits-on-windows
def extract_bitness(path):
    IMAGE_FILE_MACHINE_I386 = 332
    IMAGE_FILE_MACHINE_IA64 = 512
    IMAGE_FILE_MACHINE_AMD64 = 34404
    IMAGE_FILE_MACHINE_ARM = 452
    IMAGE_FILE_MACHINE_AARCH64 = 43620

    with open(path, 'rb') as f:
        s = f.read(2)
        if s != b'MZ':
            return None
        else:
            f.seek(60)
            s = f.read(4)
            header_offset = struct.unpack('<L', s)[0]
            f.seek(header_offset + 4)
            s = f.read(2)
            machine = struct.unpack('<H', s)[0]

            if machine == IMAGE_FILE_MACHINE_I386:
                return 32
            elif machine == IMAGE_FILE_MACHINE_IA64:
                return 64
            elif machine == IMAGE_FILE_MACHINE_AMD64:
                return 64
            elif machine == IMAGE_FILE_MACHINE_ARM:
                return 32
            elif machine == IMAGE_FILE_MACHINE_AARCH64:
                return 64
            else:
                return None


# Set up the input and output directory and call the patcher on a single file
def call_patcher(filename, hijacking_data_bool=False):
    input_path          = os.path.join(input_dir, filename)
    output_path         = os.path.join(patched_dir, filename)
    hijacking_data_path = os.path.join(hijacking_data_dir, 'hijacking_data_'+filename + '.pkl')
    enumerated_set_api_path = os.path.join(hijacking_data_dir, 'enumerated_set_api_'+filename + '.pkl')

    # If specified, we load the hijacking data from file
    # otherwise load the default configuration
    if hijacking_data_bool:
        try:
            with open(hijacking_data_path, 'rb') as f:
                hijacking_data = pickle.load(f)
            with open(enumerated_set_api_path, 'rb') as f:
                enumerated_set_api = pickle.load(f)
        except FileNotFoundError as e:
            logger.warning(e)
            return create_result_entry(filename, 11, None, None, None, None)

    else:
        hijacking_data = general['hijacking_data']

    try:
        # Get bitness information
        bits = extract_bitness(input_path)

        if hijacking_data_bool:
            hijacked_calls, injected_calls, added_section = patcher(input_path, output_path, hijacking_data, list(enumerated_set_api.keys()))
        else:
            hijacked_calls, injected_calls, added_section = patcher(input_path, output_path, hijacking_data, [])
        
        # We need to transfomr the bytes in strings
        hijacked_calls = list(map(lambda x: x[0].decode(), hijacked_calls))
        injected_calls = list(map(lambda x: x.decode(), injected_calls))

        return create_result_entry(filename, 0, hijacked_calls, injected_calls, bits, added_section)
    except MachineTypeError as e:
        logger.warning(e)
        return create_result_entry(filename, 1, None, None, bits, None)
    except InsuffientPadding as e:
        logger.warning(e)
        return create_result_entry(filename, 2, None, None, bits, None)
    except NoApiCallsToInject as e:
        logger.warning(e)
        return create_result_entry(filename, 3, None, None, bits, None)
    except NoImportToHijack as e:
        logger.warning(e)
        return create_result_entry(filename, 4, None, None, bits, None)
    except ApiCallsNeverCalled as e:
        logger.warning(e)
        return create_result_entry(filename, 5, None, None, bits, None)
    except NoXSectionFound as e:
        logger.warning(e)
        return create_result_entry(filename, 6, None, None, bits, None)
    except FileNotPE as e:
        logger.warning(e)
        return create_result_entry(filename, 7, None, None, bits, None)
    except NoDynamicBaseFlag as e:
        logger.warning(e)
        return create_result_entry(filename, 8, None, None, bits, None)
    except NoRelocationsStrippedFlag as e:
        logger.warning(e)
        return create_result_entry(filename, 9, None, None, bits, None)
    except CannotInjectAPI as e:
        logger.warning(e)
        return create_result_entry(filename, 10, None, None, bits, None)


# Custom logger to avoid collision with libraries logs
def setup_logger():
    global logger 

    logger    = logging.getLogger("custom_logger")
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    handler   = logging.FileHandler(os.path.join(output_dir, time.strftime("%d%m%Y-%H%M%S")+".log"))        
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)


# Take the filenames from the input dir
def get_pe_filenames():
    global filenames

    if general["limit_pes_to_patch"] is None:
        filenames = os.listdir(input_dir)
    else:
        filenames = os.listdir(input_dir)[:general["limit_pes_to_patch"]]


# If exists, load the list of filename to ignore, otherwise initialize it to
# the empty list
def preload_ignored_pes():
    global ignored_pes

    if os.path.exists(ignored_pes_path):
        with open(ignored_pes_path, 'rb') as f:
            ignored_pes = json.load(f)
    else:
        ignored_pes = []


# Update the ignored PEs list according with the signature defined in the config file
def filter_pes():
    global ignored_pes

    ignored_pes = [] # delete old values

    # Define the signature to match
    signatures = general["signatures"]

    # If the file contains the signature, we add it to the ignored list
    for filename in tqdm(filenames):
        path = os.path.join(input_dir, filename) # path to the file
        with open(path, 'rb') as f:
            content = f.read()
            for signature in signatures:
                if signature in content:
                    ignored_pes.append(filename)
    
    # Write the list in memory
    with open(paths["ignored_pes_path"], "w") as f:
        json.dump(ignored_pes, f)
    
    print(f"\nGood news, everyone! {len(ignored_pes)} PEs have been discarded!")


# Function to launch the patcher
def patch(hijacking_data_bool=False):
    print(f"Starting the patcher ignoring {len(ignored_pes)} PEs")

    # If the destination already exists, we delete it and create a new empty one
    if os.path.exists(patched_dir):
        shutil.rmtree(patched_dir)
    os.makedirs(patched_dir)

    # Filter the filenames to ignore
    filtered_filenames = list(filter(lambda x: x not in ignored_pes, filenames))

    # Call the patcher using n_pools different pools
    with Pool(general["n_pools"]) as p:
        summary = list(tqdm(p.imap_unordered(partial(call_patcher, hijacking_data_bool = hijacking_data_bool), filtered_filenames ), total=len(filtered_filenames)))

    # Write the summary in memory
    with open(os.path.join(output_dir, paths['summary_filename']), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    

# Set the paths and filenames used by the framework
def set_paths():
    global input_dir
    global hijacking_data_dir
    global patched_dir
    global output_dir
    global ignored_pes_path

    input_dir          = os.path.join(paths["dir"], paths["input_subdir"])
    output_dir         = os.path.join(paths["dir"], paths["output_subdir"])
    patched_dir        = os.path.join(output_dir, paths["patched_subdir"])
    hijacking_data_dir = os.path.join(paths["dir"], paths["hijacking_data_subdir"])
    ignored_pes_path   = paths["ignored_pes_path"]

    # If the destination folder does not already exist, we create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


# Function to print info extracted from the patching summary
def parse_summary():
    NUM_OF_EXIT_CODES = 12
    with open(os.path.join(output_dir, paths['summary_filename']), 'r', encoding='utf-8') as f:
        summary = json.load(f)

    cprint("\nEXIT CODES:", attrs=['bold'])
    print(
    """
    0 : exe successfully patched
    1 : error - impossible to determine the exe machine type
    2 : error - the exe has not suffient padding in the text section for
                the hijacking logic and no sufficient padding before the
                first section to add a new header
    3 : error - no one the api calls to inject is imported in the exe
    4 : error - no one of the api calls to hijack is imported in the exe
    5 : error - no one of the api calls to hijack is actually called and
                used in the exe but at least of them is imported
    6 : error - no executable section found in the exe
    7 : error - file is not a valid PE
    8 : error - impossible to determine if if Dynamic Base flag is set
    9 : error - impossible to determine if Relocations Stripped flag is 
                set
    10: error - impossible to use an API to do injection
    11: error - no hijacking data found for the PE, adv_attack failed
    """)
    
    total_number = len(summary)
    print(f"The number of total attempt was: {total_number}")

    # Print general exit codes statistics
    for i in range(NUM_OF_EXIT_CODES):
        current_number = len(list(filter(lambda x: x['exit_code'] == i, summary)))
        percentage = current_number/total_number*100 if total_number != 0 else 0
        print(f"The number of PEs with exit code {i} is: {current_number} ({round(percentage,3)}%)")
    
    print('\n')

    # Extract bitness statistics
    summary_32 = list(filter(lambda x: x['bitness'] == 32, summary))
    summary_64 = list(filter(lambda x: x['bitness'] == 64, summary))
    total_32_number = len(summary_32)
    total_64_number = len(summary_64)

    percentage1 = total_32_number/total_number*100 if total_32_number != 0 else 0
    percentage2 = total_64_number/total_number*100 if total_64_number != 0 else 0
    # Print bitness statistics
    print(f"The total number of 32bit PEs was: {total_32_number} ({round(percentage1,3)}%)")
    print(f"The total number of 64bit PEs was: {total_64_number} ({round(percentage2,3)}%)")

    print('\n')

    # Print 32bit exit codes statistics
    print('*'*30)
    print("32 bit statistics")
    for i in range(NUM_OF_EXIT_CODES):
        current_number = len(list(filter(lambda x: x['exit_code'] == i, summary_32)))
        percentage = current_number/total_32_number*100 if total_32_number != 0 else 0
        print(f"The number of 32bit PEs with exit code {i} is: {current_number} ({round(percentage,3)}%)")

    print('*'*30)
    print('\n')
    
    # Print 64bit exit codes statistics
    print('*'*30)
    print("64 bit statistics")
    for i in range(NUM_OF_EXIT_CODES):
        current_number = len(list(filter(lambda x: x['exit_code'] == i, summary_64)))
        percentage = current_number/total_64_number*100 if total_64_number != 0 else 0
        print(f"The number of 64bit PEs with exit code {i} is: {current_number} ({round(percentage,3)}%)")
    print('*'*30)


# Print the menu options
def print_menu():
    print("What do you want to do?")
    print("1) Generate the list of ignored PEs")
    print("2) Start the patcher")
    print("3) Start the patcher with the hijacking data")
    print("4) Get patching summary")
    print("5) Show configuration")
    print("6) Exit")


# Menu managment
def menu():
    while True:
        print_menu()
        choice = input("> ")
        choice = choice.replace('\n', '')
        if choice == '1':
            filter_pes()
        elif choice == '2':
            patch()
        elif choice == '3':
            patch(hijacking_data_bool=True)
        elif choice == '4':
            parse_summary()
        elif choice == '5':
            print_conf()
        elif choice == '6':
            sys.exit()
        else:
            print("Wrong choice!")
        print_divisor()


# Print divisor between menu prints
def print_divisor():
    print("\n")
    print("-" * 35)
    print("\n")


def print_conf():
    print("\n")
    
    cprint("Paths Configurations", "blue", attrs=['bold'])
    print("Working directory: ",    paths['dir'])
    print("Input subdirectory: ",   paths['input_subdir'])
    print("Output subdirectory: ",  paths['output_subdir'])
    print("Patched subdirectory: ", paths['patched_subdir'])
    print("Summary filename: ",     paths['summary_filename'])
    print("Ignored PEs path: ",     paths['ignored_pes_path'])

    print("\n")
   
    cprint("General Configurations", "blue", attrs=['bold'])
    print("Multithreading pools: ", general['n_pools'])
    print("Patch up to num PEs:  ", general['limit_pes_to_patch'])
    print("Ignore Signature PEs: ", general['signatures'])


if __name__ == "__main__":
    # Set the needed paths
    set_paths()

    # Set the logger
    setup_logger()

    # Take the filenames from the input dir 
    get_pe_filenames()

    # If exists, load the list of filename to ignore
    preload_ignored_pes()

    menu()

# Used to test the framework
def test_filter_pes():
    del sys.modules['config.config']
    from config.config import paths, general
    globals()['paths'] = getattr(sys.modules['config.config'], 'paths')
    globals()['general'] = getattr(sys.modules['config.config'], 'general')
    set_paths()
    get_pe_filenames()
    filter_pes()


# Used to test the framework
def test_patch():
    del sys.modules['config.config']
    from config.config import paths, general
    globals()['paths'] = getattr(sys.modules['config.config'], 'paths')
    globals()['general'] = getattr(sys.modules['config.config'], 'general')

    set_paths()

    # Set the logger
    setup_logger()

    # Take the filenames from the input dir 
    get_pe_filenames()

    # If exists, load the list of filename to ignore
    preload_ignored_pes()

    patch()