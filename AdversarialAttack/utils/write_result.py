import pickle
from utils.helper import *
import os
from config.config_paths import general_paths


def create_injection_dict(words_sequence, flags_sequence):
    """
    Produce the API sequence to prepend to the original API call
    Input: words and flag sequences
    Output: dict(api_to_hijack: (occurence, list_of_sequences_to_inject))
    """
    
    # Initialize the needed variables
    sequence_to_prepend = [] # sequence of API to prepend
    api_dict = {} # dict to output
    set_api = set([]) # set of APIs to inject
    set_counter = {} # dict of API occurences counter


    for i, flag in enumerate(flags_sequence):
        # If flag == 0, the API is an original API
        # If flag == 1, the API has been injected
        # If flag == 2, the API belongs to the loader
        if flag == '1':
            sequence_to_prepend.append(str(words_sequence[i]))
            set_api.add(str(words_sequence[i]))
        elif flag == '2':
            # Nothing to do, just skip this API
            continue
        elif flag == '0':
            new_key = str(words_sequence[i])
            if str(new_key) in set_counter.keys():
                set_counter[str(new_key)] += 1
            else:
                set_counter[str(new_key)] = 0
            
            # Update or create the pair (api_to_hijack: (occurence, sequence_to_inject))
            if len(sequence_to_prepend) != 0:
                counter = set_counter[str(new_key)]
                if str(new_key) in api_dict.keys():
                    api_dict[str(new_key)].append((counter, sequence_to_prepend))
                else:
                    api_dict[str(new_key)] = [(counter, sequence_to_prepend)]
                sequence_to_prepend = []
    return api_dict, set_api


def create_hijacking_data(api_dict, set_api):
    """
    Function to create the hijacking data to pass to the framework
    Input: dict(api_to_hijack: list_of_sequences_to_inject)
    Output: data structure to give to the framework
    """
    # Give an encoding for the API set
    enumerated_set_api = dict(list(enumerate(set_api)))
    enumerated_set_api = dict(map(reversed, enumerated_set_api.items()))
    
    hijacking_data = [] # data to pass to the framework
    for api_to_hijack in api_dict.keys():

        # Take the longest sequence of apis to inject (entry size-1)
        # where entry size is the size of each subsequence of the
        # data to inject in the jump table
        sequences_to_inject = [i[1] for i in api_dict[api_to_hijack]]
        list_len = [len(i) for i in sequences_to_inject]
        entry_size = max(list_len) + 1

        # Create the data to inject in the jump table
        data_to_inject = b''
        last_added_pos = 0
        for occurence_num, sequence_to_inject in api_dict[api_to_hijack]:
            # Add the padding
            data_to_inject += b'\xff' * (entry_size * (occurence_num - last_added_pos if occurence_num - last_added_pos > 0 else 0))

            current_ids_to_inject = api_names_to_framework_encoding(sequence_to_inject, enumerated_set_api)
            current_ids_to_inject = b''.join(current_ids_to_inject)
            
            data_to_inject += current_ids_to_inject.ljust(entry_size, b'\xff')
            last_added_pos = occurence_num

        # '\x00' is the counter used by the framework
        # to know at which call of the hijacked API we are
        # entry_size is the max size of each subsequence (+ the terminator '\xff')
        # of the data to inject
        control_data   = b'\x00\x00'+ entry_size.to_bytes(2, 'little')
        data_to_inject = control_data + data_to_inject

        # pos 0: API to hijack
        # pos 1: sequence of api to inject, 
        # in the format of index of the API in the set (enumerated_set_api)
        # pos 2: max entry size of each subsequence of the data to inject - used by the pather
        hijacking_data.append([api_to_hijack.encode(), data_to_inject, len(data_to_inject)-len(control_data)])
    return hijacking_data, enumerated_set_api

def write_result(name_sequence, sample_name):
    """
    Function to manage the writing on file of the data produced by the attack in 
    a format that can be used by the patcher
    """
    # Produce the words and flags sequences produced by the attack strategy
    words_sequence, flags_sequence = convert_id_seq_to_word_seq(name_sequence)
    
    # Produce the dict of API to inject
    api_dict, set_api = create_injection_dict(words_sequence, flags_sequence)
    
    # Produce the hijacking data to give to the framework
    hijacking_data, enumerated_set_api = create_hijacking_data(api_dict, set_api)

    if not os.path.exists(general_paths['hijacking_data_dir']):
        os.makedirs(general_paths['hijacking_data_dir'])

    with open(os.path.join(general_paths['hijacking_data_dir'], 'start_time'), 'r') as f:
        start_time_string = f.readline()

    if not os.path.exists(os.path.join(general_paths['hijacking_data_dir'], 'pickles_'+start_time_string)):
        os.makedirs(os.path.join(general_paths['hijacking_data_dir'], 'pickles_'+start_time_string))

    # Write the hijacking data in a file as PICKLE
    filename = f'hijacking_data_{sample_name}.pkl'
    file_path = os.path.join(general_paths['hijacking_data_dir'], 'pickles_'+start_time_string, filename)
    pickle.dump(hijacking_data, open(file_path, 'wb'))

    # Write the set api used in a file as PICKLE
    filename = f'enumerated_set_api_{sample_name}.pkl'
    file_path = os.path.join(general_paths['hijacking_data_dir'], 'pickles_'+start_time_string, filename)
    pickle.dump(enumerated_set_api, open(file_path, 'wb'))


def write_combo_result(api_dict, set_api, sample_name):
    """
    Function to manage the writing on file of the data produced by the combo attack in 
    a format that can be used by the patcher
    """
    
    # Produce the hijacking data to give to the framework
    hijacking_data, enumerated_set_api = create_hijacking_data(api_dict, set_api)

    if not os.path.exists(general_paths['hijacking_data_dir']):
        os.makedirs(general_paths['hijacking_data_dir'])

    with open(os.path.join(general_paths['hijacking_data_dir'], 'start_time'), 'r') as f:
        start_time_string = f.readline()

    if not os.path.exists(os.path.join(general_paths['hijacking_data_dir'], 'pickles_'+start_time_string)):
        os.makedirs(os.path.join(general_paths['hijacking_data_dir'], 'pickles_'+start_time_string))

    # Write the hijacking data in a file as PICKLE
    filename = f'hijacking_data_{sample_name}.pkl'
    file_path = os.path.join(general_paths['hijacking_data_dir'], 'pickles_'+start_time_string, filename)
    pickle.dump(hijacking_data, open(file_path, 'wb'))

    # Write the set api used in a file as PICKLE
    filename = f'enumerated_set_api_{sample_name}.pkl'
    file_path = os.path.join(general_paths['hijacking_data_dir'], 'pickles_'+start_time_string, filename)
    pickle.dump(enumerated_set_api, open(file_path, 'wb'))


def write_evaluation_result(result):
    """
    Function to write the result of the evaluation of the attack
    """
    if not os.path.exists(general_paths['evaluation_dir']):
        os.makedirs(general_paths['evaluation_dir'])

    # Write the result in a file as PICKLE
    filename = f'evaluation_result.pkl'
    file_path = os.path.join(general_paths['evaluation_dir'], filename)
    pickle.dump(result, open(file_path, 'wb'))


def write_attack_result(info, result):
    """
    Store the result of the attack inside attack_dir
    Name of the file: {info}_attack_result.pkl
    """
    if not os.path.exists(general_paths['attack_dir']):
        os.makedirs(general_paths['attack_dir'])

    # Write the result in a file as PICKLE
    filename = f'{info}_attack_result.pkl'
    file_path = os.path.join(general_paths['attack_dir'], filename)
    pickle.dump(result, open(file_path, 'wb'))