import numpy as np
from config.config_paths import general_paths
import os
from utils.load_dicts import *

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


# Helper function to covert a sequence of pairs (flag, id) to a sequence
# of pairs (flag, word)
def convert_id_to_word(sequence):
    return list(map(lambda x: [x[0], id2word[x[1]] ] , sequence))


# Helper function to filter a sequence of pairs (flag, name) removing
# the padding as word
def filter_pad_word(sequence):
    return list(filter(lambda x: x[1] != '_PAD_', sequence))


# Helper function to filter a sequence of pairs (flag, name) removing
# the padding as id
def filter_pad_id(sequence):
    return list(filter(lambda x: x[1] != 0, sequence))


# Helper function to transform a names sequence in a sequence of number
# according with the encoding of our framework
def api_names_to_framework_encoding(names_sequence, enumerated_set_api):
    return list(map(lambda x: int(enumerated_set_api[x]).to_bytes(1, 'little'), names_sequence))


# Helper function to transform a sequence of pairs (flag, id) into a 
# sequence of ids
def from_pairs_to_ids_sequence(pairs):
    return list(map(lambda x: x[1], pairs))


# Helper function to transform a sequence of pairs (flag, id) into a 
# sequence of flags
def from_pairs_to_flags_sequence(pairs):
    return list(map(lambda x: x[0], pairs))


# Function to convert a the output of adversarial_sequence_generation
# into two sequences of flags and ids
# Input: sequence of sequences of pairs (flag, id)
def from_pairs_to_sequences(sequences):
    
    # From pairs to sigle sequence of ids
    ids_sequence = np.array(list(map(from_pairs_to_ids_sequence, sequences)))
    ids_sequence = ids_sequence.flatten()
    
    # From pairs to sigle sequence of flags
    flags_sequence = np.array(list(map(from_pairs_to_flags_sequence, sequences)))
    flags_sequence = flags_sequence.flatten()
    
    
    return ids_sequence, flags_sequence


# Helper function to filter a sequence of pairs (flag, name) removing
# the padding
def filter_pad(sequence):
    return list(filter(lambda x: x[1] != '_PAD_', sequence))


# Helper function to transform a names sequence in a sequence of number
# according with the encoding of our framework
def api_names_to_framework_encoding(names_sequence, enumerated_set_api):
    return list(map(lambda x: int(enumerated_set_api[x]).to_bytes(1, 'little'), names_sequence))


# Function to convert a the output of adversarial_sequence_generation
# into only two sequences of flags and words
# Input: sequence of sequences of pairs (flag, id)
def convert_id_seq_to_word_seq(sequences):
    sequences = np.array(list(map(convert_id_to_word, sequences)))
    
    sequences = np.array(list(map(filter_pad, sequences)), dtype=object)
    
    sequences = np.concatenate([s for s in sequences] )    
    
    words_sequence = np.array(list( map(lambda x: x[1], sequences) ) )
    flags_sequence = np.array(list( map(lambda x: x[0], sequences) ) )
    
    return words_sequence, flags_sequence