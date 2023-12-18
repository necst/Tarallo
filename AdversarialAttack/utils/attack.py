import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../../') # Compatibility reasons
from config.config_paths import general_paths
from config.config_general import general_constant
from utils.model import *
from utils.helper import *
from utils.write_result import *
from utils.load_dicts import *
from utils.exceptions import *
from ChainFramework.config.config_paths import paths as chain_general_paths
from ChainFramework.config.config_api_args import api_args as chain_api_args
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
import __main__

def compute_entropy(file_path):
    """Compute the entropy of a file"""
    with open(file_path, 'rb') as f:
        data = f.read()
    if not data:
        return 0
    occurences = [0] * 256
    for x in data:
        try:
            occurences[x] += 1
        except ValueError:
            print(x)
            exit()
    entropy = 0
    for x in occurences:
        if x:
            p_x = float(x) / len(data)
            entropy -= p_x * math.log(p_x, 2)
    return entropy

# Function to generate a new API names sequence for each API call we want to try 
# to inject at the given index
def new_embedded_names_given_index(original_embedded_names, index, api_set_name_id):

    # Embed the api set id
    api_set_name_id = torch.tensor(api_set_name_id)  
    api_set_name_id = api_set_name_id.to(device)
    embedded_apis = model.embedder1(api_set_name_id)
    embedded_apis = embedded_apis.to('cpu').detach().numpy()

    embedded_apis = np.expand_dims(embedded_apis, axis = 1)

    # Create a M x N x A matrix repeating the original embedded sequence of names
    # M = number of apis to try
    # N = lenght of the original embedded sequence of names
    # A = size of the embedding space for names
    repeated_embedded_names = np.repeat(original_embedded_names, len(api_set_name_id), axis = 0)

    # Insert in the M x N x A matrix the new embedded apis and discard the last column to mantain the original lenght
    new_embedded_names = np.concatenate((
      repeated_embedded_names[:, :index, :], 
      embedded_apis, 
      repeated_embedded_names[:, index:original_embedded_names.shape[1]-1, :]), axis = 1)

    return new_embedded_names


def new_embedded_names_given_index_and_api(original_embedded_names, index, api_to_insert):

    # Embed the api set id
    api_name_id = torch.tensor(api_to_insert)  
    api_name_id = api_name_id.to(device)
    embedded_api = model.embedder1(api_name_id)
    embedded_api = embedded_api.to('cpu').detach().numpy()

    embedded_api = np.expand_dims(embedded_api, axis = 0)
    embedded_api = np.expand_dims(embedded_api, axis = 0)
    # Insert in the M x N x A matrix the new embedded apis and discard the last column to mantain the original lenght
    new_embedded_names = np.concatenate((
      original_embedded_names[:, :index, :], 
      embedded_api, 
      original_embedded_names[:, index:original_embedded_names.shape[1]-1, :]), axis = 1)



    return new_embedded_names



# Function to generate a new API behavior sequence for each API call we want to try to inject
# at the given index
def new_embedded_behaviors_given_index(original_embedded_behaviors, index, api_set_behavior_id):

    # Embed the api set behavior id
    api_set_behavior_id   = torch.tensor(api_set_behavior_id)
    api_set_behavior_id   = api_set_behavior_id.to(device)
    embedded_behaviors = model.embedder2(api_set_behavior_id)
    embedded_behaviors = embedded_behaviors.to('cpu').detach().numpy()
  
    # Create a M x N x A matrix repeating the original embedded sequence of behaviors
    # M = number of apis to try
    # N = lenght of the original embedded sequence of bahviors
    # A = size of the embedding space for behaviors
    repeated_original_embedded_behaviors = np.repeat(original_embedded_behaviors, len(api_set_behavior_id), axis = 0)

    # Insert in the M x N x A matrix the new embedded apis and discard the last column to mantain the original lenght
    new_embedded_behaviors = np.concatenate((
      repeated_original_embedded_behaviors[:, :index*4, :], 
      embedded_behaviors, 
      repeated_original_embedded_behaviors[:, index*4:original_embedded_behaviors.shape[1]-4, :]),
      axis = 1)
    
    return new_embedded_behaviors


def new_embedded_behaviors_given_index_and_api(original_embedded_behaviors, index, api_behavior_id):

    # Embed the api set behavior id
    api_behavior_id   = torch.tensor(api_behavior_id)
    api_behavior_id   = api_behavior_id.to(device)
    embedded_behaviors = model.embedder2(api_behavior_id)
    embedded_behaviors = embedded_behaviors.to('cpu').detach().numpy()
  
    embedded_behaviors = np.expand_dims(embedded_behaviors, axis = 0)

    # Insert in the M x N x A matrix the new embedded apis and discard the last column to mantain the original lenght
    new_embedded_behaviors = np.concatenate((
      original_embedded_behaviors[:, :index*4, :], 
      embedded_behaviors, 
      original_embedded_behaviors[:, index*4:original_embedded_behaviors.shape[1]-4, :]),
      axis = 1)
    
    return new_embedded_behaviors


def compute_best_position(jacobian_names, total_names, sample_name, position_ub):
    """
    Function to compute the best position where to inject the new API
    Input: jacobian w.r.t. the names as numpy array, position upperbound
    Output: best position
    """
    logger = logging.getLogger("adversarial_attack")
    # Remove extra dimensions
    jacobian_names = np.squeeze(jacobian_names, axis = (0,1,2))
    
    # Compute the norm over the second dimension -> to avoid loops we use
    # two different norms alternately
    if it_counter % 4 == 0:
        norm_jacobian = np.linalg.norm(jacobian_names, ord=1, axis = 1)
    else:
        norm_jacobian = np.linalg.norm(jacobian_names, ord=-1, axis = 1)
    
    
    # If the upperbound is near to the maximum size of the sequence, we exclude 
    # the last positions to avoid loops
    #print(norm_jacobian)
    if position_ub > 991:
       norm_jacobian = norm_jacobian[:991]
    else:
       norm_jacobian = norm_jacobian[:position_ub+1]

    count = 0    
    while True:
        argmax = np.argmax(norm_jacobian)

        if total_names[argmax][0] != 2:
            # If the position is acceptable, we return it
            break
        else:
            # If the position is not acceptable, we set the norm to zero and we repeat
            # the process
            #sys.stdout.write("Position not acceptable")
            #sys.stdout.flush()
            logger.debug(f"[{sample_name[:]}] Position {argmax} not acceptable")
            norm_jacobian[argmax] = -1            

        count += 1
        if count > min(len(total_names), 999):
            # Fail if we cannot find a position after checking them all
            raise CannotFindBestPositionException("Cannot compute the best position - iterated")
        
    return argmax


# Function to compute the Jacobian
def compute_jacobian(model, embedded_names, embedded_behaviors, total_names, sample_name, position_ub):
    # The Jacobian computations requires the gradients, hence the model shoul be in train mode
    # However, it must not to learn here
    model.train()
    model.training = False

    logger = logging.getLogger("adversarial_attack")
    # Compute the Jacobian w.r.t. the embedded inputs
    jacobian_names, jacobian_behaviors = torch.autograd.functional.jacobian(model.diff_forward, (embedded_names, embedded_behaviors) )

    # Compute the best position where to inject the new API
    logger.debug(f"[{sample_name[:]}] computing best position")
    best_position = compute_best_position(jacobian_names.to('cpu').numpy(), total_names, sample_name, position_ub)
    
    # Compute the sign of the Jacobian 
    sign_jacobian_names = np.sign(jacobian_names[0][0].to('cpu').detach().numpy())
    sign_jacobian_behaviors = np.sign(jacobian_behaviors[0][0].to('cpu').detach().numpy())

    model.eval()
    logger.debug(f"[{sample_name[:]}] Jacobian OK")
    return best_position, sign_jacobian_names, sign_jacobian_behaviors


def compute_jacobian_Rosenberg(model, embedded_names, embedded_behaviors):
    # The Jacobian computations requires the gradients, hence the model shoul be in train mode
    # However, it must not to learn here
    model.train()
    model.training = False

    logger = logging.getLogger("adversarial_attack")
    # Compute the Jacobian w.r.t. the embedded inputs
    jacobian_names, jacobian_behaviors = torch.autograd.functional.jacobian(model.diff_forward, (embedded_names, embedded_behaviors) )

    # Compute the sign of the Jacobian 
    sign_jacobian_names = np.sign(jacobian_names[0][0].to('cpu').detach().numpy())
    sign_jacobian_behaviors = np.sign(jacobian_behaviors[0][0].to('cpu').detach().numpy())

    model.eval()
    return sign_jacobian_names, sign_jacobian_behaviors


# Transform the original API ids sequence in an attacking API sequence
# that can be fed to the model to check if *our* attack is successful
# Input: original API ids sequence
# Output: attacking API ids sequence
def generate_attacking_sequence(original_api_id_sequence, injection_dict):
    
    # Split the original sequence in names and behaviors
    original_api_name_id_sequence, _ = original_api_id_sequence.unsqueeze(0).split([1000, 4000], 1)
    
    # Tranform the the original sequence of names for convencience
    original_api_name_id_sequence = torch.squeeze(original_api_name_id_sequence)
    original_api_name_id_sequence = original_api_name_id_sequence.detach().numpy()
    
    # Transform the sequence of name ids to a sequence of names
    original_api_name_sequence = np.array(list(map(lambda x: id2word[x], original_api_name_id_sequence)))
    
    # Initialize the attacking sequence of APIs name
    attacking_api_name_sequence = [] 
    
    for api_name in original_api_name_sequence:
        if api_name in injection_dict.keys():
            attacking_api_name_sequence += injection_dict[api_name] + [api_name]
            print(len(attacking_api_name_sequence))
            print('\n')
        else:
            attacking_api_name_sequence.append(api_name)

    attacking_api_id_sequence       = np.array(list(map(lambda x: word2id[x], attacking_api_name_sequence) ))
    attacking_api_behavior_sequence = behaviors_from_name(attacking_api_name_sequence)
    
    # Splitting in 1000 API-len windows
    attacking_api_id_sequence_windows     = np.array_split(attacking_api_id_sequence, 1000)
    
    
    return attacking_api_id_sequence


def adversarial_sequence_generation(api_sequence, model, api_set, mode, called_apis, sample_name, max_num_iterations=1500):
    logger = logging.getLogger('adversarial_attack')
    global it_counter

    model.eval()

    # Number of injected APIs counter
    apis_counter = 0

    # Number of iteractions counter
    it_counter = 0

    # Final sequence of all API names with the right order
    output_name_sequence = []

    # Split the original sequence in names and behaviors
    original_names, original_behaviors = api_sequence.split([1000, 4000], 1)

    # From api name to id
    api_set_name_id = np.array(list(map(lambda x: word2id[x], api_set)))

    # From api name to behavior
    api_set_behavior = np.array(list(map(lambda x: word2behavior[x], api_set)))

    # From behavior to behavior id
    api_set_behavior_id = np.array(list(map(tuple_to_id, api_set_behavior)))


    # Initilize the position upperbound
    position_ub = len(list(filter(lambda x: x!=0, original_names[0])))
    # print(f'Initial value for the UB: {position_ub}')
    
    # Original names is set each time to a window of 1000 APIs: 
    # when the window.shape[1] (num of API) is 0 it means that 
    # there are no meaningful (!= PAD) APIs in that window
    
    logger.debug(f'[{sample_name[:]}] Started' )

    while original_names.shape[1] != 0:
        # Add padding to the sequences
        padding_behaviors  = torch.tensor([24, 14, 14, 14] * (1000-original_names.shape[1]), dtype = int)
        padding_behaviors  = torch.unsqueeze(padding_behaviors, 0)
        original_names     = torch.nn.functional.pad(input=original_names, pad=(0, (1000-original_names.shape[1])), mode='constant', value=0)
        original_behaviors = torch.cat((original_behaviors, padding_behaviors), dim = 1)
        
        # Initialize the variables that stores all the APIs in the total sequence as pair.
        # The first binary element shows if:
        #  - they belong to the original sequence (0) or
        #  - they have been injected by the tool (1) 
        #  - they probably belong to the loader (2) 
        total_names     = np.array(list(map(lambda x: [2 if id2word[int(x)].encode() not in called_apis else 0, x], original_names[0])))
        total_behaviors = np.split(original_behaviors, 1000, axis = 1)
        total_behaviors = np.array(list(map(lambda x: [0, np.array(x[0])], total_behaviors)), dtype = object)

        # Send data to device
        original_names     = original_names.to(device)
        original_behaviors = original_behaviors.to(device)

        # Embed using the model ebedding layers
        embedded_names = model.embedder1(original_names)
        embedded_behaviors = model.embedder2(original_behaviors)


        # Compute the Jacobian sign
        position, sign_jacobian_names, sign_jacobian_behaviors = compute_jacobian(
                model, embedded_names, embedded_behaviors, 
                total_names, sample_name, position_ub)

        # Repeat M times the Jacobian sign to allow fast difference computation
        # M = number of apis to try
        if mode == 'names':
            sign_jacobian_names_repeated     = np.repeat(sign_jacobian_names, len(api_set), axis = 0)
        elif mode == 'behaviors':
            sign_jacobian_behaviors_repeated = np.repeat(sign_jacobian_behaviors, len(api_set), axis = 0)
        else:
            raise('Wrong mode')
        
        # Send back to CPU
        embedded_names = embedded_names.to('cpu').detach().numpy()
        embedded_behaviors = embedded_behaviors.to('cpu').detach().numpy()

        it_counter = 0
        score_dict = {}
        while it_counter < max_num_iterations:  


            # Update iteractions counter
            it_counter += 1

            current_score = model.single_score(embedded_names, embedded_behaviors)

            score_dict[current_score] = [total_names, total_behaviors]
                
            # Generate a new API name and behavior sequence for each API call we want to try to inject
            new_embedded_names     = new_embedded_names_given_index(embedded_names, position, api_set_name_id)
            new_embedded_behaviors = new_embedded_behaviors_given_index(embedded_behaviors, position, api_set_behavior_id)

            # Repeat M times the original sequences to allow fast difference computation
            # M = number of apis to try
            embedded_names_repeated     = np.repeat(embedded_names, len(api_set), axis = 0)
            embedded_behaviors_repeated = np.repeat(embedded_behaviors, len(api_set), axis = 0)

            #print(f'Before sub (names) {embedded_names_repeated.shape} - {new_embedded_names.shape}')
            #print(f'Before sub (behaviors) {embedded_behaviors_repeated.shape} - {new_embedded_behaviors.shape}')

            if mode == 'names':
                # Compute the sign of the difference between the previous embedded names sequence and each of the new 
                # generated embedded names sequence
                sub_embedded_names      = np.subtract(embedded_names_repeated, new_embedded_names)
                sign_sub_embedded_names = np.sign(sub_embedded_names)
                
                # Compute the difference between the sign of the difference and the sign on the Jacobian
                sub_signs_names     = np.subtract(sign_sub_embedded_names , sign_jacobian_names_repeated)
                
                # Compute the norm of the signs difference
                norm_sum     = np.linalg.norm(sub_signs_names, ord='fro', axis = (1,2))
            elif mode == 'behaviors':
                # Compute the sign of the difference between the previous embedded behaviors sequence and each of the new 
                # generated embedded behaviors sequence
                sub_embedded_behaviors      = np.subtract(embedded_behaviors_repeated, new_embedded_behaviors)
                sign_sub_embedded_behaviors = np.sign(sub_embedded_behaviors)
               
                # Compute the difference between the sign of the difference and the sign on the Jacobian
                sub_signs_behaviors = np.subtract(sign_sub_embedded_behaviors , sign_jacobian_behaviors_repeated)
                
                # Compute the norm of the signs difference
                norm_sum = np.linalg.norm(sub_signs_behaviors, ord='fro' , axis = (1,2))
            else:
                raise('Wrong mode')

            # Compute the argmin to decide the API call to inject
            selected_idx = np.argmin(norm_sum)

            # Set the current sequence equal to the selected ones
            embedded_names = new_embedded_names[selected_idx]
            embedded_names = np.expand_dims(embedded_names, axis = 0)
            embedded_behaviors = new_embedded_behaviors[selected_idx]
            embedded_behaviors = np.expand_dims(embedded_behaviors, axis = 0)

            # Update the total sequences
            total_names = np.concatenate((
                    total_names[:position],
                    np.array([(1, api_set_name_id[selected_idx])]),
                    total_names[position:]))
            total_behaviors = np.concatenate((
                    total_behaviors[:position],
                    np.array([(1, api_set_behavior_id[selected_idx])], dtype=object),
                    total_behaviors[position:]))
            
            #n = torch.tensor(list(map(lambda x: x[1], total_names[:1000]))).unsqueeze(0).to(device)
            #b = torch.tensor(list(map(lambda x: x[1], total_behaviors[:1000]))).flatten().unsqueeze(0).to(device)

            #print(f'single inference: {model.single_inference(embedded_names, embedded_behaviors)}')
            #print(f'single inference: {model.single_inference(model.embedder1(n).cpu().detach().numpy(), model.embedder2(b).cpu().detach().numpy())}')
            #print(f'single score: {model.single_score(embedded_names, embedded_behaviors)}')
            #print(f'pred given split {model.pred_given_split(n, b)}')
            #print(f'score given split {model.score_given_split(n, b)}')

            # Compute the Jacobian sign
            position, sign_jacobian_names, sign_jacobian_behaviors = compute_jacobian(
                    model,
                    torch.tensor(embedded_names).to(device), 
                    torch.tensor(embedded_behaviors).to(device),
                    total_names,
                    sample_name,
                    position_ub)
            
            
            # Repeat M times the Jacobian sign to allow fast difference computation
            # M = number of apis to try
            sign_jacobian_names_repeated = np.repeat(sign_jacobian_names, len(api_set), axis = 0)
            sign_jacobian_behaviors_repeated = np.repeat(sign_jacobian_behaviors, len(api_set), axis = 0)
        
            # Dynamically adjust the random upperbound
            position_ub = position_ub + 1 if position_ub < 999 else 999


        # Find the key with the smallest floating point value
        min_key = min(score_dict, key=float)

        if min_key > general_constant['score_threshold']:
            apis_counter = 9999999
            break
        
        print()
        print(f'[{sample_name[:]}] Score: {min_key} - Iterations: {it_counter}')

        # Retrieve the corresponding array
        total_names, total_behaviors = score_dict[min_key]

        # Update the number of injected APIs counter
        apis_counter += len(list(filter(lambda x: x[0] == 1, total_names[:1000])))
        
        output_name_sequence.append(total_names[:1000])
        total_names = total_names[1000:]
        total_behaviors = total_behaviors[1000:]

        

        # Ignore the added APIs in the new window and the padding
        total_names     = np.array(list(filter(lambda x: (x[0] == 0 or x[0] == 2) and int(x[1]) != 0, total_names)))
        total_behaviors = np.array(list(filter(lambda x: (x[0] == 0 or x[0] == 2) and (np.array([24, 14, 14, 14]) != x[1]).any(), total_behaviors)))
        
        # Resize properly and set variables for next iteration
        original_names = torch.tensor(list(map(lambda x: x[1], total_names)))
        original_names = torch.unsqueeze(original_names, 0)
        
        original_behaviors = torch.tensor(np.array(list(map(lambda x: x[1], total_behaviors))))
        original_behaviors = torch.flatten(original_behaviors)
        original_behaviors = torch.unsqueeze(original_behaviors, 0)
        
        # Update the random upperbound
        position_ub = original_names.shape[1] - 1
                             
    return apis_counter, output_name_sequence


def adversarial_sequence_generation_legacy(api_sequence, model, api_set, mode, called_apis, sample_name, max_num_iterations=1500):
    logger = logging.getLogger('adversarial_attack')
    global it_counter

    model.eval()

    # Number of injected APIs counter
    apis_counter = 0

    # Number of iteractions counter
    it_counter = 0

    # Final sequence of all API names with the right order
    output_name_sequence = []

    # Split the original sequence in names and behaviors
    original_names, original_behaviors = api_sequence.split([1000, 4000], 1)

    # From api name to id
    api_set_name_id = np.array(list(map(lambda x: word2id[x], api_set)))

    # From api name to behavior
    api_set_behavior = np.array(list(map(lambda x: word2behavior[x], api_set)))

    # From behavior to behavior id
    api_set_behavior_id = np.array(list(map(tuple_to_id, api_set_behavior)))


    # Initilize the position upperbound
    position_ub = len(list(filter(lambda x: x!=0, original_names[0])))
    # print(f'Initial value for the UB: {position_ub}')
    
    # Original names is set each time to a window of 1000 APIs: 
    # when the window.shape[1] (num of API) is 0 it means that 
    # there are no meaningful (!= PAD) APIs in that window
    
    logger.debug(f'[{sample_name[:]}] Started' )

    while original_names.shape[1] != 0 and it_counter < max_num_iterations:
        # Add padding to the sequences
        padding_behaviors  = torch.tensor([24, 14, 14, 14] * (1000-original_names.shape[1]), dtype = int)
        padding_behaviors  = torch.unsqueeze(padding_behaviors, 0)
        original_names     = torch.nn.functional.pad(input=original_names, pad=(0, (1000-original_names.shape[1])), mode='constant', value=0)
        original_behaviors = torch.cat((original_behaviors, padding_behaviors), dim = 1)
        
        # Initialize the variables that stores all the APIs in the total sequence as pair.
        # The first binary element shows if:
        #  - they belong to the original sequence (0) or
        #  - they have been injected by the tool (1) 
        #  - they probably belong to the loader (2) 
        total_names     = np.array(list(map(lambda x: [2 if id2word[int(x)].encode() not in called_apis else 0, x], original_names[0])))
        total_behaviors = np.split(original_behaviors, 1000, axis = 1)
        total_behaviors = np.array(list(map(lambda x: [0, np.array(x[0])], total_behaviors)), dtype = object)

        # Send data to device
        original_names     = original_names.to(device)
        original_behaviors = original_behaviors.to(device)

        # Embed using the model ebedding layers
        embedded_names = model.embedder1(original_names)
        embedded_behaviors = model.embedder2(original_behaviors)


        # Compute the Jacobian sign
        position, sign_jacobian_names, sign_jacobian_behaviors = compute_jacobian(
                model, embedded_names, embedded_behaviors, 
                total_names, sample_name, position_ub)

        # Repeat M times the Jacobian sign to allow fast difference computation
        # M = number of apis to try
        if mode == 'names':
            sign_jacobian_names_repeated     = np.repeat(sign_jacobian_names, len(api_set), axis = 0)
        elif mode == 'behaviors':
            sign_jacobian_behaviors_repeated = np.repeat(sign_jacobian_behaviors, len(api_set), axis = 0)
        else:
            raise('Wrong mode')
        
        # Send back to CPU
        embedded_names = embedded_names.to('cpu').detach().numpy()
        embedded_behaviors = embedded_behaviors.to('cpu').detach().numpy()

        while model.single_inference_thr(embedded_names, embedded_behaviors) != 0 and it_counter < max_num_iterations:     
            # Update iteractions counter
            it_counter += 1
                
            # Generate a new API name and behavior sequence for each API call we want to try to inject
            new_embedded_names     = new_embedded_names_given_index(embedded_names, position, api_set_name_id)
            new_embedded_behaviors = new_embedded_behaviors_given_index(embedded_behaviors, position, api_set_behavior_id)

            # Repeat M times the original sequences to allow fast difference computation
            # M = number of apis to try
            embedded_names_repeated     = np.repeat(embedded_names, len(api_set), axis = 0)
            embedded_behaviors_repeated = np.repeat(embedded_behaviors, len(api_set), axis = 0)

            #print(f'Before sub (names) {embedded_names_repeated.shape} - {new_embedded_names.shape}')
            #print(f'Before sub (behaviors) {embedded_behaviors_repeated.shape} - {new_embedded_behaviors.shape}')

            if mode == 'names':
                # Compute the sign of the difference between the previous embedded names sequence and each of the new 
                # generated embedded names sequence
                sub_embedded_names      = np.subtract(embedded_names_repeated, new_embedded_names)
                sign_sub_embedded_names = np.sign(sub_embedded_names)
                
                # Compute the difference between the sign of the difference and the sign on the Jacobian
                sub_signs_names     = np.subtract(sign_sub_embedded_names , sign_jacobian_names_repeated)
                
                # Compute the norm of the signs difference
                norm_sum     = np.linalg.norm(sub_signs_names, ord='fro', axis = (1,2))
            elif mode == 'behaviors':
                # Compute the sign of the difference between the previous embedded behaviors sequence and each of the new 
                # generated embedded behaviors sequence
                sub_embedded_behaviors      = np.subtract(embedded_behaviors_repeated, new_embedded_behaviors)
                sign_sub_embedded_behaviors = np.sign(sub_embedded_behaviors)
               
                # Compute the difference between the sign of the difference and the sign on the Jacobian
                sub_signs_behaviors = np.subtract(sign_sub_embedded_behaviors , sign_jacobian_behaviors_repeated)
                
                # Compute the norm of the signs difference
                norm_sum = np.linalg.norm(sub_signs_behaviors, ord='fro' , axis = (1,2))
            else:
                raise('Wrong mode')

            # Compute the argmin to decide the API call to inject
            selected_idx = np.argmin(norm_sum)

            # Set the current sequence equal to the selected ones
            embedded_names = new_embedded_names[selected_idx]
            embedded_names = np.expand_dims(embedded_names, axis = 0)
            embedded_behaviors = new_embedded_behaviors[selected_idx]
            embedded_behaviors = np.expand_dims(embedded_behaviors, axis = 0)

            # Update the total sequences
            total_names = np.concatenate((
                    total_names[:position],
                    np.array([(1, api_set_name_id[selected_idx])]),
                    total_names[position:]))
            total_behaviors = np.concatenate((
                    total_behaviors[:position],
                    np.array([(1, api_set_behavior_id[selected_idx])], dtype=object),
                    total_behaviors[position:]))
            
            #n = torch.tensor(list(map(lambda x: x[1], total_names[:1000]))).unsqueeze(0).to(device)
            #b = torch.tensor(list(map(lambda x: x[1], total_behaviors[:1000]))).flatten().unsqueeze(0).to(device)

            #print(f'single inference: {model.single_inference(embedded_names, embedded_behaviors)}')
            #print(f'single inference: {model.single_inference(model.embedder1(n).cpu().detach().numpy(), model.embedder2(b).cpu().detach().numpy())}')
            #print(f'single score: {model.single_score(embedded_names, embedded_behaviors)}')
            #print(f'pred given split {model.pred_given_split(n, b)}')
            #print(f'score given split {model.score_given_split(n, b)}')

            # Compute the Jacobian sign
            position, sign_jacobian_names, sign_jacobian_behaviors = compute_jacobian(
                    model,
                    torch.tensor(embedded_names).to(device), 
                    torch.tensor(embedded_behaviors).to(device),
                    total_names,
                    sample_name,
                    position_ub)
            
            
            # Repeat M times the Jacobian sign to allow fast difference computation
            # M = number of apis to try
            sign_jacobian_names_repeated = np.repeat(sign_jacobian_names, len(api_set), axis = 0)
            sign_jacobian_behaviors_repeated = np.repeat(sign_jacobian_behaviors, len(api_set), axis = 0)
        
            # Dynamically adjust the random upperbound
            position_ub = position_ub + 1 if position_ub < 999 else 999

        # Update the number of injected APIs counter
        if it_counter == max_num_iterations:
            apis_counter = 999999999
            it_counter = 999999999
            break

        apis_counter += len(list(filter(lambda x: x[0] == 1, total_names[:1000])))
        it_counter = apis_counter
        
        output_name_sequence.append(total_names[:1000])
        total_names = total_names[1000:]
        total_behaviors = total_behaviors[1000:]

        # Ignore the added APIs in the new window and the padding
        total_names     = np.array(list(filter(lambda x: (x[0] == 0 or x[0] == 2) and int(x[1]) != 0, total_names)))
        total_behaviors = np.array(list(filter(lambda x: (x[0] == 0 or x[0] == 2) and (np.array([24, 14, 14, 14]) != x[1]).any(), total_behaviors)))
        
        # Resize properly and set variables for next iteration
        original_names = torch.tensor(list(map(lambda x: x[1], total_names)))
        original_names = torch.unsqueeze(original_names, 0)
        
        original_behaviors = torch.tensor(np.array(list(map(lambda x: x[1], total_behaviors))))
        original_behaviors = torch.flatten(original_behaviors)
        original_behaviors = torch.unsqueeze(original_behaviors, 0)
        
        # Update the random upperbound
        position_ub = original_names.shape[1] - 1
                             
    
    return apis_counter, output_name_sequence


def combo_adversarial_sequence_generation(api_sequence, model, api_set, mode, called_apis, sample_name, max_num_iterations=1500):
    logger = logging.getLogger('adversarial_attack')
    global it_counter

    model.eval()

    # Number of injected APIs counter
    apis_counter = 0

    # Number of iteractions counter
    it_counter = 0

    # Final sequence of all API names with the right order
    output_name_sequence = []

    # Split the original sequence in names and behaviors
    original_names = api_sequence[0]
    original_behaviors = api_sequence[1]

    # From api name to id
    api_set_name_id = np.array(list(map(lambda x: word2id[x], api_set)))

    # From api name to behavior
    api_set_behavior = np.array(list(map(lambda x: word2behavior[x], api_set)))

    # From behavior to behavior id
    api_set_behavior_id = np.array(list(map(tuple_to_id, api_set_behavior)))

    # Initilize the position upperbound
    position_ub = min(len(list(filter(lambda x: x!=0, original_names[0]))), 999)
    # print(f'Initial value for the UB: {position_ub}')
    
    # Original names is set each time to a window of 1000 APIs: 
    # when the window.shape[1] (num of API) is 0 it means that 
    # there are no meaningful (!= PAD) APIs in that window
    
    logger.debug(f'[{sample_name[:]}] Started' )
    #print('start')
    while original_names.shape[1] != 0:

        # Initialize the variables that stores all the APIs in the total sequence as pair.
        # The first binary element shows if:
        #  - they belong to the original sequence (0) or
        #  - they have been injected by the tool (1) 
        #  - they probably belong to the loader (2) 
        total_names     = np.array(list(map(lambda x: [2 if id2word[int(x)].encode() not in called_apis else 0, x], original_names[0])))
        total_behaviors = np.split(original_behaviors, original_behaviors.shape[1]//4, axis = 1)
        total_behaviors = np.array(list(map(lambda x: [0, np.array(x[0])], total_behaviors)), dtype = object)

        if original_names.shape[1] <= 1000:
            # Add padding to the sequences
            padding_behaviors  = torch.tensor([24, 14, 14, 14] * (1000-original_names.shape[1]), dtype = int)
            padding_behaviors  = torch.unsqueeze(padding_behaviors, 0)
            original_names     = torch.nn.functional.pad(input=original_names, pad=(0, (1000-original_names.shape[1])), mode='constant', value=0)
            original_behaviors = torch.cat((original_behaviors, padding_behaviors), dim = 1)
        else: 
            original_names = original_names[:, :1000]
            original_behaviors = original_behaviors[:, :1000*4]
        

        # Send data to device
        original_names     = original_names.to(device)
        original_behaviors = original_behaviors.to(device)

        # Embed using the model ebedding layers
        embedded_names = model.embedder1(original_names)
        embedded_behaviors = model.embedder2(original_behaviors)


        # Compute the Jacobian sign
        position, sign_jacobian_names, sign_jacobian_behaviors = compute_jacobian(
            model, embedded_names, embedded_behaviors, 
            total_names, sample_name, position_ub)
    
        # Repeat M times the Jacobian sign to allow fast difference computation
        # M = number of apis to try
        if mode == 'names':
            sign_jacobian_names_repeated     = np.repeat(sign_jacobian_names, len(api_set), axis = 0)
        elif mode == 'behaviors':
            sign_jacobian_behaviors_repeated = np.repeat(sign_jacobian_behaviors, len(api_set), axis = 0)
        else:
            raise('Wrong mode')
        
        # Send back to CPU
        embedded_names = embedded_names.to('cpu').detach().numpy()
        embedded_behaviors = embedded_behaviors.to('cpu').detach().numpy()

        it_counter = 0
        score_dict = {}
        while it_counter < max_num_iterations:      
            # Update iteractions counter
            it_counter += 1

            current_score = model.single_score(embedded_names, embedded_behaviors)

            score_dict[current_score] = [total_names, total_behaviors]
                
            # Generate a new API name and behavior sequence for each API call we want to try to inject
            new_embedded_names     = new_embedded_names_given_index(embedded_names, position, api_set_name_id)
            new_embedded_behaviors = new_embedded_behaviors_given_index(embedded_behaviors, position, api_set_behavior_id)

            # Repeat M times the original sequences to allow fast difference computation
            # M = number of apis to try
            embedded_names_repeated     = np.repeat(embedded_names, len(api_set), axis = 0)
            embedded_behaviors_repeated = np.repeat(embedded_behaviors, len(api_set), axis = 0)

            #print(f'Before sub (names) {embedded_names_repeated.shape} - {new_embedded_names.shape}')
            #print(f'Before sub (behaviors) {embedded_behaviors_repeated.shape} - {new_embedded_behaviors.shape}')

            if mode == 'names':
                # Compute the sign of the difference between the previous embedded names sequence and each of the new 
                # generated embedded names sequence
                sub_embedded_names      = np.subtract(embedded_names_repeated, new_embedded_names)
                sign_sub_embedded_names = np.sign(sub_embedded_names)
                
                # Compute the difference between the sign of the difference and the sign on the Jacobian
                sub_signs_names     = np.subtract(sign_sub_embedded_names , sign_jacobian_names_repeated)
                
                # Compute the norm of the signs difference
                norm_sum     = np.linalg.norm(sub_signs_names, ord='fro', axis = (1,2))
            elif mode == 'behaviors':
                # Compute the sign of the difference between the previous embedded behaviors sequence and each of the new 
                # generated embedded behaviors sequence
                sub_embedded_behaviors      = np.subtract(embedded_behaviors_repeated, new_embedded_behaviors)
                sign_sub_embedded_behaviors = np.sign(sub_embedded_behaviors)
               
                # Compute the difference between the sign of the difference and the sign on the Jacobian
                sub_signs_behaviors = np.subtract(sign_sub_embedded_behaviors , sign_jacobian_behaviors_repeated)
                
                # Compute the norm of the signs difference
                norm_sum = np.linalg.norm(sub_signs_behaviors, ord='fro' , axis = (1,2))
            else:
                raise('Wrong mode')

            # Compute the argmin to decide the API call to inject
            selected_idx = np.argmin(norm_sum)

            # Set the current sequence equal to the selected ones
            embedded_names = new_embedded_names[selected_idx]
            embedded_names = np.expand_dims(embedded_names, axis = 0)
            embedded_behaviors = new_embedded_behaviors[selected_idx]
            embedded_behaviors = np.expand_dims(embedded_behaviors, axis = 0)

            # Update the total sequences
            total_names = np.concatenate((
                    total_names[:position],
                    np.array([(1, api_set_name_id[selected_idx])]),
                    total_names[position:]))
            total_behaviors = np.concatenate((
                    total_behaviors[:position],
                    np.array([(1, api_set_behavior_id[selected_idx])], dtype=object),
                    total_behaviors[position:]))
            
            #n = torch.tensor(list(map(lambda x: x[1], total_names[:1000]))).unsqueeze(0)
            #b = torch.tensor(list(map(lambda x: x[1], total_behaviors[:1000]))).flatten().unsqueeze(0)

            #print(f'single inference: {model.single_inference(embedded_names, embedded_behaviors)}')
            #print(f'single inference: {model.single_inference(model.embedder1(n).detach().numpy(), model.embedder2(b).detach().numpy())}')
            #print(f'single score: {model.single_score(embedded_names, embedded_behaviors)}')
            #print(f'pred given split {model.pred_given_split(n, b)}')
            #print(f'score given split {model.score_given_split(n, b)}')


            # Compute the Jacobian sign
            position, sign_jacobian_names, sign_jacobian_behaviors = compute_jacobian(
                    model,
                    torch.tensor(embedded_names).to(device), 
                    torch.tensor(embedded_behaviors).to(device),
                    total_names,
                    sample_name,
                    position_ub)
     
            
            # Repeat M times the Jacobian sign to allow fast difference computation
            # M = number of apis to try
            sign_jacobian_names_repeated = np.repeat(sign_jacobian_names, len(api_set), axis = 0)
            sign_jacobian_behaviors_repeated = np.repeat(sign_jacobian_behaviors, len(api_set), axis = 0)
        
            # Dynamically adjust the random upperbound
            position_ub = position_ub + 1 if position_ub < 999 else 999

         # Find the key with the smallest floating point value
        min_key = min(score_dict, key=float)

        if min_key > general_constant['score_threshold']:
            apis_counter = 9999999
            break
        
        print()
        print(f'[{sample_name[:]}] Score: {min_key} - Iterations: {it_counter}')

        total_names, total_behaviors = score_dict[min_key]



        apis_counter += len(list(filter(lambda x: x[0] == 1, total_names[:1000])))
        
        output_name_sequence.append(total_names[:1000])
        total_names = total_names[1000:]
        total_behaviors = total_behaviors[1000:]

        # Ignore the added APIs in the new window and the padding
        total_names     = np.array(list(filter(lambda x: (x[0] == 0 or x[0] == 2) and int(x[1]) != 0, total_names)))
        total_behaviors = np.array(list(filter(lambda x: (x[0] == 0 or x[0] == 2) and (np.array([24, 14, 14, 14]) != x[1]).any(), total_behaviors)))
        
        # Resize properly and set variables for next iteration
        original_names = torch.tensor(list(map(lambda x: x[1], total_names)))
        original_names = torch.unsqueeze(original_names, 0)
        
        original_behaviors = torch.tensor(np.array(list(map(lambda x: x[1], total_behaviors))))
        original_behaviors = torch.flatten(original_behaviors)
        original_behaviors = torch.unsqueeze(original_behaviors, 0)
        
        # Update the random upperbound
        position_ub = original_names.shape[1] - 1

    
    return apis_counter, output_name_sequence



def combo_adversarial_sequence_generation_legacy(api_sequence, model, api_set, mode, called_apis, sample_name, max_num_iterations=1500):
    logger = logging.getLogger('adversarial_attack')
    global it_counter

    model.eval()

    # Number of injected APIs counter
    apis_counter = 0

    # Number of iteractions counter
    it_counter = 0

    # Final sequence of all API names with the right order
    output_name_sequence = []

    # Split the original sequence in names and behaviors
    original_names = api_sequence[0]
    original_behaviors = api_sequence[1]

    # From api name to id
    api_set_name_id = np.array(list(map(lambda x: word2id[x], api_set)))

    # From api name to behavior
    api_set_behavior = np.array(list(map(lambda x: word2behavior[x], api_set)))

    # From behavior to behavior id
    api_set_behavior_id = np.array(list(map(tuple_to_id, api_set_behavior)))

    # Initilize the position upperbound
    position_ub = min(len(list(filter(lambda x: x!=0, original_names[0]))), 999)
    # print(f'Initial value for the UB: {position_ub}')
    
    # Original names is set each time to a window of 1000 APIs: 
    # when the window.shape[1] (num of API) is 0 it means that 
    # there are no meaningful (!= PAD) APIs in that window
    
    logger.debug(f'[{sample_name[:]}] Started' )
    #print('start')
    while original_names.shape[1] != 0 and it_counter < max_num_iterations:

        # Initialize the variables that stores all the APIs in the total sequence as pair.
        # The first binary element shows if:
        #  - they belong to the original sequence (0) or
        #  - they have been injected by the tool (1) 
        #  - they probably belong to the loader (2) 
        total_names     = np.array(list(map(lambda x: [2 if id2word[int(x)].encode() not in called_apis else 0, x], original_names[0])))
        total_behaviors = np.split(original_behaviors, original_behaviors.shape[1]//4, axis = 1)
        total_behaviors = np.array(list(map(lambda x: [0, np.array(x[0])], total_behaviors)), dtype = object)

        if original_names.shape[1] <= 1000:
            # Add padding to the sequences
            padding_behaviors  = torch.tensor([24, 14, 14, 14] * (1000-original_names.shape[1]), dtype = int)
            padding_behaviors  = torch.unsqueeze(padding_behaviors, 0)
            original_names     = torch.nn.functional.pad(input=original_names, pad=(0, (1000-original_names.shape[1])), mode='constant', value=0)
            original_behaviors = torch.cat((original_behaviors, padding_behaviors), dim = 1)
        else: 
            original_names = original_names[:, :1000]
            original_behaviors = original_behaviors[:, :1000*4]
        

        # Send data to device
        original_names     = original_names.to(device)
        original_behaviors = original_behaviors.to(device)

        # Embed using the model ebedding layers
        embedded_names = model.embedder1(original_names)
        embedded_behaviors = model.embedder2(original_behaviors)


        # Compute the Jacobian sign
        position, sign_jacobian_names, sign_jacobian_behaviors = compute_jacobian(
            model, embedded_names, embedded_behaviors, 
            total_names, sample_name, position_ub)
    
        # Repeat M times the Jacobian sign to allow fast difference computation
        # M = number of apis to try
        if mode == 'names':
            sign_jacobian_names_repeated     = np.repeat(sign_jacobian_names, len(api_set), axis = 0)
        elif mode == 'behaviors':
            sign_jacobian_behaviors_repeated = np.repeat(sign_jacobian_behaviors, len(api_set), axis = 0)
        else:
            raise('Wrong mode')
        
        # Send back to CPU
        embedded_names = embedded_names.to('cpu').detach().numpy()
        embedded_behaviors = embedded_behaviors.to('cpu').detach().numpy()

        while model.single_inference(embedded_names, embedded_behaviors) == 1 and it_counter < max_num_iterations:     
            # Update iteractions counter
            it_counter += 1
                
            # Generate a new API name and behavior sequence for each API call we want to try to inject
            new_embedded_names     = new_embedded_names_given_index(embedded_names, position, api_set_name_id)
            new_embedded_behaviors = new_embedded_behaviors_given_index(embedded_behaviors, position, api_set_behavior_id)

            # Repeat M times the original sequences to allow fast difference computation
            # M = number of apis to try
            embedded_names_repeated     = np.repeat(embedded_names, len(api_set), axis = 0)
            embedded_behaviors_repeated = np.repeat(embedded_behaviors, len(api_set), axis = 0)

            #print(f'Before sub (names) {embedded_names_repeated.shape} - {new_embedded_names.shape}')
            #print(f'Before sub (behaviors) {embedded_behaviors_repeated.shape} - {new_embedded_behaviors.shape}')

            if mode == 'names':
                # Compute the sign of the difference between the previous embedded names sequence and each of the new 
                # generated embedded names sequence
                sub_embedded_names      = np.subtract(embedded_names_repeated, new_embedded_names)
                sign_sub_embedded_names = np.sign(sub_embedded_names)
                
                # Compute the difference between the sign of the difference and the sign on the Jacobian
                sub_signs_names     = np.subtract(sign_sub_embedded_names , sign_jacobian_names_repeated)
                
                # Compute the norm of the signs difference
                norm_sum     = np.linalg.norm(sub_signs_names, ord='fro', axis = (1,2))
            elif mode == 'behaviors':
                # Compute the sign of the difference between the previous embedded behaviors sequence and each of the new 
                # generated embedded behaviors sequence
                sub_embedded_behaviors      = np.subtract(embedded_behaviors_repeated, new_embedded_behaviors)
                sign_sub_embedded_behaviors = np.sign(sub_embedded_behaviors)
               
                # Compute the difference between the sign of the difference and the sign on the Jacobian
                sub_signs_behaviors = np.subtract(sign_sub_embedded_behaviors , sign_jacobian_behaviors_repeated)
                
                # Compute the norm of the signs difference
                norm_sum = np.linalg.norm(sub_signs_behaviors, ord='fro' , axis = (1,2))
            else:
                raise('Wrong mode')

            # Compute the argmin to decide the API call to inject
            selected_idx = np.argmin(norm_sum)

            # Set the current sequence equal to the selected ones
            embedded_names = new_embedded_names[selected_idx]
            embedded_names = np.expand_dims(embedded_names, axis = 0)
            embedded_behaviors = new_embedded_behaviors[selected_idx]
            embedded_behaviors = np.expand_dims(embedded_behaviors, axis = 0)

            # Update the total sequences
            total_names = np.concatenate((
                    total_names[:position],
                    np.array([(1, api_set_name_id[selected_idx])]),
                    total_names[position:]))
            total_behaviors = np.concatenate((
                    total_behaviors[:position],
                    np.array([(1, api_set_behavior_id[selected_idx])], dtype=object),
                    total_behaviors[position:]))
            
            #n = torch.tensor(list(map(lambda x: x[1], total_names[:1000]))).unsqueeze(0)
            #b = torch.tensor(list(map(lambda x: x[1], total_behaviors[:1000]))).flatten().unsqueeze(0)

            #print(f'single inference: {model.single_inference(embedded_names, embedded_behaviors)}')
            #print(f'single inference: {model.single_inference(model.embedder1(n).detach().numpy(), model.embedder2(b).detach().numpy())}')
            #print(f'single score: {model.single_score(embedded_names, embedded_behaviors)}')
            #print(f'pred given split {model.pred_given_split(n, b)}')
            #print(f'score given split {model.score_given_split(n, b)}')


            # Compute the Jacobian sign
            position, sign_jacobian_names, sign_jacobian_behaviors = compute_jacobian(
                    model,
                    torch.tensor(embedded_names).to(device), 
                    torch.tensor(embedded_behaviors).to(device),
                    total_names,
                    sample_name,
                    position_ub)
     
            
            # Repeat M times the Jacobian sign to allow fast difference computation
            # M = number of apis to try
            sign_jacobian_names_repeated = np.repeat(sign_jacobian_names, len(api_set), axis = 0)
            sign_jacobian_behaviors_repeated = np.repeat(sign_jacobian_behaviors, len(api_set), axis = 0)
        
            # Dynamically adjust the random upperbound
            position_ub = position_ub + 1 if position_ub < 999 else 999

            
        # Update the number of injected APIs counter
        if apis_counter == max_num_iterations:
            break

        apis_counter += len(list(filter(lambda x: x[0] == 1, total_names[:1000])))
        
        output_name_sequence.append(total_names[:1000])
        total_names = total_names[1000:]
        total_behaviors = total_behaviors[1000:]

        # Ignore the added APIs in the new window and the padding
        total_names     = np.array(list(filter(lambda x: (x[0] == 0 or x[0] == 2) and int(x[1]) != 0, total_names)))
        total_behaviors = np.array(list(filter(lambda x: (x[0] == 0 or x[0] == 2) and (np.array([24, 14, 14, 14]) != x[1]).any(), total_behaviors)))
        
        # Resize properly and set variables for next iteration
        original_names = torch.tensor(list(map(lambda x: x[1], total_names)))
        original_names = torch.unsqueeze(original_names, 0)
        
        original_behaviors = torch.tensor(np.array(list(map(lambda x: x[1], total_behaviors))))
        original_behaviors = torch.flatten(original_behaviors)
        original_behaviors = torch.unsqueeze(original_behaviors, 0)
        
        # Update the random upperbound
        position_ub = original_names.shape[1] - 1
        
    if it_counter < max_num_iterations:
        #sys.stdout.write('\r' + f'Injected {apis_counter} new API calls' + ' ' * 100)
        #sys.stdout.flush()
        pass
    else:
        it_counter = 999999
        #sys.stdout.write('\r' + f'Fuck, too many APIs' + ' ' * 100)
        #sys.stdout.flush()
        #api_counter = 999999
                             
    
    return apis_counter, output_name_sequence



def random_adversarial_sequence_generation(api_sequence, model, api_set, mode, called_apis, sample_name, max_num_iterations=1500):
    """
    Generate a random adversarial sequence of API calls: random position insertion, random API call
    """
    
    logger = logging.getLogger('adversarial_attack')
    global it_counter

    model.eval()

    # Number of injected APIs counter
    apis_counter = 0

    # Number of iteractions counter
    it_counter = 0

    # Final sequence of all API names with the right order
    output_name_sequence = []

    # Split the original sequence in names and behaviors
    original_names, original_behaviors = api_sequence.split([1000, 4000], 1)

    # From api name to id
    api_set_name_id = np.array(list(map(lambda x: word2id[x], api_set)))

    # From api name to behavior
    api_set_behavior = np.array(list(map(lambda x: word2behavior[x], api_set)))

    # From behavior to behavior id
    api_set_behavior_id = np.array(list(map(tuple_to_id, api_set_behavior)))


    # Initilize the position upperbound
    position_ub = len(list(filter(lambda x: x!=0, original_names[0])))
    # print(f'Initial value for the UB: {position_ub}')
    
    # Original names is set each time to a window of 1000 APIs: 
    # when the window.shape[1] (num of API) is 0 it means that 
    # there are no meaningful (!= PAD) APIs in that window
    
    logger.debug(f'[{sample_name[:]}] random_adversarial_sequence_generation Started' )

    while original_names.shape[1] != 0 and it_counter < max_num_iterations:
        # Add padding to the sequences
        padding_behaviors  = torch.tensor([24, 14, 14, 14] * (1000-original_names.shape[1]), dtype = int)
        padding_behaviors  = torch.unsqueeze(padding_behaviors, 0)
        original_names     = torch.nn.functional.pad(input=original_names, pad=(0, (1000-original_names.shape[1])), mode='constant', value=0)
        original_behaviors = torch.cat((original_behaviors, padding_behaviors), dim = 1)
        
        # Initialize the variables that stores all the APIs in the total sequence as pair.
        # The first binary element shows if:
        #  - they belong to the original sequence (0) or
        #  - they have been injected by the tool (1) 
        #  - they probably belong to the loader (2) 
        total_names     = np.array(list(map(lambda x: [2 if id2word[int(x)].encode() not in called_apis else 0, x], original_names[0])))
        total_behaviors = np.split(original_behaviors, 1000, axis = 1)
        total_behaviors = np.array(list(map(lambda x: [0, np.array(x[0])], total_behaviors)), dtype = object)

        # Send data to device
        if it_counter == 0:
            time_start = time.time()
        original_names     = original_names.to(device)
        original_behaviors = original_behaviors.to(device)

        # Embed using the model ebedding layers
        embedded_names = model.embedder1(original_names)
        embedded_behaviors = model.embedder2(original_behaviors)
        
        # Send back to CPU
        embedded_names = embedded_names.to('cpu').detach().numpy()
        embedded_behaviors = embedded_behaviors.to('cpu').detach().numpy()

        if it_counter == 0:
            time_end = time.time()
            # logger.info(f'[{sample_name[:]}] random_adversarial_sequence_generation Embedding time: {time_end - time_start} seconds' )

        while model.single_inference(embedded_names, embedded_behaviors) == 1 and it_counter < max_num_iterations:     
            # Update iteractions counter
            it_counter += 1
        
            # Select a random position in the sequence
            position = np.random.randint(0, position_ub)
            

            # Generate a new API name and behavior sequence for a RANDOM API call we want to try to inject
            random_api_index = np.random.randint(0, len(api_set))
            new_embedded_names     = new_embedded_names_given_index_and_api(embedded_names, position, api_set_name_id[random_api_index])
            new_embedded_behaviors = new_embedded_behaviors_given_index_and_api(embedded_behaviors, position, api_set_behavior_id[random_api_index])

            # Set the current sequence equal to the selected ones
            embedded_names = new_embedded_names[0]
            embedded_names = np.expand_dims(embedded_names, axis = 0)
            embedded_behaviors = new_embedded_behaviors[0]
            embedded_behaviors = np.expand_dims(embedded_behaviors, axis = 0)
            # Dynamically adjust the random upperbound
            position_ub = position_ub + 1 if position_ub < 999 else 999

            # Update the total sequences
            total_names = np.concatenate((
                    total_names[:position],
                    np.array([(1, api_set_name_id[random_api_index])]),
                    total_names[position:]))
            total_behaviors = np.concatenate((
                    total_behaviors[:position],
                    np.array([(1, api_set_behavior_id[random_api_index])], dtype=object),
                    total_behaviors[position:]))

        # Update the number of injected APIs counter
        apis_counter += len(list(filter(lambda x: x[0] == 1, total_names[:1000])))
        
        output_name_sequence.append(total_names[:1000])
        total_names = total_names[1000:]
        total_behaviors = total_behaviors[1000:]

        # Ignore the added APIs in the new window and the padding
        total_names     = np.array(list(filter(lambda x: (x[0] == 0 or x[0] == 2) and int(x[1]) != 0, total_names)))
        total_behaviors = np.array(list(filter(lambda x: (x[0] == 0 or x[0] == 2) and (np.array([24, 14, 14, 14]) != x[1]).any(), total_behaviors)))
        
        # Resize properly and set variables for next iteration
        original_names = torch.tensor(list(map(lambda x: x[1], total_names)))
        original_names = torch.unsqueeze(original_names, 0)
        
        original_behaviors = torch.tensor(np.array(list(map(lambda x: x[1], total_behaviors))))
        original_behaviors = torch.flatten(original_behaviors)
        original_behaviors = torch.unsqueeze(original_behaviors, 0)
        
        # Update the random upperbound
        position_ub = original_names.shape[1] - 1
                                     
    return apis_counter, output_name_sequence


def minimize_injected_apis(id_names_sequence, flags_sequence):
    """
    Function to minimize the number of injected APIs generated by 
    adversarial_sequence_generation
    """
    
    removed_counter = 0 # how many APIs have been removed
    removed = True # flag to stop the minimization
    last_removed_index = 0 # index of the API removed at the last iteration
    
    while removed:
        removed = False
        
        # Indexes where the APIs have been injected taking into account the
        # already removed indexes in the previous steps
        inserted_indexes = [i for i, flag in enumerate(flags_sequence) 
                            if flag == 1 and i >= last_removed_index]
        
        for inserted_index in inserted_indexes:
            
            # Delete the API at the current inserted_index
            new_id_names_sequence = np.delete(id_names_sequence, inserted_index, 0)
            
            # Number of sequences of len=1000 to consider
            n_sequences = math.ceil(new_id_names_sequence.shape[0] / 1000)
            
            # Pad with __PAD__ API (id=0)
            padding_width = n_sequences * 1000 - new_id_names_sequence.shape[0]
            new_id_names_sequence = np.pad(new_id_names_sequence, (0, padding_width), 'constant', constant_values=0)

            # Create the bacthes
            new_id_names_batches = np.array_split(new_id_names_sequence, n_sequences, axis=0)

            total_pred = 0
            for new_id_names_batch in new_id_names_batches:
                
                # From id names sequence to behaviors sequence
                new_id_behaviors_batch = behaviors_from_id_name(new_id_names_batch)
                new_id_behaviors_batch = new_id_behaviors_batch.flatten()
                
                # Transform in tensors, resize properly and send to the device
                new_id_names_batch = torch.from_numpy(new_id_names_batch)
                new_id_names_batch = torch.unsqueeze(new_id_names_batch, dim = 0).to(device)
                new_id_behaviors_batch = torch.from_numpy(new_id_behaviors_batch)
                new_id_behaviors_batch = torch.unsqueeze(new_id_behaviors_batch, dim = 0).to(device)
                
                # Update the total prediction with the current batch prediction
                pred = model.pred_given_split(new_id_names_batch, new_id_behaviors_batch)
                total_pred += pred

            # If zero, we can remove the API without consequences and break to update the indexes
            if total_pred == 0:
                id_names_sequence = new_id_names_sequence
                flags_sequence = np.delete(flags_sequence, inserted_index, 0)
                last_removed_index = inserted_index
                removed = True
                removed_counter += 1
                #sys.stdout.write('\r' + f'Removed the injected API at position {inserted_index}'
                                #  + ' ' * 100)
                #sys.stdout.flush()
                break
    
    #sys.stdout.write('\r' + f'Removed {removed_counter} APIs' + ' ' * 100)
    #sys.stdout.flush()
    return id_names_sequence, flags_sequence, removed_counter


def Rosenberg_adversarial_sequence_generation(api_sequence, model, api_set, mode, max_num_iterations=1500):
    logger = logging.getLogger('adversarial_attack')
    global it_counter

    model.eval()

    # Number of injected APIs counter
    apis_counter = 0

    # Number of iteractions counter
    it_counter = 0

    # Final sequence of all API names with the right order
    output_name_sequence = []

    # Split the original sequence in names and behaviors
    original_names, original_behaviors = api_sequence.split([1000, 4000], 1)

    # From api name to id
    api_set_name_id = np.array(list(map(lambda x: word2id[x], api_set)))

    # From api name to behavior
    api_set_behavior = np.array(list(map(lambda x: word2behavior[x], api_set)))

    # From behavior to behavior id
    api_set_behavior_id = np.array(list(map(tuple_to_id, api_set_behavior)))


    # Initilize the position upperbound
    position_ub = len(list(filter(lambda x: x!=0, original_names[0])))
    # print(f'Initial value for the UB: {position_ub}')
    
    # Original names is set each time to a window of 1000 APIs: 
    # when the window.shape[1] (num of API) is 0 it means that 
    # there are no meaningful (!= PAD) APIs in that window
    


    while original_names.shape[1] != 0 and it_counter < max_num_iterations:
        # Add padding to the sequences
        padding_behaviors  = torch.tensor([24, 14, 14, 14] * (1000-original_names.shape[1]), dtype = int)
        padding_behaviors  = torch.unsqueeze(padding_behaviors, 0)
        # padding_behaviors  = padding_behaviors.to(device)
        original_names     = torch.nn.functional.pad(input=original_names, pad=(0, (1000-original_names.shape[1])), mode='constant', value=0)
        original_behaviors = torch.cat((original_behaviors, padding_behaviors), dim = 1)
        
        # Initialize the variables that stores all the APIs in the total sequence as pair.
        # The first binary element shows if:
        #  - they belong to the original sequence (0) or
        #  - they have been injected by the tool (1) 
        total_names     = np.array(list(map(lambda x: [0, x], original_names[0])))
        total_behaviors = np.split(original_behaviors, 1000, axis = 1)
        # total_behaviors = total_behaviors.to('cpu')
        # total_behaviors = torch.tensor(total_behaviors)
        total_behaviors = np.array(list(map(lambda x: [0, np.array(x[0])], total_behaviors)), dtype = object)

        # Send data to device
        original_names     = original_names.to(device)
        original_behaviors = original_behaviors.to(device)

        # Embed using the model ebedding layers
        embedded_names = model.embedder1(original_names)
        embedded_behaviors = model.embedder2(original_behaviors)


        # Compute the Jacobian sign
        
        sign_jacobian_names, sign_jacobian_behaviors = compute_jacobian_Rosenberg(
            model, embedded_names, embedded_behaviors)
        
        # Repeat M times the Jacobian sign to allow fast difference computation
        # M = number of apis to try
        if mode == 'names':
            sign_jacobian_names_repeated     = np.repeat(sign_jacobian_names, len(api_set), axis = 0)
        elif mode == 'behaviors':
            sign_jacobian_behaviors_repeated = np.repeat(sign_jacobian_behaviors, len(api_set), axis = 0)
        else:
            raise('Wrong mode')
        
        # Select a random position to inject the API call
        position = np.random.randint(0, position_ub)
            

        # Send back to CPU
        embedded_names = embedded_names.to('cpu').detach().numpy()
        embedded_behaviors = embedded_behaviors.to('cpu').detach().numpy()

        while model.single_inference(embedded_names, embedded_behaviors) == 1 and it_counter < max_num_iterations:     
            # Update iteractions counter
            it_counter += 1
                
            # Generate a new API name and behavior sequence for each API call we want to try to inject
            new_embedded_names     = new_embedded_names_given_index(embedded_names, position, api_set_name_id)
            new_embedded_behaviors = new_embedded_behaviors_given_index(embedded_behaviors, position, api_set_behavior_id)

            # Repeat M times the original sequences to allow fast difference computation
            # M = number of apis to try
            embedded_names_repeated     = np.repeat(embedded_names, len(api_set), axis = 0)
            embedded_behaviors_repeated = np.repeat(embedded_behaviors, len(api_set), axis = 0)

            #print(f'Before sub (names) {embedded_names_repeated.shape} - {new_embedded_names.shape}')
            #print(f'Before sub (behaviors) {embedded_behaviors_repeated.shape} - {new_embedded_behaviors.shape}')

            if mode == 'names':
                # Compute the sign of the difference between the previous embedded names sequence and each of the new 
                # generated embedded names sequence
                sub_embedded_names      = np.subtract(embedded_names_repeated, new_embedded_names)
                sign_sub_embedded_names = np.sign(sub_embedded_names)
                
                # Compute the difference between the sign of the difference and the sign on the Jacobian
                sub_signs_names     = np.subtract(sign_sub_embedded_names , sign_jacobian_names_repeated)
                
                # Compute the norm of the signs difference
                norm_sum     = np.linalg.norm(sub_signs_names, ord='fro', axis = (1,2))
            elif mode == 'behaviors':
                # Compute the sign of the difference between the previous embedded behaviors sequence and each of the new 
                # generated embedded behaviors sequence
                sub_embedded_behaviors      = np.subtract(embedded_behaviors_repeated, new_embedded_behaviors)
                sign_sub_embedded_behaviors = np.sign(sub_embedded_behaviors)
               
                # Compute the difference between the sign of the difference and the sign on the Jacobian
                sub_signs_behaviors = np.subtract(sign_sub_embedded_behaviors , sign_jacobian_behaviors_repeated)
                
                # Compute the norm of the signs difference
                norm_sum = np.linalg.norm(sub_signs_behaviors, ord='fro' , axis = (1,2))
            else:
                raise('Wrong mode')

            # Compute the argmin to decide the API call to inject
            selected_idx = np.argmin(norm_sum)

            # Set the current sequence equal to the selected ones
            embedded_names = new_embedded_names[selected_idx]
            embedded_names = np.expand_dims(embedded_names, axis = 0)
            embedded_behaviors = new_embedded_behaviors[selected_idx]
            embedded_behaviors = np.expand_dims(embedded_behaviors, axis = 0)

            # Compute the Jacobian sign
            sign_jacobian_names, sign_jacobian_behaviors = compute_jacobian_Rosenberg(
                    model,
                    torch.tensor(embedded_names).to(device), 
                    torch.tensor(embedded_behaviors).to(device),
                    )
            
            # Select a random position to inject the API call
            position = np.random.randint(0, position_ub)
            
            # Repeat M times the Jacobian sign to allow fast difference computation
            # M = number of apis to try
            sign_jacobian_names_repeated = np.repeat(sign_jacobian_names, len(api_set), axis = 0)
            sign_jacobian_behaviors_repeated = np.repeat(sign_jacobian_behaviors, len(api_set), axis = 0)
        
            # Dynamically adjust the random upperbound
            position_ub = position_ub + 1 if position_ub < 999 else 999

            # Update the total sequences
            total_names = np.concatenate((
                    total_names[:position],
                    np.array([(1, api_set_name_id[selected_idx])]),
                    total_names[position:]))
            total_behaviors = np.concatenate((
                    total_behaviors[:position],
                    np.array([(1, api_set_behavior_id[selected_idx])], dtype=object),
                    total_behaviors[position:]))

        # Update the number of injected APIs counter
        if apis_counter == max_num_iterations:
            break
        apis_counter += len(list(filter(lambda x: x[0] == 1, total_names[:1000])))
        
        output_name_sequence.append(total_names[:1000])
        total_names = total_names[1000:]
        total_behaviors = total_behaviors[1000:]

        # Ignore the added APIs in the new window and the padding
        total_names     = np.array(list(filter(lambda x: (x[0] == 0 or x[0] == 2) and int(x[1]) != 0, total_names)))
        total_behaviors = np.array(list(filter(lambda x: (x[0] == 0 or x[0] == 2) and (np.array([24, 14, 14, 14]) != x[1]).any(), total_behaviors)))
        
        # Resize properly and set variables for next iteration
        original_names = torch.tensor(list(map(lambda x: x[1], total_names)))
        original_names = torch.unsqueeze(original_names, 0)
        
        original_behaviors = torch.tensor(np.array(list(map(lambda x: x[1], total_behaviors))))
        original_behaviors = torch.flatten(original_behaviors)
        original_behaviors = torch.unsqueeze(original_behaviors, 0)
        
        # Update the random upperbound
        position_ub = original_names.shape[1]
        
    if it_counter < max_num_iterations:
        #print(it_counter)
        #sys.stdout.write('\r' + f'Injected {apis_counter} new API calls' + ' ' * 100)
        #sys.stdout.flush()
        pass
    else:
        it_counter = 999999
        #sys.stdout.write('\r' + f'Fuck, too many APIs' + ' ' * 100)
        #sys.stdout.flush()
        #api_counter = 9999999
                             
    
    return apis_counter, output_name_sequence


def helper_func(list_api):
    apis_we_can_inject = [api_name.decode() for api_name in chain_api_args.keys()]
    return list(filter(lambda x: x in apis_we_can_inject, list_api))


def convert_to_string(list_b_api_names):
    return list(map(lambda x: x.decode(), list_b_api_names))


# Function to set the needed paths
def set_paths():
    global DIR
    global model_path

    DIR = general_paths['dir']
    model_path = os.path.join(DIR, general_paths['model_name']) # name of the model to load


def count_valid_apis(name):
        return len( [x for x in name if x != 0])


def test_set_attack(original_sequence, strategy):
    """ Function to generate the adversarial sequence, given a sequence and a strategy
    Acts as a wrapper for the adversarial sequence different strategies, 
    loads only name and behavior sequences,
    and does not discard sample for imported calls or minimium number of APIs.

    Uses the logger to log the results that are then parsed to generate the plots.
    
    Args:
        original_sequence (list): list containing the original sequence
        strategy (str): strategy to use to generate the adversarial sequence
                            
    Returns:
        adversarial_sequence (list): list containing the adversarial sequence
    """
    global device
    global model
    # Set the paths
    set_paths()
    
    # Get the logger
    logger = logging.getLogger('adversarial_attack')

    # Extract the info from the data
    original_names = original_sequence[0]
    original_behaviors = original_sequence[1]


    # Load the model
    # model = torch.load(model_path)
    model = torch.load(model_path)

    # Cast the model to the new defined class
    model.__class__ = CustomNet

    # Restore the proper dimensionality
    original_names = np.expand_dims(original_names, axis=0)
    original_behaviors = np.expand_dims(original_behaviors, axis=0)

    # Check if the sample is already classified as not malicious
    # input(f'Press enter to continue {device}')
    prediction = model.pred_given_split(torch.tensor(original_names).to(device), torch.tensor(original_behaviors).to(device))
    if prediction == 0:        
        logger.warning(f'Already classified as not malicious')
        return (0, 'unknown')
    

    # Concat names and behaviors to have am array of size (1, 5000)
    test_x = np.concatenate([original_names, original_behaviors], 1)
    test_x = torch.tensor(test_x)

    # Start the attack
    if strategy == 'test_set_rosenberg_strategy' or strategy == 'malware_00375_rosenberg_strategy' :
        injectable_apis = list(word2id.keys())
        injectable_apis.pop(28) # PAD
        injectable_apis.pop(30) # __exception__
        injectable_apis = [api for api in injectable_apis if api != '__anomaly__']
        apis_counter, name_sequence = Rosenberg_adversarial_sequence_generation(test_x, model, injectable_apis, 'names', general_constant['max_it_number'])
    
    elif (strategy == 'test_set_our_strategy' or strategy == 'malware_00375_our_strategy' 
          or strategy == 'test_set_random_strategy' or strategy == 'malware_00375_random_strategy'):
        
        injectable_apis = [api for api in word2id.keys() if api != '__anomaly__' and api != '_PAD_' and api != '__exception__']
        callable_apis   = [api.encode() for api in injectable_apis]

        if strategy == 'test_set_random_strategy' or strategy == 'malware_00375_random_strategy':
            apis_counter, name_sequence = random_adversarial_sequence_generation(test_x, model, injectable_apis, 'names', callable_apis, 'unknown', general_constant['max_it_number'])
        else:
            apis_counter, name_sequence = adversarial_sequence_generation(test_x, model, injectable_apis, 'names', callable_apis, 'unknown', general_constant['max_it_number'])
    # Check if the attack has been successfully completed
    if apis_counter >= general_constant['max_it_number']:
        logger.error(f"[] Failed - {general_constant['max_it_number']} iterations reached" )
        return (-2, 'unknown')
    
    logger.info(f'[] OK - {apis_counter} added APIs' )
    
    if 'test_set' not in strategy and 'random' not in strategy:
        write_result(name_sequence, 'unknown')
    
    return (apis_counter, 'unknown')

def attack(original_sequence):
    global device
    global model
    # Set the paths
    set_paths()
    
    # Get the logger
    logger = logging.getLogger('adversarial_attack')

    # Extract the info from the data
    original_names = original_sequence[0]
    original_behaviors = original_sequence[1]
    original_import = original_sequence[2]
    original_sample_name = original_sequence[3]
    original_called = original_sequence[4]
    
    if len(original_sequence) == 6:
        attack_strategy = original_sequence[5]
    else:
        attack_strategy = 'attack'
    # logger.info(f'Strategy: {attack_strategy}')
    
    # Check if there are hijackable positions
    original_names_set = set(map(lambda x: id2word[int(x)].encode(), original_names))
    if original_names_set.intersection(set(original_called)) == set():
        
        #original

        if len(original_names_set) > len(original_called):
            path = os.path.join(chain_general_paths['dir'], chain_general_paths['dir'], chain_general_paths['input_subdir'], original_sample_name)
            
            try:
                entropy = compute_entropy(path)
            except:
                entropy = None
            
            logger.warning(f'[{original_sample_name[:]}] Discarded - there are less statically detected calls than recorded behavior (packed?), check entropy: {entropy}')
        
        elif original_names_set == set([b'_PAD_']):
            logger.warning(f'[{original_sample_name[:]}] Discarded - Cuckoo did not record any behavior')
        else:
            logger.warning(f'[{original_sample_name[:]}] Discarded - No hijackable positions, recorded calls set len: {len(original_names_set)} - static calls detected set len: {len(original_called)}')
            
        return (-4, original_sample_name)

    # Filter the data that contais too few APIs
    num_valid_apis = count_valid_apis(original_names)
    if  num_valid_apis < general_constant['min_valid_apis']:
        logger.warning(f"[{original_sample_name[:]}] Discarded - < {general_constant['min_valid_apis']} valid apis" )
        return (-3, original_sample_name)
    
    if attack_strategy != 'combo_attack':
        # Fix for when called by Colab
        setattr(__main__, "Net", Net)

        # Load the model
        if torch.cuda.is_available():
            model = torch.load(model_path)
        else:
            model = torch.load(model_path, map_location=torch.device('cpu'))

        # Cast the model to the new defined class
        model.__class__ = CustomNet

    # Restore the proper dimensionality
    original_names = np.expand_dims(original_names, axis=0)
    original_behaviors = np.expand_dims(original_behaviors, axis=0)

    # Decode and filter the list of imported APIs
    original_import_decoded  = convert_to_string(original_import)
    original_import_filtered = helper_func(original_import_decoded)

    # Check if the sample is already classified as not malicious
    if not attack_strategy == 'combo_attack':
        prediction = model.pred_given_split(torch.tensor(original_names).to(device), torch.tensor(original_behaviors).to(device))
        if prediction == 0:        
            logger.warning(f'[{original_sample_name[:]}] Already classified as not malicious')
            return (0, original_sample_name)
    
    # Check if the sample imports a suffienct number of injectable APIs
    min_number_injectable_APIs = round(general_constant['threshold_imported_api']*len(chain_api_args.keys()))
    num_imported_apis = len(original_import_filtered)
    if num_imported_apis < min_number_injectable_APIs:
            logger.warning(f'[{original_sample_name[:]}] Discarded - {num_imported_apis} < {min_number_injectable_APIs} imported apis' )
            return (-1, original_sample_name)

    # Concat names and behaviors to have am array of size (1, 5000)
    if not attack_strategy == 'combo_attack':
        test_x = np.concatenate([original_names, original_behaviors], 1)
        test_x = torch.tensor(test_x)
    else:
        test_x = [torch.tensor(original_names), torch.tensor(original_behaviors)]


    # Start the attack
    if attack_strategy == 'attack':
        try:
            apis_counter, name_sequence = adversarial_sequence_generation(test_x, model, original_import_filtered, 'names', original_called, original_sample_name, general_constant['max_it_number'])
        except CannotFindBestPositionException as e:
            logger.critical(f'[{original_sample_name[:]}] Failed - {e} - {num_imported_apis} imported APIs' )
            return (-5, original_sample_name)
        # Check if the attack has been successfully completed
        if apis_counter >= general_constant['max_apis_to_inject']:
            logger.error(f"[{original_sample_name[:]}] Failed - {general_constant['max_apis_to_inject']} iterations reached - {num_imported_apis} imported APIs" )
            return (-2, original_sample_name)
        
        logger.info(f'[{original_sample_name[:]}] OK - {apis_counter} added APIs - {num_valid_apis} valid APIs - {num_imported_apis} imported APIs' )
    
    elif attack_strategy == 'combo_attack':
        try:
            apis_counter, name_sequence = combo_adversarial_sequence_generation(test_x, model, original_import_filtered, 'names', original_called, original_sample_name, general_constant['max_it_number'])
        except CannotFindBestPositionException as e:
            logger.critical(f'[{original_sample_name[:]}] Failed - {e} - {num_imported_apis} imported APIs' )
            return (-5, original_sample_name)
        # Check if the attack has been successfully completed
        if apis_counter >= general_constant['max_apis_to_inject']:
            logger.error(f"[{original_sample_name[:]}] Failed - {general_constant['max_apis_to_inject']} iterations reached - {num_imported_apis} imported APIs" )
            return (-2, original_sample_name)
        
        logger.info(f'[{original_sample_name[:]}] OK - {apis_counter} added APIs - {num_valid_apis} valid APIs - {num_imported_apis} imported APIs' )
    
    elif attack_strategy == 'random_attack':
        try:
            apis_counter, name_sequence = random_adversarial_sequence_generation(test_x, model, original_import_filtered, 'names', original_called, original_sample_name, general_constant['max_it_number'])
        except CannotFindBestPositionException as e:
            logger.critical(f'[{original_sample_name[:]}] Failed - {e}' )
            return (-5, original_sample_name)
        # Check if the attack has been successfully completed
        if apis_counter >= general_constant['max_it_number']:
            logger.error(f"[{original_sample_name[:]}] Failed - {general_constant['max_it_number']} iterations reached" )
            return (-2, original_sample_name)
        
        logger.info(f'[{original_sample_name[:]}] OK - {apis_counter} added APIs' )
    else:
        raise Exception(f'Unknown attack strategy: {attack_strategy}')
    # Write the result

    # sys.stdout.write('before write result')
    # sys.stdout.flush()
    if attack_strategy == 'attack':
        write_result(name_sequence, original_sample_name)
    elif attack_strategy == 'combo_attack':
        return (apis_counter, name_sequence)
    # sys.stdout.write('after write result')
    # sys.stdout.flush()

    return (apis_counter, original_sample_name)


def stats_on_imported(original_sequence):
    original_sample_name = original_sequence[3]
    original_import = original_sequence[2]

    # Decode and filter the list of imported APIs
    original_import_decoded  = convert_to_string(original_import)
    original_import_filtered = helper_func(original_import_decoded)

    return original_sample_name, len(original_import_filtered)



def find_max_score(original_sequences):
    '''
    Function to evaluate different executions of the same sample and return the
    one with the highest confidence score of the model

    Args:
        original_sequences (list): list containing the original sequences of different
        executions of the same sample                        
    Returns:
       index of the sequence with the highest confidence score
    '''
    scores = []
    for original_sequence in original_sequences:
        original_names = original_sequence[0]
        original_behaviors = original_sequence[1]

        # Restore the proper dimensionality
        original_names = np.expand_dims(original_names, axis=0)
        original_behaviors = np.expand_dims(original_behaviors, axis=0)
        
        # Check if the sequence is greater than 1000
        if original_names.shape[1] > 1000:
            # Split the sequence in windows of 1000 APIs
            original_names = np.split(original_names, np.arange(1000, original_names.shape[1], 1000), axis = 1)
            original_behaviors = np.split(original_behaviors, np.arange(4000, original_behaviors.shape[1], 4000), axis = 1)
            
            # Consider the maximum score among the windows
            max_score = 0
            for names, behaviors in zip(original_names, original_behaviors):
                names = torch.tensor(names)
                behaviors = torch.tensor(behaviors)

                # Add padding to the sequences
                padding_behaviors  = torch.tensor([24, 14, 14, 14] * (1000-names.shape[1]), dtype = int)
                padding_behaviors  = torch.unsqueeze(padding_behaviors, 0)
                names     = torch.nn.functional.pad(input=names, pad=(0, (1000-names.shape[1])), mode='constant', value=0)
                behaviors = torch.cat((behaviors, padding_behaviors), dim = 1)

                
                score = model.score_given_split(names.to(device), behaviors.to(device))
                #print(f'score {score}')
                if score > max_score:
                    max_score = score
            scores.append(max_score)

        else:
            scores.append(model.score_given_split(torch.tensor(original_names).to(device), torch.tensor(original_behaviors).to(device)))
    #print(f'scores {scores}')
    if all(score <= 0.5 for score in scores):
        return -1

    return scores.index(max(scores))


def update_sequence_with_injected_apis(original_sequence, called_apis, api_dict):
    '''
    Function to update the original sequence of API calls with the injections
    returned by the adversarial attack

    Args:
        original_sequence (list): list containing the original sequence of API calls                            
    Returns:
        adversarial_sequence_names (list): list containing the adversarial sequence names
        adversarial_sequence_behaviors (list): list containing the adversarial sequence behaviors
        info_list (list): list containing the info about the injected APIs
    '''

    '''
    info_list structure:
    [None, None, ..., None, (hijacked_api, occurence, offset), None, ..., None]
    hijacked_api: the API that has been hijacked
    occurence: the occurence of the hijacked API in the original sequence
    offset: offset of the current injected API in the sequence of APIs injected for 
            the current hijacked API
    0 if loader
    '''

    adversarial_sequence_names = [] # list to store the adversarial sequence names
    inspected_api_dict = {} # dict to store the inspected APIs
    info_list = [] # list to store the info about the injected APIs


    for api in original_sequence:
        if id2word[int(api)].encode() not in called_apis:
            info_list.append(0)
            adversarial_sequence_names.append(api)
            continue

        # Update the inspected API dict
        if api not in inspected_api_dict.keys():
            inspected_api_dict[api] = 0
        else:
            inspected_api_dict[api] += 1
        
        # Add the injections to the adversarial sequence
        if api in api_dict.keys():
            occurence = inspected_api_dict[api]
            for injection_couple in api_dict[api]:
                if occurence == injection_couple[0]:
                    adversarial_sequence_names += list(map(lambda x: word2id[x], injection_couple[1]))
                    for i in range(len(injection_couple[1])):
                        info_list.append((api, occurence, i))

        # Add the original API to the adversarial sequence
        adversarial_sequence_names.append(api)
        info_list.append(None)

    return adversarial_sequence_names,behaviors_from_id_name(adversarial_sequence_names).flatten(), info_list


def update_combo_api_dict(old_api_dict, current_api_dict, current_sequence, current_info_list):
    '''
    Function to update the dict of API to inject with the injections of the last run 
    '''
    
    inspected_api_dict = {} # dict to store the inspected APIs refered to the original sequence
    fake_inspected_api_dict = {} # dict to store the inspected APIs refered to the new sequence
    modified_hijacking_dict = {} # dict to store the modified hijacking APIs


    for i, api in enumerate(current_sequence):

        if current_info_list[i] == 0:
            # If 0, the API is a loader
            continue

        if current_info_list[i] == None:
            # Update the inspected API dict (occurence counter)
            if api not in inspected_api_dict.keys():
                inspected_api_dict[api] = 0
            else:
                inspected_api_dict[api] += 1
        
        if api not in fake_inspected_api_dict.keys():
            fake_inspected_api_dict[api] = 0
        else:
            fake_inspected_api_dict[api] += 1

        if api in current_api_dict.keys():
            # The API must bi hijacked
            if current_info_list[i] == None:
                # If None, the API to hijack is not an injected API in a previous step       

                for injection_couple in current_api_dict[api]:
                    
                    # Check if the occurence is the right one
                    if injection_couple[0] != fake_inspected_api_dict[api]:
                        continue

                    if api not in old_api_dict.keys():
                        # If the API is not present in the old dict, create it
                        old_api_dict[api] = []

                    # If the occurence is already present, update the list of injections
                    list_copy = old_api_dict[api]
                    old_api_dict[api] = []  
                    modified = False
                    for cc in list_copy:
                        if cc[0] == inspected_api_dict[api]:
                            old_api_dict[api].append((cc[0], 
                                                    cc[1] + injection_couple[1]))
                            modified = True
                        else:
                            old_api_dict[api].append(cc)
                    if not modified:
                        # If the occurence is not present, add it
                        old_api_dict[api].append((inspected_api_dict[api], injection_couple[1]))
            else:
                hijacked_api = current_info_list[i][0]
                occurence = current_info_list[i][1]
                offset = current_info_list[i][2]

                # Update the modified hijacking dict and compute the new offset due to previous
                # modifications of the same hijacked API
                if str(hijacked_api)+str(occurence) not in modified_hijacking_dict.keys():
                    modified_hijacking_dict[str(hijacked_api)+str(occurence)] = len(current_api_dict[api])
                else:
                    offset += modified_hijacking_dict[str(hijacked_api)+str(occurence)]
                    modified_hijacking_dict[str(hijacked_api)+str(occurence)] += len(current_api_dict[api])

                for injection_couple in current_api_dict[api]:
                    
                    # Check if the occurence is the right one
                    if injection_couple[0] != fake_inspected_api_dict[api]:
                        continue

                    new_data = []
                    for cc in old_api_dict[hijacked_api]:
                        if cc[0] != occurence:    
                            new_data.append(cc)
                        else:
                            new_data.append((occurence, 
                                            cc[1][:offset] + 
                                            injection_couple[1] + 
                                            cc[1][offset:])) 
                    old_api_dict[hijacked_api] = new_data
    return old_api_dict


def combo_attack(original_sequences):
    global model

    # Get the logger
    logger = logging.getLogger('adversarial_attack')
    
    # Set the paths
    set_paths()

    # Fix for when called by Colab
    setattr(__main__, "Net", Net)

    # Load the model
    if torch.cuda.is_available():
        model = torch.load(model_path)
    else:
        model = torch.load(model_path, map_location=torch.device('cpu'))

    # Cast the model to the new defined class
    model.__class__ = CustomNet

    # Initilize the counter of the number of APIs injected
    final_apis_counter = 0

    # Initialize the disct of APIs to inject
    final_api_dict = {}

    # Initialize the list of info_list
    info_lists = []
    for original_sequence in original_sequences:
        total_names = list(map(lambda x: 0 if id2word[int(x)].encode() not in original_sequence[4] else None, original_sequence[0]))
        info_lists.append(total_names)
    
    # Initialize the list of new sequences
    new_sequences = original_sequences
    
    # Initialize the set of APIs to inject
    final_set_api_dict = set([])

    index = 0
    while index != -1 and final_apis_counter < general_constant['max_apis_to_inject']:

        # Get the execution with the highest confidence score
        index = find_max_score(new_sequences)


        # Attack the sample
        apis_counter, name_sequence = attack(new_sequences[index])

        if apis_counter < 0:
            return (apis_counter, new_sequences[0][3])
                
        # Update the number of injected APIs counter
        final_apis_counter += apis_counter

        # Produce the words and flags sequences produced by the attack strategy
        words_sequence, flags_sequence = convert_id_seq_to_word_seq(name_sequence)
        
        # Produce the dict of API to inject
        api_dict, set_api_dict = create_injection_dict(words_sequence, flags_sequence)
        final_set_api_dict = final_set_api_dict.union(set_api_dict)
        # Transform the keys of the dict from words to ids
        api_dict = {new_key: api_dict[old_key] for old_key, new_key in zip(api_dict.keys(), map(lambda x: word2id[x], api_dict.keys()))}

        # Update the dict of APIs to inject
        #print(f'New execution {api_dict}')
        final_api_dict = update_combo_api_dict(final_api_dict, api_dict, new_sequences[index][0], info_lists[index])
        #print(f'New combo {final_api_dict}')

        # Update the original sequences with the injected APIs
        for i, original_sequence in enumerate(original_sequences):
            adversarial_names, adversarial_behaviors, new_info_list = update_sequence_with_injected_apis(original_sequence[0], original_sequence[4], final_api_dict)
            new_sequences[i] = (adversarial_names, adversarial_behaviors, original_sequence[2], original_sequence[3], original_sequence[4], original_sequence[5])
            info_lists[i] = new_info_list
        
    final_api_dict = {new_key: final_api_dict[old_key] for old_key, new_key in zip(final_api_dict.keys(), map(lambda x: id2word[x], final_api_dict.keys()))}
    write_combo_result(final_api_dict, final_set_api_dict, original_sequences[0][3])
    return final_apis_counter, original_sequences[0][3]            