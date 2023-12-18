import numpy as np
import torch
from config.config_general import general_constant
from config.config_paths import general_paths
from utils.evaluate import evaluate
from utils.attack import attack, test_set_attack, combo_attack, stats_on_imported
from tqdm import tqdm
from multiprocessing import Pool
import torch.utils.data    as Data
from utils.exceptions import *
from utils.write_result import *
import os
import itertools
import __main__
from utils.model import Net


def test(test_x_name, test_x_behaviors):
    """
    Function to test the model performance
    """
    # For test purpose: we assume that only malwares are given
    test_y = np.ones(test_x_name.shape[0])
    test_x = np.concatenate([test_x_name, test_x_behaviors], 1)

    test_xt = torch.from_numpy(test_x)
    test_yt = torch.from_numpy(test_y.astype(np.float32))

    test_data = Data.TensorDataset(test_xt, test_yt)

    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=128,
        num_workers=1,
    )

    con, acc, precision, recall, f1, b_acc = test(test_loader)

    print(f'Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Balanced Accuracy: {b_acc:.4f}')


def load_data(mode):
    # Path of the features to load
    if (mode == 'attack' or mode == 'combo_attack' or mode == 'random_attack' or mode == 'malware_00375_our_strategy' 
        or mode == 'malware_00375_rosenberg_strategy' or mode == 'malware_00375_random_strategy' or mode == 'stats'):
        features_path = os.path.join(general_paths['features_dir'], general_paths['original_features_filename'])
    elif mode == 'evaluation':
        features_path = os.path.join(general_paths['features_dir'], general_paths['patched_features_filename'])
    elif mode == 'test_set_our_strategy' or mode == 'test_set_rosenberg_strategy' or mode == 'test_set_random_strategy':
        tesi_cwd = os.environ['TESI_CWD'] # Works with export command
        features_path = os.path.join(tesi_cwd, 'Malware-Detection-API-Sequence-Intrinsic-Features/data_demo.npz')
        # features_path = os.path.join(os.environ['TESI_CWD'],'Malware-Detection-API-Sequence-Intrinsic-Features/data_demo.npz')
    else:
        raise WrongModeException(f'Wrong mode {mode}. Please choose between attack and evaluation')

    data = np.load(features_path, allow_pickle=True)
    
    # Extract the info from the data
    test_x_name         = data['x_name']
    test_x_behaviors    = data['x_semantic']
    
    if mode == 'attack' or mode == 'combo_attack' or mode == 'random_attack' or mode =='evaluation' or mode == 'stats':
        test_x_imported     = data['imported_apis']
        test_x_called       = data['called_apis']
        test_x_sample_names = data['sample_names']
    else:
        test_x_imported     = None
        test_x_called       = None
        test_x_sample_names = None
    return test_x_name, test_x_behaviors, test_x_imported, test_x_sample_names, test_x_called

def show_results(results):
    """
    Function to print the results
    """
    positive = list(filter(lambda x: x[1] == 1, results))
    negative = list(filter(lambda x: x[1] == 0, results))
    print('Positive: ', len(positive))
    print('Negative: ', len(negative))
    print(f'Accuracy: {len(positive)/len(results)*100} %')

def show_results_multi_submission(results):
    results_per_sample = {}
    for result in results:
        if result[0] in results_per_sample:
            if result[1] == 1:
                results_per_sample[result[0]]['positive'] += 1
            else:
                results_per_sample[result[0]]['negative'] += 1
        else:
            stats = {'positive': 0, 'negative': 0}
            if result[1] == 1:
                stats['positive'] += 1
            else:
                stats['negative'] += 1
            
            results_per_sample[result[0]] = stats
    

    # Compute the evasion score - the higher the better
    # evasion_score = (number of negative results - classified as NOT malware) / (number of positive results + number of negative results)
    evasion_score_per_sample = {}
    for sample in results_per_sample:
        evasion_score_per_sample[sample] = results_per_sample[sample]['negative'] / (results_per_sample[sample]['positive'] + results_per_sample[sample]['negative'])

    print('Consider multiple submission of each patched sample to achieve evasion.')
    print('The evasion score for each sample is computed as follows: ')
    print('(num of execution classified as goodware) / (number of positive results + number of negative results)')
    print('The higher the evasion score, the better the evasion.')

    print(f'The average evasion score is: {np.mean(list(evasion_score_per_sample.values()))}')
    num_samples_always_malware = 0
    for score in evasion_score_per_sample.values():
        if score == 0:
            num_samples_always_malware += 1
    print(f'There are {num_samples_always_malware} samples that were always classified as malware')
    print(f'There are {len(evasion_score_per_sample) - num_samples_always_malware} samples that were sometimes classified as goodware')
    
def dispatcher(mode = 'attack'):
    global start_time_string
    # Load the data
    test_x_name, test_x_behaviors, test_x_imported, test_x_sample_names, test_x_called = load_data(mode)
    if mode == 'attack' or mode == 'random_attack' or mode == 'evaluation' or mode == 'stats':
        original_sequence = zip(test_x_name, test_x_behaviors, test_x_imported, test_x_sample_names, test_x_called, itertools.repeat(mode))
    else:
        original_sequence = zip(test_x_name, test_x_behaviors)

    
    # Create pool of processes
    #general_constant['n_pools']
    if mode == 'attack':
        # with Pool(general_constant['n_pools']) as p:
        #     # Apply the function to the sequences
        #     results = list(tqdm(p.imap_unordered(attack, original_sequence), total=len(test_x_name)))
        #     write_attack_result('attack', results)

        results = []
        unique_sample_names = np.unique(test_x_sample_names)
        for unique_sample_name in tqdm(unique_sample_names, total=len(unique_sample_names)):
            # Get the indexes of the sample
            sample_indexes = np.where(test_x_sample_names == unique_sample_name)[0]

            # Select the index of the sample with the maximum number of called APIs
            maximum = 0
            maximum_index = 0
            for sample_index in sample_indexes:
                if len(test_x_name[sample_index]) > maximum:
                    maximum = len(test_x_called[sample_index])
                    maximum_index = sample_index

            # Get the sample
            names = test_x_name[maximum_index]
            # Get the behaviors
            behaviors = test_x_behaviors[maximum_index]
            # Get the imported apis
            imported_apis = test_x_imported[maximum_index]
            # Get the called apis
            called_apis = test_x_called[maximum_index]
            # Get the mode
            modes = 'attack'
            # Create the sequence
            original = (names, behaviors, imported_apis, unique_sample_name, called_apis, modes)
            # Apply the attack
            results.append(attack(original))
        write_attack_result('attack', results)

        #for e in tqdm(original_sequence, total=len(test_x_name)):
        #    results = attack(e)
        #    write_attack_result('attack', results)
        # attack( original_sequence)
    elif mode == 'evaluation':
        setattr(__main__, "Net", Net)
        with Pool(general_constant['n_pools']) as p:
            # Apply the function to the sequences
            # results = list(tqdm(p.imap_unordered(evaluate, original_sequence), total=len(test_x_name)))
            results = []
            for e in tqdm(original_sequence, total=len(test_x_name)):
                if len(list(filter(lambda x: x != 0, e[0]))) < general_constant['min_valid_apis']:
                    print('Discarded')
                    continue
                result = evaluate(e)
                results.append(result)
        write_evaluation_result(results)
        show_results(results)
    
    
    elif mode == 'random_attack':
        with Pool(general_constant['n_pools']) as p:
            # Apply the function to the sequences
            results = list(tqdm(p.imap_unordered(attack, original_sequence), total=len(test_x_name)))
        write_attack_result('random', results)
    
    elif mode == 'test_set_our_strategy':
        for e in tqdm(original_sequence, total=len(test_x_name)):
            results = test_set_attack(e, 'test_set_our_strategy')
        write_attack_result('test_set_our_strategy', results)
    
    elif mode == 'test_set_rosenberg_strategy':
        for e in tqdm(original_sequence, total=len(test_x_name)):
            results = test_set_attack(e, 'test_set_rosenberg_strategy')
        write_attack_result('test_set_rosenberg_strategy', results)
    
    elif mode == 'test_set_random_strategy':
        for e in tqdm(original_sequence, total=len(test_x_name)):
            results = test_set_attack(e, 'test_set_random_strategy')
        write_attack_result('test_set_random_strategy', results)
    
    elif mode == 'malware_00375_our_strategy':
        for e in tqdm(original_sequence, total=len(test_x_name)):
            results = test_set_attack(e, 'malware_00375_our_strategy')
        write_attack_result('malware_00375_our_strategy', results)
    
    elif mode == 'malware_00375_rosenberg_strategy':
        for e in tqdm(original_sequence, total=len(test_x_name)):
            results = test_set_attack(e, 'malware_00375_rosenberg_strategy')
        write_attack_result('malware_00375_rosenberg_strategy', results)
    
    elif mode == 'malware_00375_random_strategy':
        for e in tqdm(original_sequence, total=len(test_x_name)):
            results = test_set_attack(e, 'malware_00375_random_strategy')
            # write_attack_result('malware_00375_random_strategy', results)

    elif mode == 'combo_attack':
        results = []
        unique_sample_names = np.unique(test_x_sample_names)
        for unique_sample_name in tqdm(unique_sample_names, total=len(unique_sample_names)):
            # Get the indexes of the sample
            sample_indexes = np.where(test_x_sample_names == unique_sample_name)[0]
            # Get the sample
            names = test_x_name[sample_indexes]
            # Get the behaviors
            behaviors = test_x_behaviors[sample_indexes]
            # Get the imported apis
            imported_apis = test_x_imported[sample_indexes]
            # Get the called apis
            called_apis = test_x_called[sample_indexes]
            # Get the sample name
            sample_name = itertools.repeat(unique_sample_name)
            # Get the mode
            modes = itertools.repeat('combo_attack')
            # Create the sequence
            original_sequences = list(zip(names, behaviors, imported_apis, sample_name, called_apis, modes))
            # Apply the attack
            results.append(combo_attack(original_sequences))
        write_attack_result('combo_attack', results)

    elif mode == 'stats':
        results = []
        for e in tqdm(original_sequence, total=len(test_x_name)):
            results.append(stats_on_imported(e))
        write_attack_result('stats', results)
        
    
    else:
        raise WrongModeException(f'Wrong mode {mode}. Please choose between attack and evaluation')