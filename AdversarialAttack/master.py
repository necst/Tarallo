from utils.dispatcher import dispatcher
from config.config_paths import general_paths
from config.config_general import general_constant
#from utils.black_box_model import evaluate_zhang
from utils.parse_results import parse_results, compare_results
# from utils.model import *
import warnings
import logging
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import re
import pickle

# Print the menu options
def print_menu():
    print("What do you want to do?")
    print("1) Start the attack")
    print("2) Evaluate")
    print("3) Print statistics from log file")
    print("4) Start the random on the test set with random strategy")
    print("5) Start the attack on the test set with Rosenberg's strategy")
    print("6) Start the attack on the test set with our strategy")
    print("7) Print comparison statistics on the test set")
    print("8) Start the attack on malware_dataset 00375 with our strategy")
    print("9) Start the attack on malware_dataset 00375 with Rosenberg's strategy")
    print("10) Start the attack on malware_dataset 00375 with random strategy")
    print("11) Print comparison statistics on malware_dataset 00375 between our strategy and Rosenberg's strategy")
    print("12) Print plots of end-to-end attack")
    print("13) Evaluate using Zhang's model on original malware dataset")
    print("14) Evaluate using Zhang's model on patched malware dataset")
    print("15) Print Zhang original evaluation statistics")
    print("16) Print Zhang patched evaluation statistics")
    print("17) Compare Zhang patched and original evaluation statistics")
    print("18) Start the combo attack")
    print("19) Statistics on attack")
    print("0) Exit")


# Print divisor between menu prints
def print_divisor():
    print("\n")
    print("-" * 35)
    print("\n")

def setup_logger(mode, start_time_string, level=logging.INFO):
    logger    = logging.getLogger("adversarial_attack")
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    file_path = os.path.join(general_paths['hijacking_data_dir'], mode + start_time_string+".log")
    handler   = logging.FileHandler(file_path)        
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    
    logger.info(f"""
    threshold_imported_api: {general_constant['threshold_imported_api']},
    max_it_number: {general_constant['max_it_number']},
    n_pools: {general_constant['n_pools']},
    min_valid_apis: {general_constant['min_valid_apis']},
    """)

    return file_path

def list_log_files():
    try:
        tesi_cwd = os.environ['TESI_CWD']
    except:
        print("Error: TESI_CWD environment variable not set")
        return []
    log_dir = os.path.join(tesi_cwd, "hijacking_data")
    log_files = []
    for file in os.listdir(log_dir):
        if file.endswith(".log"):
            log_files.append([file, os.path.join(log_dir, file)]) 
    return log_files


# Menu managment
def menu():
    global status_file_path
    global start_time_string
    while True:
        print_menu()
        choice = input("> ")
        choice = choice.replace('\n', '')

        if choice == '1' or choice == '18' or choice == '19':
            if choice == '1':
                mode = 'attack'
            elif choice == '18':
                mode = 'combo_attack'
            elif choice == '19':
                mode = 'stats'


            # Attack the test set with our strategy
        
            features_path = os.path.join(general_paths['features_dir'], general_paths['original_features_filename'])
            print(f'[+] Attack on these features: {features_path}')
            print("Do you want to continue? (y/n)")
            print('Select "n" if you want to change the features file')

            choice = input("> ")
            choice = choice.replace('\n', '')

            if choice == 'n':
                filename = askopenfilename( initialdir = general_paths['features_dir'] )
                general_paths['original_features_filename'] = filename

            start_time_string = time.strftime("%d_%m--%H%M%S")
            status_file_path = setup_logger('adv_att_', start_time_string, logging.INFO)

            with open(os.path.join(general_paths['hijacking_data_dir'], 'start_time'), 'w') as f:
                f.write(start_time_string)

            print(f'Starting the attack with the following parameters:')
            print(f"threshold_imported_api: {general_constant['threshold_imported_api']},")
            print(f"max_it_number: {general_constant['max_it_number']},")
            print(f"n_pools: {general_constant['n_pools']},")
            print(f"min_valid_apis: {general_constant['min_valid_apis']},")

            input("Press Enter to continue... (Ctrl+C to stop the attack)")
            
            dispatcher(mode)
        
        elif choice == '2':
            features_path = os.path.join(general_paths['features_dir'], general_paths['patched_features_filename'])
            print(f'[+] Evaluation on these features: {features_path}')
            print("Do you want to continue? (y/n)")
            print('Select "n" if you want to change the features file')
            
            choice = input("> ")
            choice = choice.replace('\n', '')

            if choice == 'n':
                filename = askopenfilename( initialdir = general_paths['features_dir'] )
                general_paths['patched_features_filename'] = filename
    
            dispatcher('evaluation')
        
        elif choice == '3':
            print("Choose the log file:")
            # log_files = list_log_files()
            # for i, el in enumerate(log_files):
            #     print(f'{i}) {el[0]}')
            # idx = input("> ").replace('\n', '')
            filename = askopenfilename( initialdir = general_paths['hijacking_data_dir'] )            
            print_statistics(filename)
            # print_statistics(log_files[int(idx)][1])
        
        elif choice == '4':
            # Attack the test set with Random strategy
            tesi_cwd = os.environ['TESI_CWD'] # Works with export command
            features_path = os.path.join(tesi_cwd,'Malware-Detection-API-Sequence-Intrinsic-Features/data_demo.npz')
            print(f'[+] Attack on these features: {features_path}')
            
            start_time_string = time.strftime("%d_%m--%H%M%S")
            status_file_path = setup_logger('random_adv_att_', start_time_string, logging.INFO)

            with open(os.path.join(general_paths['hijacking_data_dir'], 'start_time'), 'w') as f:
                f.write(start_time_string)

            seed = 42
            np.random.seed(seed)
            
            print(f'Starting the attack with the following parameters:')
            print(f"max_it_number: {general_constant['max_it_number']},")
            print(f"n_pools: {general_constant['n_pools']},")
            print(f'Random seed: {seed}')

            input("Press Enter to continue... (Ctrl+C to stop the attack)")

            dispatcher('test_set_random_strategy')

        
        elif choice == '5':
            # Attack the test set with Rosenberg' strategy
            tesi_cwd = os.environ['TESI_CWD'] # Works with export command
            features_path = os.path.join(tesi_cwd,'Malware-Detection-API-Sequence-Intrinsic-Features/data_demo.npz')
            print(f'[+] Attack on these features: {features_path}')
            
            start_time_string = time.strftime("%d_%m--%H%M%S")
            status_file_path = setup_logger('Rosenberg_adv_att_', start_time_string, logging.INFO)

            with open(os.path.join(general_paths['hijacking_data_dir'], 'start_time'), 'w') as f:
                f.write(start_time_string)

            print(f'Starting the attack with the following parameters:')
            print(f"max_it_number: {general_constant['max_it_number']},")
            print(f"n_pools: {general_constant['n_pools']},")

            input("Press Enter to continue... (Ctrl+C to stop the attack)")

            seed = 45
            np.random.seed(seed)
            
            dispatcher('test_set_rosenberg_strategy')

        elif choice == '6':
            # Attack the test set with our strategy
            tesi_cwd = os.environ['TESI_CWD']
            features_path = os.path.join(tesi_cwd,'Malware-Detection-API-Sequence-Intrinsic-Features/data_demo.npz')
            print(f'[+] Attack on these features: {features_path}')
            
            start_time_string = time.strftime("%d_%m--%H%M%S")
            status_file_path = setup_logger('our_strategy_adv_att_', start_time_string, logging.INFO)

            with open(os.path.join(general_paths['hijacking_data_dir'], 'start_time'), 'w') as f:
                f.write(start_time_string)

            print(f'Starting the attack with the following parameters:')
            print(f"max_it_number: {general_constant['max_it_number']},")
            print(f"n_pools: {general_constant['n_pools']},")

            input("Press Enter to continue... (Ctrl+C to stop the attack)")

            dispatcher('test_set_our_strategy')

        elif choice == '7':
            # Print comparison statistics on the test set

            filename1 = 'our_strategy_adv_att_28_03--101402.log'
            filename2 = 'Rosenberg_adv_att_28_03--042412.log'
            filename3 = 'random_adv_att_29_03--164753.log'
            filename1 = os.path.join('', filename1)
            filename2 = os.path.join('', filename2)
            filename3 = os.path.join('', filename3)            
            print_comparison_test_set(filename1, filename2, filename3)

        elif choice == '8' or choice == '9' or choice == '10':
            # Attack the test set with our strategy or Rosenberg's strategy

            strategies = ['malware_00375_our_strategy', 'malware_00375_rosenberg_strategy', 'malware_00375_random_strategy']
            strategy = strategies[int(choice) - 8]

            if 'random' in strategy:
                seed = 42
                np.random.seed(seed)

            features_path = os.path.join(general_paths['features_dir'], general_paths['original_features_filename'])
            print(f'[+] Attack on these features: {features_path}')
            print(f"max_it_number: {general_constant['max_it_number']},")
            print(f"n_pools: {general_constant['n_pools']},")
            print(f'Random seed or None: {seed if seed else None}')
            print("Do you want to continue? (y/n)")
            print('Select "n" if you want to change the features file')

            choice = input("> ")
            choice = choice.replace('\n', '')

            if choice == 'n':
                filename = askopenfilename( initialdir = general_paths['features_dir'] )
                general_paths['original_features_filename'] = filename

            start_time_string = time.strftime("%d_%m--%H%M%S")
            with open(os.path.join(general_paths['hijacking_data_dir'], 'start_time'), 'w') as f:
                f.write(start_time_string)
            
            status_file_path = setup_logger(strategy+'_', start_time_string, logging.INFO)
            print(f'[+] malware dataset 00375 {strategy}')
            dispatcher(strategy)

        elif choice == '11':
            # Print comparison statistics on the 375

            filename1 = ''
            filename2 = ''
            filename3 = ''
            filename1 = os.path.join('', filename1)
            filename2 = os.path.join('', filename2)
            filename3 = os.path.join('', filename3)
            
            print_comparison_test_set(filename1, filename2, filename3)

        elif choice == '12':
            # Print plots of end-to-end attack
            filename1 = 'adv_att_30_03--203707.log'
            filename2 = 'evaluation_result_modified_375_1_4.pkl'
            filename1 = os.path.join('', filename1)
            filename2 = os.path.join('', filename2)
            print_comparison_375_end_to_end(filename1, filename2)
            print_375_end_to_end_evasion_score(filename1, filename2)
        elif choice == '13':
            from utils.black_box_model import evaluate_zhang
            original_features_path = os.path.join(general_paths['features_dir'], general_paths['zhang_original_features_dir'])
            original_results_path   = os.path.join(general_paths['evaluation_dir'], general_paths['zhang_original_results_filename'])
            evaluate_zhang(original_features_path, original_results_path)
        elif choice == '14':
            from utils.black_box_model import evaluate_zhang
            patched_features_path = os.path.join(general_paths['features_dir'], general_paths['zhang_patched_features_dir'])
            patched_results_path   = os.path.join(general_paths['evaluation_dir'], general_paths['zhang_patched_results_filename'])
            evaluate_zhang(patched_features_path, patched_results_path)
        elif choice == '15':
            original_results_path   = os.path.join(general_paths['evaluation_dir'], general_paths['zhang_original_results_filename'])
            parse_results(original_results_path)
        elif choice == '16':
            patched_results_path   = os.path.join(general_paths['evaluation_dir'], general_paths['zhang_patched_results_filename'])
            parse_results(patched_results_path)
        elif choice == '17':
            original_results_path   = os.path.join(general_paths['evaluation_dir'], general_paths['zhang_original_results_filename'])
            patched_results_path   = os.path.join(general_paths['evaluation_dir'], general_paths['zhang_patched_results_filename'])
            compare_results(original_results_path, patched_results_path)

        elif choice == '0':
            quit()
        else:
            print("Wrong choice!")
        print_divisor()

def compute_statistics(filepath=None):
    global status_file_path
    if filepath is not None:
        status_file_path = filepath
    
    count_discarded = 0
    count_success   = 0
    count_fail      = 0
    count_classified_as_goodware = 0
    overhead        = []
    sample_list     = [] # List of samples successfully attacked with 

    with open(status_file_path, 'r') as f:
        lines = f.readlines()

    threshold_imported_api = None
    max_it_number          = None    
    n_pools                = None
    min_valid_apis         = None

    for line in lines:

        # Parsing initial setting of the experiment
        if threshold_imported_api == None and "threshold_imported_api" in line:
            line = line.strip('\n')
            threshold_imported_api = float(line.split(' ')[-1].strip(','))
        if max_it_number == None and "max_it_number" in line:
            line = line.strip('\n')
            max_it_number = float(line.split(' ')[-1].strip(','))
        if n_pools == None and "n_pools" in line:
            line = line.strip('\n')
            n_pools = float(line.split(' ')[-1].strip(','))
        if min_valid_apis == None and "min_valid_apis" in line:
            line = line.strip('\n')
            min_valid_apis = float(line.split(' ')[-1].strip(','))


        if "INFO" in line and '[' in line:
            count_success += 1
        elif "WARNING" in line and '[' in line and 'returning' not in line:
            count_discarded += 1
        elif "ERROR" in line and '[' in line:
            count_fail += 1
        elif "CRITICAL" in line and '[' in line:
            count_fail += 1
        
        if "INFO" in line and 'OK' in line:
            regex = r"\[(\w+)\] OK - (\d+) added APIs - (\d+) valid APIs - (\d+) imported APIs"
            match = re.search(regex, line)
            if match:
                sample_name = match.group(1)
                added_apis = match.group(2)
                valid_apis = match.group(3)
                imported_apis = match.group(4)
                
                sample_list.append({'sample_name':sample_name, 'added_apis': int(added_apis), 
                                    'valid_apis': int(valid_apis), 'imported_apis': int(imported_apis), 'evasion': True})
                overhead.append(int(added_apis))
            else:
                # Try old version of the log file
                regex = r"OK - (\d+) added APIs"
                match = re.search(regex, line)

                if match:
                    added_apis = match.group(1)
                    overhead.append(int(added_apis))
                else:
                    print("No match found")
        if "ERROR" in line:
            regex = r"\[(\w+)\] Failed - (\d+) iterations reached - (\d+) imported APIs"
            match = re.search(regex, line)
            if match:
                sample_name = match.group(1)
                imported_apis = match.group(3)
                sample_list.append({'sample_name':sample_name, 'added_apis': None, 
                                    'valid_apis': None, 'imported_apis': int(imported_apis), 'evasion': False})
        if "WARNING" in line and "Already classified as not malicious" in line:
            count_classified_as_goodware += 1

    
    return {
        'threshold_imported_api': threshold_imported_api,
        'max_it_number': max_it_number,
        'n_pools': n_pools,
        'min_valid_apis': min_valid_apis,
        'count_discarded': count_discarded,
        'count_success': count_success,
        'count_fail': count_fail,
        'count_classified_as_goodware': count_classified_as_goodware,
        'sample_list': sample_list,
        'overhead': overhead
    }


def print_comparison_test_set(filename1=None, filename2=None, filename3=None):
    if filename1 == None:
        print("Choose the log file of our strategy:")
        filename1 = askopenfilename( initialdir = general_paths['hijacking_data_dir'] )            
    our_strategy       = compute_statistics(filename1)

    if filename2 == None:
        print("Choose the log file of Rosenberg strategy:")
        filename2 = askopenfilename( initialdir = general_paths['hijacking_data_dir'] )            
    rosenberg_strategy = compute_statistics(filename2)

    if filename3 != None:
        random_strategy = compute_statistics(filename3)
    
    overhead_comparison_histogram(our_strategy['overhead'], rosenberg_strategy['overhead']) # 
    overhead_comparison_box_plot(our_strategy['overhead'], rosenberg_strategy['overhead'])
    if filename3:
        effectiveness_comparison_line_plot(our_strategy, rosenberg_strategy, random_strategy)
        print(f'[Our Approach] Effectiveness score at 20% overhead: {effectiveness_at_overhead_percentage(our_strategy, 20):.4f}')
        print(f'[Our Approach] Effectiveness score at 50% overhead: {effectiveness_at_overhead_percentage(our_strategy, 50):.4f}')
        print(f'[Our Approach] Effectiveness score at 70% overhead: {effectiveness_at_overhead_percentage(our_strategy, 70):.4f}')
        print(f'[Our Approach] Effectiveness score at 90% overhead: {effectiveness_at_overhead_percentage(our_strategy, 90):.4f}')

        print(f'[Rosenberg] Effectiveness score at 20% overhead: {effectiveness_at_overhead_percentage(rosenberg_strategy, 20):.4f}')
        print(f'[Rosenberg] Effectiveness score at 50% overhead: {effectiveness_at_overhead_percentage(rosenberg_strategy, 50):.4f}')
        print(f'[Rosenberg] Effectiveness score at 70% overhead: {effectiveness_at_overhead_percentage(rosenberg_strategy, 70):.4f}')
        print(f'[Rosenberg] Effectiveness score at 90% overhead: {effectiveness_at_overhead_percentage(rosenberg_strategy, 90):.4f}')

def effectiveness_at_overhead_percentage(strategy, overhead_percentage):
    # compute effectiveness at a given overhead ratio
    count_success = 0
    count_fail = 0
    window_len = 1000
    overhead = window_len * (overhead_percentage/100)
    for i in range(len(strategy['overhead'])):
        if strategy['overhead'][i] < overhead:
            count_success += 1
        else:
            count_fail += 1

    return count_success / (strategy['count_success'] + strategy['count_fail'])


def print_comparison_375_end_to_end(filename1=None, filename_evaluation_results=None):
    if filename1 == None:
        print("Choose the log file of our strategy:")
        filename1 = askopenfilename( initialdir = general_paths['hijacking_data_dir'] )    

    if filename_evaluation_results == None:
        print("Choose the evaluation results file:")
        filename1 = askopenfilename( initialdir = general_paths['evaluation_dir'] )      

    our_strategy_statistics       = compute_statistics(filename1)
    # num_imported_apis_list = range(3, 150, 25)
    num_imported_apis_list = [3, 10, 25, 50]
    overhead_limits = range(10,951,10)
    plot_info_list = []

    # Filter only samples that were evaluated
    # our_strategy_statistics['sample_list'] = [sample for sample in our_strategy_statistics['sample_list'] if sample['sample_name'] in evaluation_dict.keys()]
    # print(our_strategy_statistics['sample_list'])
    for imported_apis_limit in num_imported_apis_list:

        # Extract samples with at least n imported APIs, both failed and successful
        sample_list_imported_apis_limited = extract_samples_with_at_least_n_imported_apis(our_strategy_statistics['sample_list'], imported_apis_limit)
        print(f'Number of samples with at least {imported_apis_limit} imported APIs: {len(sample_list_imported_apis_limited)}')
        effectiveness_list = []
        for overhead_limit in overhead_limits:
            # Extract samples with at most n overhead
            sample_list_imported_apis_limited_and_overhead_limited = extract_samples_with_at_most_n_overhead(sample_list_imported_apis_limited, overhead_limit)
            # Compute effectiveness
            if overhead_limit == overhead_limits[-1]:
                num_success = len([sample for sample in sample_list_imported_apis_limited_and_overhead_limited if sample['evasion'] == True])
                num_samples = len(sample_list_imported_apis_limited_and_overhead_limited)
                print(f'Number of samples with at least {imported_apis_limit} imported APIs and at most {overhead_limit} overhead: {num_samples}')
                print(f'Number of successful samples with at least {imported_apis_limit} imported APIs and at most {overhead_limit} overhead: {num_success}')
            effectiveness = compute_effectiveness_score_sample_list(sample_list_imported_apis_limited_and_overhead_limited)
            effectiveness_list.append(effectiveness)
            
        plot_info_list.append({'x': overhead_limits, 'y': effectiveness_list, 'label': f'Min Imported APIs: {round(100*imported_apis_limit/255)} % of total'})        
    print_line_plots(plot_info_list)
    

def print_375_end_to_end_evasion_score(filename1=None, filename_evaluation_results=None):
    if filename1 == None:
        print("Choose the log file of our strategy:")
        filename1 = askopenfilename( initialdir = general_paths['hijacking_data_dir'] )    

    if filename_evaluation_results == None:
        print("Choose the evaluation results file:")
        filename1 = askopenfilename( initialdir = general_paths['evaluation_dir'] )      

    
    our_strategy_statistics       = compute_statistics(filename1)

    evaluation_dict = extract_evaluation_dict_from_pickle(filename_evaluation_results)

    samples_attacked_successfully = our_strategy_statistics['sample_list']

    print(f'Number of samples attacked successfully: {len(samples_attacked_successfully)}')
    print(f'Number of samples attacked unsuccessfully: {our_strategy_statistics["count_fail"]}')
    # for sample in samples_attacked_successfully:
    #     print(f"Sample {sample['sample_name']}: {compute_sample_evasion_score_given_evaluation(evaluation_dict, sample['sample_name'])}")
    #     try:
    #         print(F"Sample {sample['sample_name']}: {evaluation_dict[sample['sample_name']]}")
    #     except:
    #         print(f"Sample {sample['sample_name']}: not in evaluation results")

    print(f"Number of samples for which end-to-end attack produced at least one evasive execution: {len([s for s in samples_attacked_successfully if compute_sample_evasion_score_given_evaluation(evaluation_dict, s['sample_name']) > 0])}")
    
    # Consider samples with at least n imported APIs
    num_imported_apis_list = [3, 12, 25, 50]
    ratio_list = []
    for lim in num_imported_apis_list:
        samples_attacked_successfully = extract_samples_with_at_least_n_imported_apis(samples_attacked_successfully, lim)
        print(f' Number of samples with at least {lim} imported APIs: {len(samples_attacked_successfully)}')

        print(f"Number of samples for which end-to-end attack with at least {lim} API injected produced at least one evasive execution: {len([s for s in samples_attacked_successfully if compute_sample_evasion_score_given_evaluation(evaluation_dict, s['sample_name']) > 0])}")
        ratio = len([s for s in samples_attacked_successfully if compute_sample_evasion_score_given_evaluation(evaluation_dict, s['sample_name']) > 0]) / len(samples_attacked_successfully)
        print(f"Ratio: {ratio}")
        ratio_list.append(ratio)
    
    # Bar plot with the ratio of samples with at least n imported APIs that have at least one evasive execution
    apis_we_can_inject   = 255
    labels_fontsize      = 18
    ticks_fontsize       = 12
    
    x_axis = [round(100*t/apis_we_can_inject) for t in  num_imported_apis_list]
    plt.bar( x_axis, ratio_list)
    plt.xticks([0,1,5,10,15,20], fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    # plt.set_title('Comparison of Overhead Between Strategies')
    plt.xlabel('Threshold of imported APIs (% of total)', fontsize=labels_fontsize)
    plt.ylabel('Ratio of samples with at least one evasive execution', fontsize=labels_fontsize)
    plt.show()

   
def num_samples_at_least_1_evasion(sample_list, evaluation_dict):
    """Number of samples for which end-to-end attack produced at least one evasive execution"""
    return len([s for s in sample_list if compute_sample_evasion_score_given_evaluation(evaluation_dict, s['sample_name']) > 0])


def num_samples_0_evasion(sample_list, evaluation_dict):
    """Number of samples for which end-to-end attack produced at least one evasive execution"""
    return len([s for s in sample_list if compute_sample_evasion_score_given_evaluation(evaluation_dict, s['sample_name']) == 0])


def extract_evaluation_dict_from_pickle(pickle_filename):
    """ Extract the evaluation results from the pickle file and 
    return a dictionary with the sample name as key and the list of evaluation results as value"""

    with open(pickle_filename, 'rb') as f:
        evaluation_results = pickle.load(f)
    evaluation_dict = {}
    for result in evaluation_results:
        if result[0] not in evaluation_dict.keys():
            evaluation_dict[result[0]] = [result[1]]
        else:
            evaluation_dict[result[0]].append(result[1])
    return evaluation_dict
        


def compute_sample_evasion_score_given_evaluation(evaluation_results, sample_name):
    """ Compute the evasion score of a sample, 
    where the evasion score is computed as the percentage of executions of the sample that were not detected as malware"""
    if sample_name not in evaluation_results.keys():
        return -1
    else:
        return sum([1 for e in evaluation_results[sample_name] if e == 0]) / len(evaluation_results[sample_name])

def compute_end_to_end_effectiveness_score_sample_list(sample_list, evaluation_results):
    """
    Compute the effectiveness score of a sample list, where the effectiveness is computed as the percentage of samples
      that were NOT detected as malware at least once.
    """
    at_least_1_count_success = 0
    at_least_1_count_fail = 0

    evaluation_dict = {}
    for result in evaluation_results:
        if result[0] not in evaluation_dict.keys():
            evaluation_dict[result[0]] = [result[1]]
        else:
            evaluation_dict[result[0]].append(result[1])
        
    #
    for sample in sample_list:
        if sample['sample_name'] in evaluation_dict.keys():
            if 0 in evaluation_dict[sample['sample_name']]:
                """ If at least one execution of the sample was not detected as malware, then the sample is considered as a success"""
                at_least_1_count_success += 1
            else:
                at_least_1_count_fail += 1

    if at_least_1_count_success + at_least_1_count_fail == 0:
        return 0
    else:
        return at_least_1_count_success / (at_least_1_count_success + at_least_1_count_fail)


def compute_evasion_score_per_sample(evaluation_results, sample_name, strategy='average'):
    scores = []
    for result in evaluation_results:
        if result[0] == sample_name:
            scores.append(result[1])
    if len(scores) == 0:
        raise Exception(f'Could not find sample {sample_name} in evaluation results')
    if strategy == 'average':
        return sum(scores) / len(scores)
    elif strategy == 'at_least_one_evasion':
        for score in scores:
            if score == 1:
                return 1
        return 0
    return 

def extract_samples_with_at_least_n_imported_apis(sample_list, n_imported_apis):
    """ Extract samples with at least n imported APIs,
    mimicking the experiment for which we discard samples with less than n imported APIs"""
    new_sample_list = []
    for sample in sample_list:
        if sample['imported_apis'] >= n_imported_apis:
            new_sample_list.append(sample)
    return new_sample_list

def extract_samples_with_at_most_n_overhead(sample_list, n_overhead):
    new_sample_list = []
    for sample in sample_list:
        if sample['added_apis'] == None:
            # Sample could not be attacked
            new_sample_list.append(sample)
        elif sample['added_apis'] <= n_overhead:
            # Sample attacked successfully with at most n_overhead overhead
            new_sample_list.append(sample)

    return new_sample_list

def compute_effectiveness_score_sample_list(sample_list):
    """ Compute the effectiveness score of a sample list, 
    assuming that the list contains also the samples for which the attack failed
    """
    if len(sample_list) != 0: 
        return len([sample for sample in sample_list if sample['evasion'] == True]) / len(sample_list)
    else:
        return 0


def print_line_plots(plot_info_list, x_label=None, y_label=None):
    for plot_info in plot_info_list:
        x = plot_info['x']
        score = plot_info['y']
        label = plot_info['label']
        if 'linestyle' in plot_info.keys():
            linestyle = plot_info['linestyle']
        else:
            linestyle = '-'
        plt.plot(x, score, label=label, linestyle=linestyle)
        plt.fill_between(x, score, alpha=0.2)
    labels_fontsize      = 18
    annotations_fontsize = 14

    # Set the axis labels and legend
    if x_label != None:
        plt.xlabel(x_label, fontsize=labels_fontsize)
    if y_label != None:
        plt.ylabel(y_label, fontsize=labels_fontsize)
    
    plt.xlabel("Overhead limit (API calls)", fontsize=labels_fontsize)
    plt.ylabel("Effectiveness score", fontsize=labels_fontsize)
    plt.legend(fontsize=labels_fontsize)

    # Show the plot
    plt.show()  

def overhead_comparison_histogram(my_strategy_overheads, other_strategy_overheads, save_path=None):
    bin_size = 50

    # create bins for each strategy
    my_strategy_bins = np.arange(0, 901, bin_size)
    other_strategy_bins = np.arange(0, 901, bin_size)

    # create histograms
    my_strategy_hist, _ = np.histogram(my_strategy_overheads, bins=my_strategy_bins)
    other_strategy_hist, _ = np.histogram(other_strategy_overheads, bins=other_strategy_bins)

    # create x tick labels
    labels = [f"{i}-{i+bin_size-1}" for i in range(0, 901, bin_size)]

    # plot histograms
    fig, ax = plt.subplots()
    ax.bar(my_strategy_bins[:-1], my_strategy_hist, width=bin_size, alpha=0.5, label='Proposed Approach')
    ax.bar(other_strategy_bins[:-1], other_strategy_hist, width=bin_size, alpha=0.5, label='Rosenberg et al.')

    # set x tick labels
    ticks_fontsize = 12
    labels_fontsize = 18
    ax.set_xticks(my_strategy_bins, fontsize=ticks_fontsize)
    ax.set_xticklabels(labels, fontsize=labels_fontsize-7)
    
    # add labels and legend
    ax.set_xlabel('Overhead (API calls)', fontsize=labels_fontsize)
    ax.set_ylabel('Number of Samples', fontsize=labels_fontsize)
    # ax.set_title('Comparison of Overhead Between Strategies')
    ax.legend(fontsize=labels_fontsize)

    if save_path != None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

    
def overhead_comparison_box_plot(my_strategy_overheads, other_strategy_overheads):
    data_to_plot = [my_strategy_overheads, other_strategy_overheads]

    # create a box plot with custom colors and labels
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data_to_plot, patch_artist=True, notch=False, vert=False, widths=0.7,
                    # boxprops=dict(facecolor="C0", color="k"),
                    # whiskerprops=dict(color="k", linestyle="-"),
                    # capprops=dict(color="k", linestyle="-"),
                    flierprops=dict(marker="o", markerfacecolor="C2", markersize=2, alpha=0.5),
                    medianprops=dict(linestyle='None'), meanprops=dict(marker='None'),
                    showmeans=False,
                    )

    # set the axis labels and title
    ax.set_yticklabels(["Our Strategy", "Rosenberg et al."], fontsize=18)
    ax.set_xlabel("Overhead (API calls)", fontsize=12)
    # ax.set_title("Comparison of Overhead Distributions", fontsize=14)
    # add horizontal grid lines
    ax.grid(True, axis="x", linestyle="-", linewidth=0.5, alpha=0.5)

    # add legends and annotations
    colors = ["C0", "C1"]
    labels = ["Our Strategy", "Rosenberg et al."]
    for i in range(len(data_to_plot)):
        bp["boxes"][i].set_facecolor(colors[i]) 
        # ax.annotate("Median: {:.1f}".format(bp["medians"][i].get_xdata()[0]),
        #             xy=(bp["medians"][i].get_xdata()[0], i),
        #             xytext=(5, 5), textcoords="offset points",
        #             fontsize=10, color=colors[i], ha="left", va="center")
    ax.legend([bp["boxes"][0], bp["boxes"][1]], labels, fontsize=18)

    # show the plot
    plt.show()

def process_effectiveness_stats(strategy_statistics, overhead_limits):
    """Compute the effectiveness of a strategy given the overhead limits."""
    # compute the number of samples for the strategy
    my_strategy_count    = strategy_statistics['count_success'] + strategy_statistics['count_fail']

    # compute the number of samples for the strategy that fall within the overhead limits
    my_strategy_effective_count    = [0] * len(overhead_limits)
    
    for overhead in strategy_statistics['overhead']:
        for i, limit in enumerate(overhead_limits):
            if overhead <= limit:
                my_strategy_effective_count[i] += 1
    
    # compute the effectiveness of the strategy
    my_strategy_effectiveness    = [count / my_strategy_count    for count in my_strategy_effective_count]
    # return the effectiveness of the strategy
    return my_strategy_effectiveness


def effectiveness_comparison_line_plot(my_strategy_statistics, other_strategy_statistics, third_strategy_statistics=None):
    # Overhead limits for each score
    overhead_limits = range(50, 951, 50)
    
    # Effectiveness scores for our strategy and the state-of-the-art strategy
    my_scores   = process_effectiveness_stats(my_strategy_statistics, overhead_limits)
    # Create a line plot
    plt.plot(overhead_limits, my_scores, label="Proposed Approach")
    plt.fill_between(overhead_limits, my_scores, alpha=0.2)

    
    if other_strategy_statistics:
        # Effectiveness scores for our strategy and the state-of-the-art strategy
        sota_scores = process_effectiveness_stats(other_strategy_statistics, overhead_limits) 
        # Create a line plot
        plt.plot(overhead_limits, sota_scores, label="Rosenberg et al.")
        plt.fill_between(overhead_limits, sota_scores, alpha=0.2)


    # If there are three strategies, calculate the effectiveness scores for the third strategy
    if third_strategy_statistics:
        third_scores = process_effectiveness_stats(third_strategy_statistics, overhead_limits)
        # Create a line plot for the third strategy
        plt.plot(overhead_limits, third_scores, label="Random Insertion")
        plt.fill_between(overhead_limits, third_scores, alpha=0.2)

    # In academic writing range from 8-12 pt for axis labels, tick labels, and legends, and 10-14 pt for title and annotations.

    labels_fontsize      = 18
    ticks_fontsize       = 12

    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)

    # Set the axis labels and legend
    plt.xlabel("Overhead limit (API calls)", fontsize=labels_fontsize)
    plt.ylabel("Effectiveness score", fontsize=labels_fontsize)
    plt.legend(fontsize=labels_fontsize)

    # Show the plot
    plt.show()

    ###


    # # Set up the plot
    # fig, ax = plt.subplots()

    # # Plot the shaded regions
    # ax.fill_between(overhead_limits, my_scores, alpha=0.5, label='Proposed Approach')
    # ax.fill_between(overhead_limits, sota_scores, alpha=0.5, label='State of the Art')

    # # Plot the lines
    # ax.plot(overhead_limits, my_scores, label='Proposed Approach')
    # ax.plot(overhead_limits, sota_scores, label='State of the Art')

    # # Set the axis labels and legend
    # ax.set_xlabel('Overhead Limit (API Calls)')
    # ax.set_ylabel('Effectiveness Score')
    # ax.legend()

    # # Show the plot
    # plt.show()

def plot_overhead(overhead):
    
    plt.figure()
    plt.hist(overhead, bins=25)
    plt.xlabel('Overhead (number of API calls added)')
    plt.ylabel('Number of samples')
    plt.show()


def print_statistics(filepath=None):
    if filepath is not None:
        global status_file_path
        status_file_path = filepath
    statistics = compute_statistics()
    if statistics['count_discarded'] + statistics['count_success'] + statistics['count_fail'] == 0:
        print("No samples found")
        return
    
    print(f'Attack carried out with the following parameters:')
    print(f'threshold_imported_api: {statistics["threshold_imported_api"]},')
    print(f'max_it_number: {statistics["max_it_number"]},')
    print(f'n_pools: {statistics["n_pools"]},')
    print(f'min_valid_apis: {statistics["min_valid_apis"]},')

    print("Total: {}".format(statistics['count_discarded'] + statistics['count_success'] + statistics['count_fail']))
    print("Discarded: {}".format(statistics['count_discarded']))
    print("Discarded because classified as benign: {}".format(statistics['count_classified_as_goodware']))
    print("Success: {}".format(statistics['count_success']))
    print("Fail: {}".format(statistics['count_fail']))

    print("Discarded : {:.2f}% of total samples".format(100 * statistics['count_discarded'] / (statistics['count_discarded'] + statistics['count_success'] + statistics['count_fail'])))

    if statistics['count_success'] + statistics['count_fail'] == 0:
        print("No attacked samples found")
        return
    print("Success rate: {:.2f}% of attacked samples".format(100 * statistics['count_success'] / (statistics['count_success'] + statistics['count_fail'])))
    print("Fail rate: {:.2f}% of attacked samples".format(100 * statistics['count_fail'] / (statistics['count_success'] + statistics['count_fail'])))
    print("Average overhead: {:.2f} api calls".format(np.mean(statistics['overhead'])))
    print("Median overhead: {:.2f} api calls".format(np.median(statistics['overhead'])))
    plot_overhead(statistics['overhead'])


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    menu()