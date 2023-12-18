import json

def parse_results(results_path):
    majority_vote = 0
    total_vote = 0
    single_vote = 0
    benign_at_least_once = 0
    total_malware_count = 0
    total_execution_count = 0

    with open(results_path, 'r') as f:
        results = json.load(f)
    
    for key, value in results.items():
        lenght = len(value)
        malware_count = value.count(1)
        benigh_count = value.count(0)

        total_malware_count += malware_count
        total_execution_count += lenght

        if malware_count > lenght / 2:
            majority_vote += 1
            single_vote += 1
        elif malware_count > 0:
            single_vote += 1

        if malware_count == lenght:
            total_vote += 1

        if benigh_count > 0:
            benign_at_least_once +=1
        
        
    print("The number of malware according to majority vote: ", majority_vote)
    print(f"{majority_vote} / {len(results)} \t {majority_vote / len(results) * 100}%")

    print("The number of malware according to single vote: ", single_vote)
    print(f"{single_vote} / {len(results)} \t {single_vote / len(results) * 100}%")

    print("The number of benign at least once: ", benign_at_least_once)
    print(f"{benign_at_least_once} / {len(results)} \t {benign_at_least_once / len(results) * 100}%")

    print("The number of malware according to total vote: ", total_vote)
    print(f"{total_vote} / {len(results)} \t {total_vote / len(results) * 100}%")

    print(f"Overall accuracy: {total_malware_count / total_execution_count * 100}%")


def compare_results(results_path_original, results_path_patched):
    """
    Compare the results from original malware and modified malware
    """
    majority_vote = 0
    total_vote = 0
    single_vote = 0
    benign_at_least_once = 0
    total_malware_count = 0
    total_execution_count = 0

    malware_to_compare = [] # malware to check when modified

    with open(results_path_original, 'r') as f:
        results = json.load(f)
    
    for key, value in results.items():
        lenght = len(value)
        malware_count = value.count(1)
        benigh_count = value.count(0)

        total_malware_count += malware_count
        total_execution_count += lenght

        if malware_count > lenght / 2:
            majority_vote += 1
            single_vote += 1
            
        elif malware_count > 0:
            single_vote += 1

        if malware_count == lenght:
            malware_to_compare.append(key)
            total_vote += 1

        if benigh_count > 0:
            benign_at_least_once +=1

    if len(malware_to_compare) == 0:
        print("No malware to compare")
        return

    majority_vote = 0
    total_vote = 0
    single_vote = 0
    benign_at_least_once = 0
    total_malware_count = 0
    total_execution_count = 0
    with open(results_path_patched, 'r') as f:
        results = json.load(f)

    print("Malware to compare: ", len(malware_to_compare))
    print("Total malware after: ", len(results))


    sample_counter = 0
    for key, value in results.items():

        # Check if we need to check the modified version of
        # the malware
        if key not in malware_to_compare:
            continue

        sample_counter += 1

        lenght = len(value)
        malware_count = value.count(1)
        benigh_count = value.count(0)

        total_malware_count += malware_count
        total_execution_count += lenght

        if malware_count > lenght / 2:
            majority_vote += 1
            single_vote += 1
        elif malware_count > 0:
            single_vote += 1

        if malware_count == lenght:
            total_vote += 1

        if benigh_count > 0:
            benign_at_least_once +=1
    print("The number of malware according to majority vote: ", majority_vote)
    print(f"{majority_vote} / {sample_counter} \t {majority_vote / sample_counter * 100}%")

    print("The number of malware according to single vote: ", single_vote)
    print(f"{single_vote} / {sample_counter} \t {single_vote / sample_counter * 100}%")

    print("The number of benign at least once: ", benign_at_least_once)
    print(f"{benign_at_least_once} / {sample_counter} \t {benign_at_least_once / sample_counter * 100}%")

    print("The number of malware according to total vote: ", total_vote)
    print(f"{total_vote} / {sample_counter} \t {total_vote / sample_counter * 100}%")

    print(f"Overall accuracy: {total_malware_count / total_execution_count * 100}%")

    print(f"Overall attack effectiveness: {100 - total_malware_count / total_execution_count * 100}%")

    print("Total number of samples: ", sample_counter)

    print("Total number of malware: ", total_malware_count)

    print("Total number of executions: ", total_execution_count)

