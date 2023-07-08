import requests
from termcolor import cprint
from tqdm import tqdm


HEADERS       = {"Authorization": "REDACTED"}
URL           = "http://localhost:8090/"
CUCKOO_STATUS = "cuckoo/status"
API_LIST_TASK = "tasks/list"
API_DEL_TASK  = "tasks/delete/"


def cuckoo_status():
    # Retrieve cuckoo status
    try:
        r              = requests.get(URL + CUCKOO_STATUS, headers=HEADERS)
    except requests.exceptions.ConnectionError:
        cprint("Cuckoo is not running!", "red", attrs=['bold'])
        return
    if not r.ok:
        cprint("You are not using the right cuckoo api server!", "red", attrs=['bold'])
        return
    tasks          = r.json()['tasks']
    diskspace      = r.json()['diskspace']
    machines       = r.json()['machines']
    avail_machines = machines['available']
    used_analyses  = diskspace['analyses']['used']
    total_analyses = diskspace['analyses']['total']
    
    cprint("Tasks", "blue", attrs=['bold'])
    print(tasks)
    print()
    cprint("Diskspace", "blue", attrs=['bold'])
    print(f"Analyses use {used_analyses/total_analyses*100:.3f}% of {total_analyses/10**9:.1f} GB")
    cprint("Machines", "blue", attrs=['bold'])
    print(f'Machines available {avail_machines}')


def deleter():    
    # Retrieve how many tasks
    try :
        r       = requests.get(URL + API_LIST_TASK, headers=HEADERS)
    except requests.exceptions.ConnectionError:
        cprint("Cuckoo is not running!", "red", attrs=['bold'])
        return
    if not r.ok:
        cprint("You are not using the right cuckoo api server!", "red", attrs=['bold'])
        return
    tasks   = r.json()['tasks']
    n_tasks = len(tasks)
    start   = tasks[0]['id']

    # Delete tasks
    for i in tqdm(range(start, start + n_tasks + 1)):
        r = requests.get(URL + API_DEL_TASK + str(i), headers=HEADERS)


def write_dir_api_lists(files_dir, dir_api_list):
    """
    Write the imported api list in a file for each file
    for which the apis were extracted, given the directory
    """
    
    for file, imported_api_list in dir_api_list:
        write_api_list(files_dir, file, imported_api_list)