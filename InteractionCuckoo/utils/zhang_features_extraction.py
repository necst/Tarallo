import os
import shutil
import json
import zipfile
from multiprocessing import Pool
from utils.Cuckoo2DMDS import Cuckoo2DMDS
import requests
from config.config_general import general
from tqdm import tqdm
import datetime

HEADERS = {"Authorization": "REDACTED"}
URL = "http://localhost:8090/tasks/report/"

# Open and read the file with the task ids written by the submitter
def read_task_ids(tasks_file):
    try:
        with open(tasks_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(e)


# Helper function to write a single json report to the tmp directory
def helper_move_rename_report(task_id):
    current_url = URL + str(task_id)
    try:
        # Get request to cuckoo's REST API
        r = requests.get(current_url, headers=HEADERS)
        json_response = r.json()
    except:
        return
    
    # Get the name of the PE file
    pe_name = json_response["target"]["file"]["name"] 

    # Write the report to the output directory
    with open(os.path.join(tmp_folder, pe_name +'.json'), 'w') as f:
        json.dump(json_response, f)


# Function to write the json reports to the tmp directory
def move_rename_reports(task_ids):
    print("Moving and renaming the reports...")

    with Pool(general["n_pools"]) as p:
        list(tqdm(p.imap_unordered(helper_move_rename_report, task_ids), total=len(task_ids)))

# Function to extract the features from the reports
def extract_features_from_reports(input_path, output_path):
    print("Extracting features from reports...")
    max_len = 1000

    files = [f[:-5] for f in os.listdir(input_path) if f[-5:] == '.json']
    
    p = Pool(general['n_pools'])
    for f in tqdm(files):
        task = Cuckoo2DMDS(f, input_path, output_path, max_len, f)
        p.apply_async(task.run())
    p.close()
    p.join()


# Function to zip the extracted features
def zip_files(input_path, zip_name, suffix):

    files = [os.path.join(input_path,f) for f in os.listdir(input_path) if f[-4:] == suffix]
    
    zip_path = os.path.join(input_path, zip_name)

    with zipfile.ZipFile(zip_path, mode="w") as archive:
        for f in files:
            archive.write(f, f.split('/')[-1])
    

# Main function
def zhang_features_extraction(tasks_file, tmp_path, output_path):
    global tmp_folder

    tmp_folder = tmp_path

    # Read the task ids from file
    task_ids = read_task_ids(tasks_file)

    # Copy the reports to the Zhang's tmp folder
    move_rename_reports(task_ids)

    # Runs the feature extraction of Zhang Paper on those reports
    extract_features_from_reports(tmp_path, output_path)

    # Zips .npy files
    now = datetime.datetime.now() # get the current date and time
    filename = "reports" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".zip"
    zip_files(output_path, filename, suffix='.npy')

    # Delete the tmp folder
    shutil.rmtree(tmp_path)