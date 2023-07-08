# Configuration file with the needed paths
general_paths = {
    "dir": "", # working directory

    # tmp subdir for the extraction of the features for the Zhang model
    "tmp_zhang_subdir": ""
}

original_pes_paths = {
    "pes_subdir":  "", # subfolder containing the PEs to analyze
    "output_subdir": "", # output folder name
    "tasks_filename": "", # where store the successed submssion ids
    "errors_filename": "", # where store the failed submssion ids

    # where store the extracted api calls sets for comparison
    "api_calls_subdir_comparison": "", 

    # where store the extracted api calls sets for attack
    "api_calls_subdir_attack": "", 

    # where store the extracted features using Li's method
    "li_features_subdir": "",

    # where store the extracted features using Zhang's method
    "zhang_features_subdir": "",

    # where store the extracted imported api calls sets
    "imported_api_calls_subdir": ""
}

patched_pes_paths = {
    "pes_subdir": "", # subfolder containing the PEs to analyze
    "output_subdir": "",  # output folder name
    "tasks_filename": "", # where store the successed submssion ids
    "errors_filename": "", # where store the failed submssion ids

    # where store the extracted api calls sets for comparison
    "api_calls_subdir_comparison": "",

    # where store the extracted api calls sets for attack
    "api_calls_subdir_attack": "", 

    # where store the extracted features using Zhang's method
    "zhang_features_subdir": "",

    # where store the extracted imported api calls sets
    "imported_api_calls_subdir": ""
}

comparison_paths = {
    "summary_path": "", # summary written by the PE patcher
    "output_subdir": "",  # output folder name
    "comparison_results_filename": "" # where store the comparison results
}