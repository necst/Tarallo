# Configuration file with the needed paths
general = {
    "n_pools": 40, # number of pools to use
    "n_subm": 1, # number of times PE exe is executed by Cuckoo
    "webserver_url": "http://localhost:8090/tasks/report/", # URL of the webserver
    "header": {"Authorization": "REDACTED"}, # header for the webserver


    # API call for wich we ignore the arguments in the comparison
    "ignore_arguments": ["NtFreeVirtualMemory", "NtTerminateProcess", "NtMapViewOfSection", "NtAllocateVirtualMemory", \
        "NtClose", "LdrGetProcedureAddress", "FindResourceExW", "LoadResource", "SizeofResource", "RegEnumKeyExW", \
        "NtUnmapViewOfSection", "LdrGetDllHandle", "NtDuplicateObject", "__exception__", "NtResumeThread", "CreateThread", \
        "ReadProcessMemory", "NtQueryValueKey", "RegOpenKeyExW", "NtCreateSection", "LoadStringW", "NtProtectVirtualMemory", \
        "NtCreateThreadEx"], 
    "ignored_arguments": ["section_handle", "process_identifier", "process_handle", "handle", "module_address", "module_name",\
        "function_address", "resource_handle", "module_handle", "base_handle", "key_handle", "object_handle", "provider_handle",\
        "file_handle", "thread_handle", "thread_identifier", "base_address", "thread_name", "free_bytes_available",\
        "source_process_identifier", "source_handle", "target_process_identifier", "target_process_handle", "newfilepath",\
        "target_handle", "source_process_handle", "buffer", "registers", "size", "parameter", "pointer", "oldfilepath",\
        "region_size", "key_name", "mutant_handle", "uuid", "socket", "crypto_handle", "input_buffer", "total_number_of_bytes",\
        "output_buffer", "value", "port", "x", "y", "s", "string", "regkey", "regkey_r", "skipped", "mutant_name",\
        "section_name", "window_handle", "service_handle", "service_manager_handle", "process_name", "process_identifier",\
        "hook_identifier", "callback_function", "filepath", "reg_type", "milliseconds", "hook_handle", "filepath_r", \
        "newfilepath_r", "oldfilepath_r", "device_handle", "hash_handle", "token_handle", "owner_handle", "total_number_of_free_bytes",
        "application_name", "offset", "buf", "hostname", "view_size", "file_size_low", "id", "length", "dirpath_r", "dirpath", \
        "current_directory", "command_line", "filename", "disposition", "cert_store", "parent_window_handle", "instance_handle", \
        "snapshot_handle", "directory_handle", "exception", "KeyHandle", "stacktrace", "crypto_export_handle", "url", "parameters", \
        "shortpath", "section_offset", "number_of_free_clusters", "job_handle", "tid", "parent_hwnd", "child_after_hwnd", "post_data"],
    # Some PEs cannot be compared (e.g. installer with integrity cheks)
    "ignored_pes": []
}