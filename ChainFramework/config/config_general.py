# General configurations
general = {
    "n_pools": 40, # number of pools to use for the pathing operations
    "limit_pes_to_patch": 300, # limit in the number of PEs to patch
    "signatures": [b'Installer integrity check has failed.', b'The file "%modname%" seems to be corrupt!', \
        b'I\x00n\x00s\x00t\x00a\x00l\x00l\x00e\x00r\x00 \x00i\x00n\x00t\x00e\x00g\x00r\x00i\x00t\x00y\x00 \x00c\x00h\x00e\x00c\x00k\x00 \x00h\x00a\x00s\x00 \x00f\x00a\x00i\x00l\x00e\x00d',\
        b'Setup program invalid or damaged', \
        b'the setup files are corrupted', \
        b'The installer you are trying to use is corrupted or incomplete'], # signature to match to find the PEs to ignore
    # Fake hijacking fata for testing purposes
    "hijacking_data" : [
        [b'memset', b'\x00\x00\x03\x00\x00\x01\xff', 3], [b'memcpy', b'\x00\x00\x03\x00\x00\x01\xff', 3], 
        [b'puts', b'\x00\x00\x03\x00\x00\x01\xff', 3], [b'scanf', b'\x00\x00\x03\x00\x00\x01\xff', 3],
        [b'GetLastError', b'\x00\x00\x03\x00\x00\x01\xff', 3], [b'GetProcAddress', b'\x00\x00\x03\x00\x00\x01\xff', 3], 
        [b'WriteFile', b'\x00\x00\x03\x00\x00\x01\xff', 3], [b'GetFileType', b'\x00\x00\x03\x00\x00\x01\xff', 3],
        [b'CloseHandle', b'\x00\x00\x03\x00\x00\x01\xff', 3], [b'FreeLibrary', b'\x00\x00\x03\x00\x00\x01\xff', 3],
        [b'UnhandledExceptionFilter', b'\x00\x00\x03\x00\x00\x01\xff', 3], [b'_CorExeMain', b'\x00\x00\x03\x00\x00\x01\xff', 3]
    ] 
}

