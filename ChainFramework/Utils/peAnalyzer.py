#!/usr/bin/python3
import os
import pefile
from tqdm import tqdm
from config.config_general import general
from multiprocessing import Pool
import shutil
from .exceptions import *
import re
from pwn import u32

def print_info(pe):
    """ Prints useful information on the loaded PE

    Args:
        pe (pefile.PE() object): a PE object created by pefile.PE()
    """
    for section in pe.sections:
        print(section.Name, hex(section.VirtualAddress), hex(section.Misc_VirtualSize), section.SizeOfRawData )
    
    for entry in pe.DIRECTORY_ENTRY_IMPORT:
        print(entry.dll)
        for imp in entry.imports:
            print('\t', hex(imp.address), imp.name, hex(imp.thunk_rva), hex(imp.thunk_offset))


def iat_entry_address_by_name(pe, function_name):
    """ Finds the address of the IAT entry within a pefile, given a function name

    Args:
        pefile (pe object): PE object in where to search
        function_name (string): Name of the function that we want to find in the IAT

    Raises:
        Exception: Raise exception just to see quickly if it does not work

    Returns:
        int: returns the address where the IAT entry is
    """
    try:
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            for imp in entry.imports:
                if(imp.name == function_name): 
                    return imp.address - pe.OPTIONAL_HEADER.ImageBase
        if 'DIRECTORY_ENTRY_DELAY_IMPORT' in dir(pe):
            for entry in pe.DIRECTORY_ENTRY_DELAY_IMPORT:
                for imp in entry.imports:
                    if(imp.name == function_name): 
                        return imp.address - pe.OPTIONAL_HEADER.ImageBase
    except:
        pass
    return None


def iat_entry_name_by_address(pe, address):
    """ Finds the name of the IAT entry within a pefile imports, given a its address

    Args:
        pefile (pe object): PE object in where to search
        address  (address): Address without the imagebase

    Raises:
        Exception: Raise exception just to see quickly if it does not work

    Returns:
        The name found or None
    """
    try:
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            for imp in entry.imports:
                if imp.address - pe.OPTIONAL_HEADER.ImageBase == address:
                    return imp.name
        if 'DIRECTORY_ENTRY_DELAY_IMPORT' in dir(pe):
            for entry in pe.DIRECTORY_ENTRY_DELAY_IMPORT:
                for imp in entry.imports:
                    if imp.address - pe.OPTIONAL_HEADER.ImageBase == address:
                        return imp.name
    except:
        pass
    
    return None
            

def compute_padding_of_section(section, file_alignment):
    """Computes the length of the padding of a section, if any
    """    
    
    raw_data_size = section.SizeOfRawData       # Size of the section on disk 
    virtual_size  = section.Misc_VirtualSize    # Size of the section when it will be loaded in memory

    if(raw_data_size > virtual_size):
        """
        It means that the section on the file( on disk) is bigger than "necessary", due to
        the file alignment that every section must respect.
        """
        padding_dimension = file_alignment - (virtual_size % file_alignment)
    else:
        padding_dimension = 0
        
    return padding_dimension
    

def retrieve_x_sections(pe):
    """Retrieve all the executable sections

    Args:
        pe (pe file): the pe file from which we extract the executable sections

    Returns:
        dict: dict of (executable section:available padding)
    """
    file_alignment = pe.OPTIONAL_HEADER.FileAlignment
    x_sections = {}
    for s in pe.sections:
        if s.IMAGE_SCN_MEM_EXECUTE == 1:
            x_sections.update({s : compute_padding_of_section(s,file_alignment)})
    return x_sections


def retrieve_w_sections(pe):
    """Retrieve all the writable sections

    Args:
        pe (pe file): the pe file from which we extract the executable sections

    Returns:
        dict: dict of (writable section:available padding)
    """
    file_alignment = pe.OPTIONAL_HEADER.FileAlignment
    w_sections = {}
    for s in pe.sections:
        if s.IMAGE_SCN_MEM_WRITE == 1:
            w_sections.update({s : compute_padding_of_section(s,file_alignment)})
    return w_sections


def info_on_padding_available(sections_dict):
    """prints the padding space available for each eXecutable section on a pe file

    Args:
        pe (pe file): the pe file
    """
    
    for section, padding in sections_dict.items():
        # padd = compute_padding_of_section(s,pe.OPTIONAL_HEADER.FileAlignment)
        print(f"In section { section.Name.decode() } we have {padding} bytes available")
        print()


def write_api_list(output_folder, file, imported_api_list, suffix = '_imported_apis'):
    """Write the imported api list in a file"""
    
    # This could write the same API multiple times
    with open(os.path.join(output_folder, file + suffix), 'wb') as f:
        for api in set(imported_api_list):
            if api != None:
                f.write(api + b'\n')


def retrieve_imported_api_list(pe):
    """Returns a list of all the imported API

    Returns:
        list: list of all the imported API
    """
    api_list = []
    if 'DIRECTORY_ENTRY_IMPORT' in dir(pe):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            for imp in entry.imports:
                api_list.append(imp.name)
    
    if 'DIRECTORY_ENTRY_DELAY_IMPORT' in dir(pe):
        for entry in pe.DIRECTORY_ENTRY_DELAY_IMPORT:
            for imp in entry.imports:
                api_list.append(imp.name)
    return api_list

def retrieve_call_to_imported_functions(pe):
    """
    This function will retrieve all the calls to imported functions
    and will store them in a set. This is done by looking for the CALL
    stubs in the executable sections of the PE file.
    Input: pefile object
    Output: set of api_names
    """
    
    calls_set = set([])
    if  pe.FILE_HEADER.Machine == 0x014c:
        bits=32
    elif pe.FILE_HEADER.Machine == 0x8664:
        bits=64
    else:
        raise MachineTypeError
    
    reg_call_stub = rb'(\xff\x15.{4})'
    reg_jump_stub = rb'(\xff\x25.{4})'
    
    exec_sections = [s for s in pe.sections if s.IMAGE_SCN_MEM_EXECUTE == 1]

    for section in exec_sections:
        start_section   = section.VirtualAddress
        end_section     = start_section + section.Misc_VirtualSize
        bytes_to_search = pe.get_memory_mapped_image()[start_section:end_section]

        # Searching for the CALL stubs
        for m in re.finditer(reg_call_stub, bytes_to_search):
            #FF 15 .. .. .. .. 
            call_address = m.start() + section.VirtualAddress
            operand      = u32(m[0][2:])  
            
            if bits == 32:
                # Direct call example: 
                # 0: ff 15 10 20 40 40
                # 0: call   DWORD PTR ds:0x40402010
                iat_entry_addr   = operand - pe.OPTIONAL_HEADER.ImageBase
                function_name    = iat_entry_name_by_address(pe, iat_entry_addr) 
            elif bits == 64:
                # Call rip-offset example
                # 0: ff 15 10 20 40 40
                # 0: call   QWORD PTR [rip+0x40402010]   
                # Actual call to 0x40402016 <-- notice the +6
                operand      = u32(m[0][2:], sign='signed')  

                len_instr      = 6
                offset         = operand      + len_instr 
                iat_entry_addr = call_address + offset  #  - pe.OPTIONAL_HEADER.ImageBase
                function_name  = iat_entry_name_by_address(pe, iat_entry_addr) 
            
            if function_name is not None:
                calls_set.add(function_name)

        # Searching for the JUMP stubs
        for m in re.finditer(reg_jump_stub, bytes_to_search):
            
            jump_address = m.start() + section.VirtualAddress
            operand      = u32(m[0][2:]) 
            
            if bits == 32:
                # Direct call example: 
                # 0: ff 25 10 20 40 40
                # 0: jmp    DWORD PTR ds:0x40402010
                iat_entry_addr   = operand - pe.OPTIONAL_HEADER.ImageBase
                function_name = iat_entry_name_by_address(pe, iat_entry_addr) 
            elif bits == 64:
                # Call rip-offset example
                # 0: ff 15 10 20 40 40
                # 0: jmp    QWORD PTR [rip+0x40402010]
                # Actual jump to 0x40402016 <-- notice the +6
                operand      = u32(m[0][2:], sign='signed')  

                len_instr      = 6
                offset         = operand      + len_instr 
                iat_entry_addr = jump_address + offset    #- pe.OPTIONAL_HEADER.ImageBase
                function_name  = iat_entry_name_by_address(pe, iat_entry_addr) 
            
            if function_name is not None:
                calls_set.add(function_name)
    
    return calls_set

def retrieve_imported_api_list_from_dir(input_dir, files_list, output_dir):
    global input_folder
    global output_folder
    """Returns a list of all the imported API from a directory

    Returns:
        list: list of all the imported API
    """

    input_folder = input_dir
    output_folder = output_dir

    # If the output folder alredy exists, delete it
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    with Pool(general["n_pools"]) as p:
        list(tqdm(p.imap_unordered(retrieve_and_write_imported_api_list_from_dir, files_list), total=len(files_list)))
    

def retrieve_and_write_imported_api_list_from_dir(file):
    filepath = os.path.join(input_folder, file)
    try:
        pe = pefile.PE(filepath)
    except:
        return
    
    imported_api_list = retrieve_imported_api_list(pe)
    
    try:
        called_api_list   = retrieve_call_to_imported_functions(pe)
    except:
        called_api_list = []
    write_api_list(output_folder, file, imported_api_list, suffix='_imported_apis')
    write_api_list(output_folder, file, called_api_list, suffix='_called_apis')