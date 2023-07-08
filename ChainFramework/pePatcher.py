#!/usr/bin/python3
from fileinput import filename
import logging
import re
import pefile
from pefile import DLL_CHARACTERISTICS
from pwn import u32, context, disasm          
from Utils.peAnalyzer                 import *
from Utils.x64Bytecode                import *
from Utils.x86Bytecode                import *
from Utils.importedFunction           import *
from Utils.exceptions                 import *
from Utils.sectionAdder               import *
from pwn import *

def create_log_entry(addr, instr, bits, section, filepath):
    try:
        sec_name = section.Name.decode().replace('\x00','')
    except:
        sec_name = "InvalidName"

    s = f'''
    Filepath = {filepath}
    Bits     = {bits}
    Section  = {sec_name}
    Address  = {hex(addr)}
    {disasm(instr)}
    '''
    return s

def retrieve_call_to_imported_functions(calls_dict, pe, filepath):
    # logger = setup_logger()
    logger = logging.getLogger("custom_logger")
    if   pe.FILE_HEADER.Machine == 0x014c:
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
                if function_name not in calls_dict.keys():
                    it                        = importedFunction(function_name, bits)
                    # it.import_type            = ImportType.CALL_STUB
                    calls_dict[function_name] = it  
                    
                calls_dict[function_name].add_address(call_address, ImportType.CALL_STUB)
            else:
                # target_instr = pe.get_memory_mapped_image()[iat_entry_addr:iat_entry_addr+2]
                # Se la call va su una jump, non loggare, in caso si rompe la jump
                # if target_instr != b'\xff\x25':
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug(create_log_entry(call_address+pe.OPTIONAL_HEADER.ImageBase,
                                    m[0], bits, section, filepath
                                    ))
                pass

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
                if function_name not in calls_dict.keys():
                    it                        = importedFunction(function_name, bits)
                    # it.import_type            = ImportType.JUMP_STUB            
                    calls_dict[function_name] = it
                
                calls_dict[function_name].add_address(jump_address, ImportType.JUMP_STUB)
            else:
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug(create_log_entry(jump_address+pe.OPTIONAL_HEADER.ImageBase,
                                    m[0], bits, section, filepath
                                ))
                pass           


# Function to patch the Base Relocation Table to not break the ASLR
def base_relocation_table_patcher(force_no_rebase_addresses):
    try:
        for relocation_base in pe.DIRECTORY_ENTRY_BASERELOC:
            for entry in relocation_base.entries:
                if entry.rva in force_no_rebase_addresses:
                    entry.type = 0x0 # type 0 is ignored
    except:
        pass

# Function to check if the PE is afflicted by the pefile bug with old-style import descriptor
def check_dump_pefile():
    try:
        for delay_import in pe.DIRECTORY_ENTRY_DELAY_IMPORT:
            if delay_import.struct.grAttrs == 0 and pe.FILE_HEADER.Machine == 332:
                return True
        return False
    except:
        return False

# Patch the pefile bug with old-style import descriptor
def debug_dump_pefile(input_path):
    didata_size = 0 # initialization
    didata_rva = 0 # initialization

    # Find the didata section and take its info
    for section in pe.sections:
        if section.Name == b'.didata\x00':
            didata_size = section.Misc
            didata_rva = section.VirtualAddress

    if didata_rva != 0 and didata_size != 0:
        didata_offset_start = pe.get_offset_from_rva(didata_rva)
        didata_offset_end = pe.get_offset_from_rva(didata_rva + didata_size)

        # Take the didata section from the original PE
        pe2 = pefile.PE(input_path)
        original_didata = pe2.__data__[ didata_offset_start : didata_offset_end ]

        # Set the didata of the new PE
        pe.set_bytes_at_rva(didata_rva, original_didata)

        # Remove the pefile structures to prevent pefile changes
        structures_to_delete = []
        for structure in pe.__structures__:
            if structure.name == 'IMAGE_DELAY_IMPORT_DESCRIPTOR':
                structures_to_delete.append(structure)
            if structure.name == 'IMAGE_THUNK_DATA' and didata_offset_start <= structure.get_file_offset() < didata_offset_end:
                structures_to_delete.append(structure)
        for structure_to_delete in structures_to_delete:
                pe.__structures__.remove(structure_to_delete)

        

def patcher(input_path, output_path, hijacking_data, enumerated_injection_set_keys):
    global pe
    global dynamic_base
    global logger

    added_section = False # True if a section has been added to the PE

    logger = logging.getLogger("custom_logger")
    try:
        pe =  pefile.PE(input_path, fast_load=False)
    except: 
        raise FileNotPE

    if pe.FILE_HEADER.Machine == 0x014c:
        context.update(arch='i386', bits=32)
    elif pe.FILE_HEADER.Machine == 0x8664:
        context.update(arch='amd64', bits=64)
    else:
        raise MachineTypeError ('Impossible to determine the machine type')

    # True if Dynamic Base flag is set
    try:
        dynamic_base = pe.OPTIONAL_HEADER.IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE
    except:
        raise NoDynamicBaseFlag ('Impossible to determine if Dynamic Base flag is set')
    
    # True if Relocation Table flag is set
    try:
        relocs_stripped = pe.FILE_HEADER.IMAGE_FILE_RELOCS_STRIPPED
    except:
        raise NoRelocationsStrippedFlag ('Impossible to determine if Relocations Stripped flag is set')

    x_sections_dict                     = retrieve_x_sections(pe)

    # Check if there are executable sections
    if not x_sections_dict:
        raise NoXSectionFound (f'No executable section found in: {input_path}')

    target_section                    = max(x_sections_dict, key=x_sections_dict.get)
    target_section_padding            = x_sections_dict.get(target_section)

    calls_to_imported_functions = {}
    retrieve_call_to_imported_functions(calls_to_imported_functions, pe, input_path)

    writable_zone = {}
    iat_addresses = {} # dict of the hijacked API calls IAT addresses
    calls_patched = False # True if at least one call has been patched
    for element in hijacking_data: 
        function_name = element[0]  # function name
        iat_address = iat_entry_address_by_name(pe, function_name)
        if iat_address is None:
            # Cannot be hijacked, not found in the IAT
            hijacking_data = list(filter(lambda x : x[0] != function_name, hijacking_data))
            continue
        if function_name not in calls_to_imported_functions:
            # Cannot be hijacked, no calls to the api to hijack
            hijacking_data = list(filter(lambda x : x[0] != function_name, hijacking_data))
            continue
        else:
            iat_addresses[function_name] = iat_address
            calls_patched = True
    if len(hijacking_data) == 0:
        raise NoImportToHijack (f'No imported api to hijack in: {input_path}')
    elif not calls_patched:
        raise ApiCallsNeverCalled (f'No actually called apis to hijack in: {input_path}')

    api_injections = [api.encode() for api in enumerated_injection_set_keys]
    
    if context.arch == 'i386':

        # Check if the api that we want to inject are actually imported
        for api in api_injections:
            iat_address = iat_entry_address_by_name(pe, api)
            if iat_address is None:
                api_injections = list(filter(lambda x : x != api, api_injections))
        if len(api_injections) != 0:
            # We need to check if we have enough space in the writable zone
            # before doing all the patching. If not so, we create a new section
            length = len(hijacking_data) * JUMP_TABLE_CODE_SIZE_x86 + CODE_SIZE_x86
            if True: #target_section_padding < length:
                pe_data, jump_table_address = add_section()
                pe = pefile.PE(data=pe_data)
                added_section = True
            else:
                target_virtual_size               = target_section.Misc_VirtualSize  
                target_section_RVA                = target_section.VirtualAddress # this is RVA
                writable_zone['start'] = target_section_RVA+target_virtual_size     
                writable_zone['end'] = target_section_RVA+target_virtual_size+target_section_padding 
                jump_table_address        = writable_zone["start"]

            iat_entries, force_no_rebase_addresses = patch_calls(hijacking_data, calls_to_imported_functions, jump_table_address, iat_addresses, input_path)
            injected_code = make_x86_bytecode(len(hijacking_data), jump_table_address, pe, iat_entries, api_injections)
        else:
            raise NoApiCallsToInject(f'No api call to inject in: {input_path}')
    elif context.arch == 'amd64':

        # Check if the api that we want to inject are actually imported
        for api in api_injections:
            iat_address = iat_entry_address_by_name(pe, api)
            if iat_address is None:
                api_injections = list(filter(lambda x : x != api, api_injections))
        if len(api_injections) != 0:
            # We need to check if we have enough space in the writable zone
            # before doing all the patching. If not so, we create a new section
            length = len(hijacking_data) * JUMP_TABLE_CODE_SIZE_x64 + CODE_SIZE_x64
            if True: #target_section_padding < length:
                pe_data, jump_table_address = add_section()
                pe = pefile.PE(data=pe_data)
                added_section = True
            else:
                target_virtual_size               = target_section.Misc_VirtualSize  
                target_section_RVA                = target_section.VirtualAddress # this is RVA  
                writable_zone['start'] = target_section_RVA+target_virtual_size      
                writable_zone['end'] = target_section_RVA+target_virtual_size+target_section_padding 
                
                jump_table_address        =  writable_zone["start"]

            iat_entries, force_no_rebase_addresses = patch_calls(hijacking_data, calls_to_imported_functions, jump_table_address, iat_addresses, input_path)
            injected_code = make_x64_bytecode(len(hijacking_data), jump_table_address, pe, iat_entries, api_injections)
        else:
            raise NoApiCallsToInject(f'No api call to inject in: {input_path}')
    else:
        raise MachineTypeError ('Impossible to determine the machine type')


    writable_zone["start"] = jump_table_address


    # Write the injected code in the PE
    pe.set_bytes_at_rva(writable_zone['start'], injected_code)

    # Check if is needed to patch the Base Relocation Table to avoid problems with the ASLR
    if dynamic_base and not relocs_stripped:    
        base_relocation_table_patcher(force_no_rebase_addresses) # patch the base relocation table
    
    # Solve the pefile problem with old-style import descriptor
    if check_dump_pefile():
        debug_dump_pefile(input_path)

    pe.write(output_path) # save the final modified PE

        
    """for relocation_base in pe.DIRECTORY_ENTRY_BASERELOC:
        for entry in relocation_base.entries:
            print(hex(entry.rva))
    for section in pe.sections:
        print (section.Name, hex(section.VirtualAddress),hex(section.Misc_VirtualSize), section.SizeOfRawData )"""
    return hijacking_data, api_injections, added_section


def add_section():
    start_first_section = pe.sections[0].section_min_addr # first section beginning

    if 'DIRECTORY_ENTRY_BOUND_IMPORT' in dir(pe):
        # Start of the Entry Bound Import Table
        start_entry_bound_import = pe.DIRECTORY_ENTRY_BOUND_IMPORT[0].struct.dump_dict()['TimeDateStamp']['FileOffset']

        # Size of the Entry Bound Import Table
        size_bound_import = pe.OPTIONAL_HEADER.DATA_DIRECTORY[11].dump_dict()['Size']['Value']

        end_headers = start_entry_bound_import + size_bound_import
    else:
        end_headers = pe.sections[-1].get_file_offset() + 40
    
    # Check if we have enough space for a new header
    available_space = pe.get_physical_by_rva(start_first_section) - pe.get_physical_by_rva(end_headers)
    if available_space < 40:
        raise InsuffientPadding (f'ERROR - cannot be hijacked, no sufficient padding')
    

    return addSection(pe, b'.added', 0x400)

def patch_calls(hijack_functions, calls_to_imported_functions, jump_table_address, iat_addresses, input_path):
    iat_entries = []
    force_no_rebase_addresses = []
    if context.arch == 'i386':
        JUMP_TABLE_ROW_SIZE = JUMP_TABLE_ROW_SIZE_32
    elif context.arch == 'amd64':
        JUMP_TABLE_ROW_SIZE = JUMP_TABLE_ROW_SIZE_64
    else:
        raise MachineTypeError ('Impossible to determine the machine type in importedFunction')
    
    previous_jmp_table_row_size = 0
    
    for element in hijack_functions: 
        function_name = element[0]  # function name
        api_data = element[1]  # raw data
        single_call_data_size = element[2] # size of single call raw data
        try:
            calls_to_imported_functions[function_name].patch_function(jump_table_address, previous_jmp_table_row_size, pe)
            
            # If ASLR is on, save the call stub addresses to patch the correspondant Base Relocation
            # Table entry
            if dynamic_base:
                for addr, imp_type  in calls_to_imported_functions[function_name]._call_addresses:
                    # The Base Relocation Table contains the addresses of the absolute references
                    # in the code, hence we need to take the addresses of the calls arguments
                    force_no_rebase_addresses.append(addr+2)

            previous_jmp_table_row_size += JUMP_TABLE_ROW_SIZE + len(api_data)
            iat_entries.append([iat_addresses[function_name], api_data, single_call_data_size])
        
        except Exception as e:
            logger.warning('patch_function failed!\n' + input_path +'\n'+ str(e))
            # API imported but never called
            pass
    return iat_entries, force_no_rebase_addresses