from pwn import *
from Utils.peAnalyzer import *
from config.config_api_args import api_args

CODE_SIZE_x86   = 76
CODE_SIZE_ABOVE = 5
JUMP_TABLE_CODE_SIZE_x86 = 20
CODE_STRING_OFFSET = 500 # Offset of the first string after the code section


def is_debug_on():
    logger = logging.getLogger('custom_logger')
    return logger.getEffectiveLevel() <= logging.DEBUG
def is_log_info_on():
    logger = logging.getLogger('custom_logger')
    return logger.getEffectiveLevel() <= logging.INFO 


def call_iat_entry_32(iat_entry_name ,len_until_call, start_actual_code_address, pe):
    """
    Function to create the assembly code to call an API using its IAT entry
    Input: 
    - name of the API to call
    - address where our code will be injected
    - size of the code that preceed the current API call to build
    - peFile to modify
    """
    global strings_already_written_size
    global string_arguments

    first_call_absolute_address = start_actual_code_address + len_until_call
    iat_entry_address           = iat_entry_address_by_name(pe, iat_entry_name)
    eip_offset                  = iat_entry_address - first_call_absolute_address

    try:
        args_list = api_args[iat_entry_name]
    except:
        raise CannotInjectAPI(f"Cannot inject API {iat_entry_name} because it is not in the config file")
    
    push_arguments = ""
    
    # Reading the arguments list from the config file
    for arg in args_list:
        # If the argument is 'directly pushable'
        if type(arg) != bytes: 
            # Pushing the arguments of the API call
            push_arguments += f"push {arg}\n"
        
        # Strategy to deal with 'not directly pushable' args
        else:
            # Compute the address of the string
            string_pointer_offset = CODE_STRING_OFFSET + strings_already_written_size

            # Pushing the address of the string
            push_arguments += "mov edx, ebx\n"
            push_arguments += f"add edx, {string_pointer_offset}\n"
            push_arguments += "push edx\n"

            # Update the size of the alread written strings
            strings_already_written_size += len(arg)

            # Update the string arguments
            string_arguments += arg

    call_code  = ""
    call_code += f"""
    push {eip_offset}
    pop ecx
    add ecx, ebx
    call dword ptr [ecx]
    """

    return push_arguments + call_code


# Function to compute the total size of the jumb table
def compute_jmp_table_size(iat_entries):
    jmp_table_size  = len(iat_entries) * JUMP_TABLE_CODE_SIZE_x86
    
    # Each element of this list is the number (size in bytes) of API to inject
    # for a specific original API call
    list_of_len = list(map(lambda x: len(x[1]), iat_entries))

    jmp_table_size += sum(list_of_len)

    return jmp_table_size

def make_x86_bytecode(len_hijack_functions, jump_table_address, pe, iat_entries, api_injections):
    global strings_already_written_size
    global string_arguments

    strings_already_written_size = 0
    string_arguments = b''
    
    logger = logging.getLogger('custom_logger')
    # Total jump table size
    jmp_table_size = compute_jmp_table_size(iat_entries)

    # The address of the first instruction of the injected code: it's the call to retrieve the IP
    start_actual_code_address = jump_table_address + jmp_table_size

    # Force 32 bit settings
    context.update(arch='i386', bits=32)

    # Code to write and execute BEFORE the original API call
    prolog = """
    call lab
    lab:
    """
    len_until_call = len(asm(prolog)) #offset until this first call
    prolog += """
    push ebx
    push edi
    push esi
    push ebp
    add esp, 0x10 /* adjust the stack to pop the saved stuff */
    pop ebx /* instruction pointer */
    pop esi /* max data size */
    pop ebp /* start data */
    pop edi /* offset entry IAT table */
    add edi, ebx /* IAT entry abs address */
    sub esp, 0x20 /* restore the stack to previous position */
    """

    # TODO 
    pre_injected_api_calls = f"""
    movzx eax, word ptr [ebp+ebx] /* take the counter (#calls * entry size) from the data */
    cmp eax, esi /* compare the counter reg with the max data size */
    je original_call
    movzx ecx, word ptr [ebp+ebx+2] /* take the data entry size */
    add ecx, eax /* update the counter (#calls * entry size) reg */
    mov word ptr [ebp+ebx], cx /* write the updated counter */
    add ebp, eax /* compute start of current data entry */
    loop:
    movzx esi, byte ptr [ebp+ebx+4] /* read the API index from data entry */
    cmp esi, 0xff /* compare with the end of data entry symbol */
    je original_call
    """

    # Write the code to manage the injected API calls
    injected_api_calls = ""
    for i, api in enumerate(api_injections):
        api_call_code = call_iat_entry_32(api, len_until_call, start_actual_code_address, pe)

        injected_api_calls += f"cmp esi, {i}\n"
        injected_api_calls += f"jne $+{len(asm(api_call_code)) + 2}\n"
        injected_api_calls += api_call_code

    # TODO 
    post_injected_api_calls = f"""
    inc ebp /* go to next data entry index */ 
    jmp loop
    """

    # Call to the original API call (variadic function case)
    original_call_code = """
    original_call:
    mov eax, edi /* IAT entry abs address */
    pop ebp
    pop esi
    pop edi
    pop ebx
    add esp, 0x10 /* restore the very original stack */
    jmp dword ptr [eax] /* original API call */
    """


    if is_log_info_on(): 
        t  = '\n******JUMP TABLE*******'
    
    # Log stuff
    l  = ''

    injected_jump_table = b''

    # Current len of the jmp table
    current_jmp_table_len = 0

    
    for i,iat_entry in enumerate(iat_entries):
        
        # Address of the API call IAT entry
        iat_address   = iat_entry[0] 
        
        # List of integers (one byte): each int represents an API to inject
        api_to_inject = iat_entry[1] 
        
        # Size of each entry
        single_call_data_size = iat_entry[2]
        

        # Number (size in bytes) of API to inject
        size_of_data = len(api_to_inject)

        current_jmp_table_len += JUMP_TABLE_CODE_SIZE_x86

        # Offset of the data section start of the current entry w.r.t. the start actual code address
        start_data_offset = jmp_table_size - current_jmp_table_len + len_until_call

        current_jmp_table_len += size_of_data

        # Encoding list of integers in raw data: each byte represents an API to inject
        raw_data = b''.join([ i.to_bytes(1, 'little') for i in api_to_inject ])

        first_call_absolute_address = start_actual_code_address + len_until_call
        offset                      = iat_address - first_call_absolute_address
        
        fixed_size_code         = b''
        fixed_size_code        += asm(f"push {hex(offset)}").ljust(5,b'\x90')        # offset to compute the IAT entry address
        fixed_size_code        += asm(f"push {-start_data_offset}").ljust(5,b'\x90') # start of the data section 
        fixed_size_code        += asm(f"push {single_call_data_size}").ljust(5, b'\x90')  # size
        fixed_size_code        += asm(f"jmp $+{jmp_table_size - current_jmp_table_len + size_of_data + 5 }").ljust(5,b'\x90')
        

        # The jump table entry is appended to the jump table
        injected_jump_table += fixed_size_code + raw_data # jump table entry = fixed size code + var size raw data

        
        ### Logging
        if is_debug_on():
            name = iat_entry_name_by_address(pe, iat_address)
            l += f'''
            {i}) {name}
            start_actual_code_address   : {hex(start_actual_code_address)}({hex(start_actual_code_address+ib)})
            len_until_call              : {len_until_call}
            first_call_absolute_address : start_actual_code_address + len_until_call
            first_call_absolute_address : {hex(start_actual_code_address)} + {len_until_call}
            first_call_absolute_address : {hex(first_call_absolute_address)}({hex(first_call_absolute_address+ib)})
            iat_address                 : {hex(iat_address)}({hex(iat_address+ib)})'
            offset                      : {offset}({hex(offset)})'
            offset = iat_address - first_call_absolute_address
            offset = {hex(iat_address)} - {hex(first_call_absolute_address)}
            
            --> Check that first_call_absolute_address+offset = addr of IAT entry {name}
            
            where_to_jump = $+size_j_table - (row_size*i + code_size_above)'
            where_to_jump = $+{jmp_table_size} - ({JUMP_TABLE_CODE_SIZE_x86}*{i} + {CODE_SIZE_ABOVE})'
            where_to_jump = $+{hex(jmp_table_size - (JUMP_TABLE_CODE_SIZE_x86*i + CODE_SIZE_ABOVE))}'
            push   {hex(offset)}'
            jmp $+ {hex(jmp_table_size - (JUMP_TABLE_CODE_SIZE_x86*i + CODE_SIZE_ABOVE))}'
            
            --> Check that the jump jumps always to the start of the jump table
            '''
        if is_log_info_on():
            name = iat_entry_name_by_address(pe, iat_address)
            t += f'''
            {i}) {name}
            -------------------
            push {hex(offset)}'
            jmp  $+{hex(jmp_table_size - (JUMP_TABLE_CODE_SIZE_x86*i + CODE_SIZE_ABOVE))}'
            -------------------
            '''
        ###

    if is_debug_on():
        logger.debug(l)

    if is_log_info_on(): 
        t += '\n******   END    *******'
        logger.info(t)
        
    injection = injected_jump_table + asm(prolog + pre_injected_api_calls + injected_api_calls + post_injected_api_calls + original_call_code )
    injection = injection.ljust((len_until_call+jmp_table_size+CODE_STRING_OFFSET), b'\x00')
    injection = injection + string_arguments

    return injection
