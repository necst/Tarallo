from pwn import *
from Utils.peAnalyzer import *
from config.config_api_args import api_args

JUMP_TABLE_CODE_SIZE_x64 = 20
CODE_SIZE_x64 = 94
CODE_SIZE_ABOVE = 5
LOG = False
CODE_STRING_OFFSET = 500


def log (msg):
    if LOG is True: print(msg)


def call_iat_entry_64(api_name , start_actual_code_address, size_code_above, pe):
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

    # Reading the arguments list from the config file
    try:
        args_list = api_args[api_name]
    except:
        raise CannotInjectAPI(f"Cannot inject API {api_name} because it is not in the config file")
    # If they are at most 4, they are passed through registers RCX, RDX, R8, and R9
    # Remaining arguments get pushed on the stack in right-to-left order
    args_regs = ['rcx', 'rdx', 'r8', 'r9'] # register to pass the first 4 args
    prepare_arguments_code = ""


    # Manage the stack alignment
    if len(args_list) in [1, 2, 3, 4, 6, 8, 10, 12, 14]:
        prepare_arguments_code  += f"""
        test rsp, 0x8
        jz no_extra_push{str(size_code_above)}
        push rbx
        no_extra_push{str(size_code_above)}:
        """
    else:
        prepare_arguments_code  += f"""
        test rsp, 0x8
        jnz no_extra_push{str(size_code_above)}
        push rbx
        no_extra_push{str(size_code_above)}:
        """

    # First four arguments are passed in registers from left to right, then the rest
    # is passed on the stack from right to left
    args_list = args_list[-4:][::-1] + args_list[:-4]
    
    for i, arg in enumerate(args_list):
        
        # If the argument is 'directly passable'
        if type(arg) != bytes: 

            if i < 4:
                # Arguments in registers

                # Moving the argument to the right register     
                prepare_arguments_code += f"mov {args_regs[i]}, {arg}\n"

                if len(args_list) > 4:
                    # Push the argument to fill the stack
                    prepare_arguments_code += "push 0x0\n"
            else:
                # Arguments on the stack
                # Pushing the arguments of the API call
                prepare_arguments_code += f"push {arg}\n"
        
        
        # Strategy to deal with 'not directly passable' args
        else:
            # Compute the address of the string
            string_pointer_offset = CODE_STRING_OFFSET + strings_already_written_size

            if i < 4:
                # Arguments in registers
                # Moving the address of the string to the right register
                
                # r14 has the address of the start of the code section - call pop

                prepare_arguments_code += f"mov {args_regs[i]}, {string_pointer_offset}\n"
                prepare_arguments_code += f"add {args_regs[i]}, r14\n"

                if len(args_list) > 4:
                    # Push the argument to fill the stack
                    prepare_arguments_code += "push 0x0\n"
            else:
                # Arguments on the stack
                # Pushing the address of the string

                # r14 has the address of the start of the code section - call pop
                prepare_arguments_code += "mov rax, r14\n"
                prepare_arguments_code += f"add rax, {string_pointer_offset}\n"
                prepare_arguments_code += "push rax\n"

            # Update the size of the alread written strings
            strings_already_written_size += len(arg)

            # Update the string arguments
            string_arguments += arg


    # Compute the offset between the IAT entry to call and the position where the call
    # will be performed
    instruction_size     = 6 # assembly call instruction size
    iat_entry_address    = iat_entry_address_by_name(pe, api_name)
    current_code_address = start_actual_code_address + size_code_above # where the call is performed

    iat_code_offset      = iat_entry_address - current_code_address - instruction_size - len(asm(prepare_arguments_code))

    # Assembly code to call the desired API
    call_code  = f"call qword ptr [rip + {iat_code_offset}]\n"


    # If there are more than 4 arguments, we need to clean the stack
    if len(args_list) > 4:
        call_code += f"add rsp, {len(args_list)*8}\n"
    
    # Manage the stack alignment
    call_code  += f"""
    cmp QWORD PTR [rsp], rbx
    jne no_extra_pop{str(size_code_above)}
    add rsp, 0x8
    no_extra_pop{str(size_code_above)}:
    """

    # Logging stuff
    log(f"Iat_entry_Address {hex(iat_entry_address)}")
    log(f"actual_rip {hex(current_code_address)}")
    log(f"size_code_above {hex(size_code_above)}")
    log(f"call qword ptr [rip+{hex(iat_code_offset)}]")

    return prepare_arguments_code + call_code, len(asm(prepare_arguments_code+call_code))

# Function to compute the total size of the jumb table
def compute_jmp_table_size(iat_entries):
    jmp_table_size  = len(iat_entries) * JUMP_TABLE_CODE_SIZE_x64
    
    # Each element of this list is the number (size in bytes) of API to inject
    # for a specific original API call
    list_of_len = list(map(lambda x: len(x[1]), iat_entries))

    jmp_table_size += sum(list_of_len)

    return jmp_table_size

def make_x64_bytecode(len_hijack_functions, jump_table_address, pe, iat_entries, api_injections):
    global strings_already_written_size
    global string_arguments

    strings_already_written_size = 0
    string_arguments = b''

    # Total jump table size
    jmp_table_size = compute_jmp_table_size(iat_entries)
    
    start_actual_code_address = jump_table_address + jmp_table_size

    context.update(arch='amd64', bits=64)

    prolog = '''
    call lab
    lab:
    '''
    len_until_call = len(asm(prolog)) #offset until this first call
    
    prolog += '''
    push r15
    push r14
    push r13
    push rbx
    push rcx
    push rdx
    push r8
    push r9
    add rsp, 0x40 /* adjust the stack to pop the saved stuff */
    pop r14 /* instruction pointer */
    pop r15 /* max data size */
    pop r13 /* start data */
    pop rbx /* offset entry IAT table */
    add rbx, r14 /* IAT entry abs address */
    sub rsp, 0x60 /* restore the stack to previous position */
    '''

    pre_injected_api_calls = f'''
    movzx rax, word ptr [r13+r14] /* take the counter (#calls * entry size) from the data */
    cmp rax, r15 /* compare the counter reg with the max data size */
    je original_call
    movzx ecx, word ptr [r13+r14+2] /* take the data entry size */
    add rcx, rax /* update the counter (#calls * entry size) reg */
    mov word ptr [r13+r14], cx /* write the updated counter */
    add r13, rax /* compute start of current data entry */
    loop:
    movzx r15, byte ptr [r13+r14+4] /* read the API index from data entry */
    cmp r15, 0xff /* compare with the end of data entry symbol */
    je original_call
    '''
    # Write the code to manage the injected API calls
    injected_api_calls = ""

    size_code_above          = 88 # prolog + pre_injected_api_calls size
    for i, api in enumerate(api_injections):
        injected_api_calls += f"cmp r15, {i}\n"
        injected_api_calls += f"jne end_call_{i}\n"

        size_cmp     = 4 
        size_jne     = 2
        size_cmp_jne = size_cmp + size_jne
        
        size_code_above   += size_cmp_jne

        
        api_call_code, call_size = call_iat_entry_64(api, start_actual_code_address, size_code_above, pe)
        
        size_code_above    += call_size

        injected_api_calls += api_call_code
        injected_api_calls += f'end_call_{i}: \n'


    post_injected_api_calls = f'''
    inc r13 /* go to next data entry index */
    jmp loop
    '''

    original_call_code = '''
    original_call:
    mov rax, rbx /* IAT entry abs address */
    pop r9
    pop r8
    pop rdx
    pop rcx
    pop rbx
    pop r13
    pop r14
    pop r15
    add rsp, 0x20 /* restore the very original stack */
    jmp qword ptr [rax] /* original API call */
    '''


    # Current len of the jmp table
    injected_jump_table   = b''
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

        current_jmp_table_len += JUMP_TABLE_CODE_SIZE_x64

        # Offset of the data section start of the current entry w.r.t. the start actual code address
        start_data_offset = jmp_table_size - current_jmp_table_len + len_until_call

        current_jmp_table_len += size_of_data

        # Encoding list of integers in raw data: each byte represents an API to inject
        raw_data = b''.join([ i.to_bytes(1, 'little') for i in api_to_inject ])

        first_call_absolute_address = start_actual_code_address + len_until_call
        offset = iat_address - first_call_absolute_address
        
        log(f'*************')
        log(f'iat_address           : {hex(iat_address)}')
        log(f'offset                : {hex(offset)}')
        log(f'call_absolute_address : {hex(first_call_absolute_address+pe.OPTIONAL_HEADER.ImageBase)}')
        log(f'*************')
        
        fixed_size_code  = b''
        fixed_size_code += asm(f"push {hex(offset)}").ljust(5,b'\x90')        # offset to compute the IAT entry address
        fixed_size_code += asm(f"push {-start_data_offset}").ljust(5,b'\x90') # start of the data section - TODO neg
        fixed_size_code += asm(f"push {single_call_data_size}").ljust(5, b'\x90') # size
        fixed_size_code += asm(f"jmp $+{jmp_table_size - current_jmp_table_len + size_of_data + 5 }").ljust(5,b'\x90')
        
        # The jump table entry is appended to the jump table
        injected_jump_table += fixed_size_code + raw_data # jump table entry = fixed size code + var size raw data


    nop_sled = 'nop\n' * 115 # avoid short jumps
    injection = injected_jump_table + asm(prolog + pre_injected_api_calls + injected_api_calls + post_injected_api_calls + nop_sled + original_call_code )
    injection = injection.ljust((len_until_call+jmp_table_size+CODE_STRING_OFFSET), b'\x00')
    injection = injection + string_arguments

    return injection