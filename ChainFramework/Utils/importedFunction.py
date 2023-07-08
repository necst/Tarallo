from enum import Enum, auto
import logging

from .x64Bytecode import JUMP_TABLE_CODE_SIZE_x64 as JUMP_TABLE_ROW_SIZE_64
from .x86Bytecode import JUMP_TABLE_CODE_SIZE_x86 as JUMP_TABLE_ROW_SIZE_32
from .exceptions import MachineTypeError
from pwn import asm, p32

LOG = False


def is_debug_on():
    logger = logging.getLogger('custom_logger')
    return logger.getEffectiveLevel() <= logging.DEBUG
def is_log_info_on():
    logger = logging.getLogger('custom_logger')
    return logger.getEffectiveLevel() <= logging.INFO


class ImportType(Enum):
    UNDEFINED = auto()
    JUMP_STUB = auto()
    CALL_STUB = auto()


class importedFunction:
    def __init__(self, name, bits ):
        self._name = name
        self._size = 6
        self._call_addresses = []
        self.bits = bits
        # self._import_type = ImportType.UNDEFINED
    
    @property
    def name(self):
        return self._name
    
    
    @property
    def call_addresses(self):
        return self._call_addresses
    
    @call_addresses.setter
    def call_addresses(self, dict_calls):
        self._call_addresses = dict_calls
        
    # def add_call_address(self, address):
    #     self._call_addresses.append(address)
    
    
    @property
    def import_type(self):
        return self._import_type
    
    @import_type.setter
    def import_type(self, it):
        self._import_type = it
    
    def add_address(self, address, import_type):
        self._call_addresses.append((address, import_type))
        
    
    def dump(self, pe):
        i = 0
        ib = pe.OPTIONAL_HEADER.ImageBase
        d = f'''
        Name               = {self._name}
        occurrences        = [
        '''

        for addr, imp_type in self._call_addresses:
            d+=f'''
            {i}) {hex(addr)}({hex(addr+ib)}) {imp_type}
            '''
            i += 1
        d +='''
        ]

        '''
        return d

    def patch_function(self, jump_table_address, previous_jmp_table_row_size, pe):
        ib     = pe.OPTIONAL_HEADER.ImageBase
        logger = logging.getLogger('custom_logger')
        len_instruction = 6 # We search for ff 15/ff 25 and 4 bytes operand

        if len(self._call_addresses) == 0:
            return
        
        if is_debug_on():
            l = f"Patching function {self._name}\n"

        # if self._import_type == ImportType.CALL_STUB:
            # if is_debug_on():
            #     l += f"Logging {self._name}: it has {len(self._call_addresses)} occurrences\n"

        if self.bits == 32:
            JUMP_TABLE_ROW_SIZE = JUMP_TABLE_ROW_SIZE_32
        elif self.bits == 64:
            JUMP_TABLE_ROW_SIZE = JUMP_TABLE_ROW_SIZE_64
        else:
            raise MachineTypeError ('Impossible to determine the machine type in importedFunction')


        for addr, imp_type in self._call_addresses:
            # we want to jump in the relative row of the jump table
            if imp_type == ImportType.CALL_STUB:
                where_to_jump           = jump_table_address + previous_jmp_table_row_size
                call_instruction_size   = 5
                address_call_stub       = addr                                         
                displacement            = where_to_jump - address_call_stub - call_instruction_size 
                new_code                = b'\xe8' + p32(displacement)
                new_code                = new_code.ljust(len_instruction, b'\x90')
                pe.set_bytes_at_rva(addr, new_code)

                ## Logging
                if is_debug_on():
                    l += f'''
                    Patching a {imp_type} 
                    current address         = {hex(address_call_stub)}({hex(address_call_stub+ib)})
                    address_call_stub       = {hex(address_call_stub)}({hex(address_call_stub+ib)})
                    row_size                = {JUMP_TABLE_ROW_SIZE}
                    call_instruction_size   = {call_instruction_size}
                    jump_table_address      = {hex(jump_table_address)}({hex(jump_table_address+ib)})
                    where_to_jump           = jump_table_address + 
                    where_to_jump           = {hex(where_to_jump)}({hex(where_to_jump+ib)})
                    displacement            = where_to_jump - address_call_stub - call_instruction_size
                    displacement            = {hex(displacement)}
                    call {hex(displacement)}
                    WRITING {new_code}      @ {hex(address_call_stub)}({hex(address_call_stub+ib)})
                    '''
                

            elif imp_type == ImportType.JUMP_STUB:
                where_to_jump           = jump_table_address + previous_jmp_table_row_size
                # We now encode the jump to the previously computed address

                address_jump_stub       = addr                              
                displacement            = where_to_jump - address_jump_stub     
                new_code                = asm(f'jmp $+{hex(displacement)}')
                new_code                = new_code.ljust(len_instruction, b'\x90')
                pe.set_bytes_at_rva(address_jump_stub, new_code)

                ### Logging
                if is_debug_on():
                    l += f'''
                    Patching a {imp_type} 
                    current address         = {hex(address_jump_stub)}({hex(address_jump_stub+ib)})
                    row_size                = {JUMP_TABLE_ROW_SIZE}
                    jump_table_address      = {hex(jump_table_address)}({hex(jump_table_address+ib)})
                    where_to_jump           = jump_table_address +previous_jmp_table_row_size
                    where_to_jump           = {hex(where_to_jump)}({hex(where_to_jump+ib)})
                    displacement            = where_to_jump - addr_jump_stub
                    displacement            = {hex(displacement)}
                    jmp $+{hex(displacement)}
                    WRITING {new_code}      @ {hex(address_jump_stub)}({hex(address_jump_stub+ib)})
                    '''
                ###

        if is_debug_on():
            logger.debug(l)
        if is_log_info_on():
            logger.info(self.dump(pe))