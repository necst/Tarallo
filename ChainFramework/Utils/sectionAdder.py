import pefile

IMAGE_DIRECTORY_ENTRY_BOUND_IMPORT = 11


def align(val_to_align, alignment):
    return int((val_to_align + alignment - 1) / alignment) * alignment


def addSection(input_PE, section_name, raw_size, characteristics=0xe0000020, data=None):    

    section_header_len = 0x28   # This is fixed for each section header

    virtual_size = raw_size     # Virtual size is the size of the section when loaded in memory

    bound_import_table_rva  = input_PE.OPTIONAL_HEADER.DATA_DIRECTORY[IMAGE_DIRECTORY_ENTRY_BOUND_IMPORT].VirtualAddress
    bound_import_table_size = input_PE.OPTIONAL_HEADER.DATA_DIRECTORY[IMAGE_DIRECTORY_ENTRY_BOUND_IMPORT].Size

    if bound_import_table_size != 0:
        # If there is a bound import table, move it down after the last section header
        
        # 1. Read the import table
        bound_import_table = input_PE.get_data(bound_import_table_rva, bound_import_table_size)
        
        # 2. Rewrite it 
        input_PE.set_bytes_at_offset(bound_import_table_rva + section_header_len, bound_import_table)

        # 3. Change the pointer, now the bound import table is @ old_location + 0x28
        input_PE.OPTIONAL_HEADER.DATA_DIRECTORY[IMAGE_DIRECTORY_ENTRY_BOUND_IMPORT].VirtualAddress += section_header_len

        # 4. Telling pefile not to ruin our work
        for s in input_PE.__structures__:
            if  s.name == 'IMAGE_BOUND_IMPORT_DESCRIPTOR' or s.name == 'IMAGE_BOUND_FORWARDER_REF' :
                current_RVA = s.get_file_offset()
                new_RVA     = current_RVA + section_header_len
                s.set_file_offset(new_RVA)

    
    # Code from the pull request https://github.com/erocarrera/pefile/pull/286/commits/e6d732c0e5fe92bb1a6bbe4ead0ab5954fd4a77e

    # if characteristics == 0:
    #     # CODE | EXECUTE | READ | WRITE
    #     characteristics = 0xE0000020

    if data and raw_size < len(data):
        raise Exception("Invalid raw_size.")
    if data and virtual_size < len(data):
        raise Exception("Invalid virtual_size.")

    # If there is an overlay save it
    overlay = input_PE.get_overlay()
   
    number_of_section  = input_PE.FILE_HEADER.NumberOfSections
    last_section       = number_of_section - 1
    file_alignment     = input_PE.OPTIONAL_HEADER.FileAlignment
    section_alignment  = input_PE.OPTIONAL_HEADER.SectionAlignment
    new_section_offset = (input_PE.sections[number_of_section - 1].get_file_offset() + section_header_len)


    # Look for valid values for the new section header
    raw_size           = align(raw_size, file_alignment)
    virtual_size       = align(virtual_size, section_alignment)
    raw_offset         = align((input_PE.sections[last_section].PointerToRawData +
                                input_PE.sections[last_section].SizeOfRawData),
                                file_alignment)
    virtual_offset     = align((input_PE.sections[last_section].VirtualAddress +
                            input_PE.sections[last_section].Misc_VirtualSize),
                            section_alignment)

    if overlay is not None:
        # Raw offset is pushed after the overlay BUT the virtual address is the same
        # since the overlay will not be loaded in memory
        raw_offset      = align((input_PE.sections[last_section].PointerToRawData +
                            input_PE.sections[last_section].SizeOfRawData + len(overlay)),
                            file_alignment)

    # Section name must be equal to 8 bytes
    if len(section_name) < 8:
        section_name += ((8 - len(section_name)) * b'\x00')

    # Create the section
    # Set the section_name
    input_PE.set_bytes_at_offset(new_section_offset, section_name)
    # Set the virtual size
    input_PE.set_dword_at_offset(new_section_offset + 8, virtual_size)
    # Set the virtual offset
    input_PE.set_dword_at_offset(new_section_offset + 12, virtual_offset)
    # Set the raw size
    input_PE.set_dword_at_offset(new_section_offset + 16, raw_size)
    # Set the raw offset
    input_PE.set_dword_at_offset(new_section_offset + 20, raw_offset)
    # Set the following fields to zero
    input_PE.set_bytes_at_offset(new_section_offset + 24, (12 * b'\x00'))
    # Set the characteristics
    input_PE.set_dword_at_offset(new_section_offset + 36, characteristics)

    # Edit the value in the File and Optional headers
    input_PE.FILE_HEADER.NumberOfSections += 1
    input_PE.OPTIONAL_HEADER.SizeOfImage = virtual_size + virtual_offset

    # extend file
    if len(input_PE.__data__) < raw_offset:
        input_PE.__data__ += (raw_offset - len(input_PE.__data__)) * b'\x00'
    input_PE.__data__ = input_PE.__data__[:raw_offset]
    input_PE.__data__ += raw_size * b'\x00'

    # write new data
    if data:
        write_data = data
        if len(write_data) < raw_size:
            write_data += (raw_size - len(write_data)) * b'\x00'
        input_PE.set_bytes_at_offset(raw_offset, write_data)
    else:
        input_PE.set_bytes_at_offset(raw_offset, raw_size * b'\x00')
        #print(f"Data written @ {hex(raw_offset)}")

    return input_PE.write(), virtual_offset  