class MachineTypeError(Exception):
    pass

class InsuffientPadding(Exception):
    pass

class NoApiCallsToInject(Exception):
    pass

class NoImportToHijack(Exception):
    pass

class ApiCallsNeverCalled(Exception):
    pass

class NoXSectionFound(Exception):
    pass

class FileNotPE(Exception):
    pass

class NoDynamicBaseFlag(Exception):
    pass

class NoRelocationsStrippedFlag(Exception):
    pass

class AdjustPEError(Exception):
    pass

class CannotInjectAPI(Exception):
    pass