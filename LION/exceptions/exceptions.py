# add non-coder related errors pop up add to this to improve overall quality of error handling and reporting
# general rule: if it's possible that error condition will occur, use an exception, not an assertion

class NoDataException(Exception):
    pass

class LIONSolverException(Exception):
    pass