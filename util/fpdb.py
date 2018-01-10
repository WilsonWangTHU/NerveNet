# -----------------------------------------------------------------------------
#   @brief:
#       some helper functions about pdb.
# -----------------------------------------------------------------------------

import sys
import pdb


class fpdb(pdb.Pdb):
    '''
        @brief:
            a Pdb subclass that may be used from a import forked multiprocessing
            child
    '''

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
