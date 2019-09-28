"""The main module of Zharfa Face Recognition software
"""
from mpi4py import MPI  #pylint: disable = import-error
COMM = MPI.COMM_WORLD  #pylint: disable = c-extension-no-member
RANK = COMM.Get_rank()

# Kivy GUI Process #################################################################################
if RANK == 0:
    from utils.gui import ZharfaApp
    ZharfaApp(COMM).run()

# Input Image Process ##############################################################################
if RANK == 1:
    from utils.imagetools import InputImageProcess
    InputImageProcess(COMM, 'images/Demo1.mkv', 'images/Demo1.mkv',
                      'images/Demo2.mp4').run()
