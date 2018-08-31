#!/usr/bin/python
#hello.py
from mpi4py import MPI
import mpi4py
comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

print ("hello world from process ", rank,"of", size)
print (mpi4py.get_config())
