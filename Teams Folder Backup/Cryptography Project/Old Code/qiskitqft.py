import numpy as np
from numpy import pi
# importing Qiskit
from qiskit import QuantumCircuit, transpile, assemble, Aer, IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, plot_bloch_multivector
qc = QuantumCircuit(3)
qc.h(2)
qc.draw()
qc.cp(pi/2, 1, 2) # CROT from qubit 1 to qubit 2
qc.draw()
qc.cp(pi/4, 0, 2) # CROT from qubit 2 to qubit 0
qc.draw()
qc.h(1)
qc.cp(pi/2, 0, 1) # CROT from qubit 0 to qubit 1
qc.h(0)
qc.draw()
qc.swap(0,2)
qc.draw()