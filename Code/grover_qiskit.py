from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, transpile
from qiskit.visualization import plot_histogram
import qiskit.quantum_info as qi
import matplotlib.pyplot as plt
import numpy as np

def hadamard(qcircuit, x): #Apply a H-gate to x number of qubits in qcircuit
    for i in range(x):
        qcircuit.h(i)
    return qcircuit

def oracle(x,i): #x is number of qubits and i is number of solution qubit 
    #ie if number of qubits is x, then i has to be between 1 and 2^x
    #in the case of 3 qubits i=0 000, i=1 001, i=2 010, i=3 011, etc
    oracle = np.identity(2**x)
    oracle[i,i]= -1
    return oracle

def op_j(x): #creates operator J, x is number of qubits
    j = -1*np.identity(2**x)
    j[0,0]= 1
    return j

def diffusion(x,circuit): #x is number of qubits
    circuit = hadamard(circuit,x)
    j_matrix = op_j(x)
    j_op = qi.Operator(j_matrix)
    circuit.unitary(j_op,range(x),label='J')
    circuit = hadamard(circuit,x)
    return circuit

n = 10 #number of qubits
circuit = QuantumCircuit(n)
simulator = Aer.get_backend('aer_simulator')

circuit = hadamard(circuit, n)
oraclematrix = oracle(n,3)
oracle_op = qi.Operator(oraclematrix)
iterations= int(np.ceil(np.sqrt(2**n)))

for i in range(iterations):
    circuit.unitary(oracle_op, range(n), label='ORACLE')
    diffusion(n,circuit)
circuit.measure_all()

counts = []
for i in range(4):
    job = simulator.run(circuit, shots=1000)
    result = job.result()
    c = result.get_counts(circuit)
    counts.append(c)

legend = ['First execution', 'Second execution', 'Third execution', 'Fourth execution']
plot_histogram([counts[0], counts[1], counts[2], counts[3]], legend=legend, figsize=(7, 5), bar_labels=False)
plt.tight_layout()
plt.show()