from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, transpile
from qiskit.visualization import plot_histogram
import qiskit.quantum_info as qi
import matplotlib.pyplot as plt
import numpy as np
import random

from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler

class Grover:

	def __init__(self,oracle_function,bits,verbose=None):
		if verbose is None:
			self.verbose = False
		else:
			self.verbose = verbose
		self.bitnumber = bits

		self.circuit = QuantumCircuit(self.bitnumber)
		self.reflector = self.build_reflector()
		self.quantum_oracle = self.build_oracle(oracle_function)

	def build_reflector(self):
		if self.verbose: print("Computing Diffuser...")
		reflector_matrix = -1*np.identity(2**self.bitnumber)
		reflector_matrix[0,0] = 1
		reflector = qi.Operator(reflector_matrix)
		if self.verbose: print("Done!") 
		return reflector

	def build_oracle(self,oracle_function):
		if self.verbose: print("Computing Quantum Oracle...")
		oracle_matrix = np.identity(2**self.bitnumber)
		for i in range(2**self.bitnumber):	# Construct oracle on CPU as looping over elements on the GPU is slow
			try:	# Use try/except in case the number of bits exceeds original register bitlength
				if oracle_function(i):
					oracle_matrix[i,i] = -1
			except:
				continue

		quantum_oracle = qi.Operator(oracle_matrix)
		if self.verbose: print("Done!")
		return quantum_oracle

	@staticmethod
	def hadamards(qcircuit,bits):
		"""
		qcircuit = qiskit circuit to apply hadamard gates to
		bits = number of bits in the circuit
		"""
		for i in range(bits):
			qcircuit.h(i)
		return qcircuit

	@staticmethod
	def diffusion(reflector,circuit,bits):
		circuit = Grover.hadamards(circuit,bits)
		circuit.unitary(reflector,range(bits),label='J')
		circuit = Grover.hadamards(circuit,bits)
		return circuit

	def search(self,iterations):
		"""
		Performs a Grover Search for a given number of iterations. Returns a numpy array.
		Iterations: The number of iterations to compute (int)
		"""
		self.circuit = self.hadamards(self.circuit, self.bitnumber)

		for i in range(iterations):
			self.circuit.unitary(self.quantum_oracle, range(self.bitnumber), label='ORACLE')
			self.circuit = self.diffusion(self.reflector,self.circuit,self.bitnumber)
			if self.verbose: print("Started {}/{} Grover Iterations".format(i+1,iterations),end="\r",flush=True)
		self.circuit.measure_all()

		if self.verbose: print("\nDone!")

def main():
	database = []
	length = 2**5

	service = QiskitRuntimeService(channel="ibm_quantum", token="d3af0ce8c5b71e2209966865c7116d786b01d7fea4462bb74467986202181701eb74190653a02ee77087f4db1e0c2e96b2e60a5e8d8eea7799267123fe92ec08")
	for i in range(length):
		database.append(0)
	targets=1
	for i in range(targets):
		database[random.randint(0,length-1)]=1

	def f(x):
		if database[x] == 1:
			return True
		else:
			return False
			
	bits = int(np.ceil(np.log2(len(database))))
	iterations = int(np.ceil(np.sqrt(len(database)/targets)))
	J = Grover(f,bits,verbose=True)
	J.search(iterations)
	shots = 1024
	counts = []
	for i in range(1):
		with Session(service=service, backend="ibmq_manila") as session:
			sampler = Sampler(session=session)
			job = sampler.run(circuits=J.circuit,shots=1024)
			result = job.result()
			c = result.get_counts(J.circuit)
			counts.append(c)
	legend = ['First execution', 'Second execution', 'Third execution', 'Fourth execution']
	plot_histogram([counts[0], counts[1], counts[2], counts[3]], legend=legend, figsize=(7, 5), bar_labels=False)
	plt.tight_layout()
	plt.show()

if __name__=="__main__":
	main()