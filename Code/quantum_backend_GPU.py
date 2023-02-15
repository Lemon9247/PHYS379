import numpy as np
import cupy as cp
import random

def measure(inputq=None):
	"""
	Measures the N qubit register, simulating quantum randomness.
	Uses algorithm given in the PHYS379 Quantum Computer project notes.
	inputq = state vector to be measured (numpy array - Cannot be cupy array!)
	"""
	if inputq is None:
		raise SyntaxError("Qubit state vector to measure not specified!")
	q = 0
	r = random.random() # Random number between 0 and 1
	qbitnum = int(np.log2(len(inputq))) # Number of qubits

	for i,state_component in enumerate(inputq):
		q += state_component**2 # Adds value to existing q
		if q > r:
			return i # Returns the measured bit state in its decimal representation
	return len(inputq)-1 # Necessary due to floating point imprecision for large qubit counts

def extend_hadamards(bits=None,verbose=None):
	"""
	Extend unary gate to an N qubit state. If no target is supplied then the gate is applied to all qubits.
	targets = indices of qubits the gate is applied to. Indexing of qubits starts from ZERO! (int list)
	gate = unary gate to be extended to a multi-qubit state. (2D cupy array, size 2*2)
	bits = number of qubits (int)
	"""
	if bits is None:
		raise SyntaxError("Number of qubits not specified")
	if verbose is None:
		verbose = False

	if bits == 1:
		return 1/(np.sqrt(2))*cp.array([[1,1],[1,-1]],dtype=cp.float32)
	else:
		kron = 1/(np.sqrt(2))*cp.array([[1,1],[1,-1]],dtype=cp.float32)
		temp_gate = 1/(np.sqrt(2))*cp.array([[1,1],[1,-1]],dtype=cp.float32)
		for i in range(bits-1):
			temp_gate = cp.kron(temp_gate,kron)
			if verbose: print("Computed {}/{} tensor products".format(i+1,bits-1), end="\r",flush=True)
		
		if verbose: print()
		return temp_gate

class Grover:

	def __init__(self,oracle_function,bits,verbose=None):
		if verbose is None:
			self.verbose = False
		else:
			self.verbose = verbose
		self.bitnumber = bits

		if self.verbose: print("Computing Hadamard Network...")
		self.hadamards = extend_hadamards(bits=self.bitnumber,verbose=self.verbose)
		if self.verbose: print("Done!")

		self.diffuser = self.compute_diffuser()
		self.quantum_oracle = self.compute_oracle(oracle_function)

	def compute_diffuser(self):
		if self.verbose: print("Computing Diffuser...")
		diffuser = -1*cp.identity(2**self.bitnumber,dtype=cp.float32)
		diffuser[0,0] = 1
		diffuser = cp.matmul(self.hadamards,diffuser)
		diffuser = cp.matmul(diffuser,self.hadamards)

		if self.verbose: print("Done!")
		return diffuser

	def compute_oracle(self,oracle_function):
		if self.verbose: print("Computing Quantum Oracle...")
		quantum_oracle = cp.identity(2**self.bitnumber,dtype=cp.float32)
		for i in range(2**self.bitnumber):	# Construct oracle on CPU as looping over elements on the GPU is slow
			try:	# Use try/except in case the number of bits exceeds original register bitlength
				if oracle_function(i):
					quantum_oracle[i,i] = -1
			except:
				continue

		if self.verbose: print("Done!")
		return quantum_oracle

	def search(self,iterations):
		"""
		Performs a Grover Search for a given number of iterations. Returns a numpy array.
		Iterations: The number of iterations to compute (int)
		"""
		target_state = cp.zeros(2**self.bitnumber,dtype=cp.float32)
		target_state[0] = 1
		target_state = cp.matmul(self.hadamards,target_state)

		circuit = cp.identity(2**self.bitnumber,dtype=cp.float32)
		for i in range(iterations):
			circuit = cp.matmul(circuit,self.quantum_oracle)
			circuit = cp.matmul(circuit,self.diffuser)
			
			if self.verbose: print("Started {}/{} Grover Iterations".format(i+1,iterations),end="\r",flush=True)

		target = cp.matmul(circuit,target_state)
		if self.verbose: print("\nFinalising calculations. This may take a little while, please wait...")
		target_cpu = cp.asnumpy(target)
		if self.verbose: print("Done!")
		return target_cpu

# Create temporary instance of Grover class to initialise backend
Grover(lambda x: True if x == 1 else False,1)
print("GPU backend ready!")
