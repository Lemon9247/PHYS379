import numpy as np
import random

def measure(inputq=None):
	"""
	Measures the N qubit register, simulating quantum randomness.
	Uses algorithm given in the PHYS379 Quantum Computer project notes.
	inputq = state vector to be measured (Numpy Array)
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
	return len(inputq)-1	# Necessary due to floating point imprecision for large qubit counts

def extend_unary(targets=None,gate=None,bits=None,verbose=None):
	"""
	Extend unary gate to an N qubit state. If no target is supplied then the gate is applied to all qubits.
	targets = indices of qubits the gate is applied to. Indexing of qubits starts from ZERO! (int list)
	gate = unary gate to be extended to a multi-qubit state. (2D complex numpy array, size 2*2)
	bits = number of qubits (int)
	"""
	if gate is None:
		raise SyntaxError("No gate specified")
	if bits is None:
		raise SyntaxError("Number of qubits not specified")
	if verbose is None:
		verbose = False

	temp_gate = np.array(1,dtype=np.float32)
	if targets is None:
		for i in range(bits):
			temp_gate = np.kron(temp_gate,gate)
			if verbose: print("Computed {}/{} tensor products".format(i+1,bits),end="\r",flush=True)
	else:
		for i in range(bits):
			if i in targets:
				temp_gate = np.kron(temp_gate,gate)
			else: # Insert an identity matrix to act on a different qubit
				temp_gate = np.kron(temp_gate,np.identity(2,dtype=np.float32))
			if verbose: print("Computed {}/{} tensor products".format(i+1,bits),end="\r",flush=True)
	if verbose: print()
	return temp_gate


def get_error_matrix(bits,errorp):
	error_size = 0.01
	# Define generators of U(2) and the identity matrix
	X = np.array([[0,1],[1,0]])
	Y = np.array([[0,-1j],[1j,0]])
	Z = np.array([[1,0],[0,-1]])
	I = np.array([[1,0],[0,1]])

	# Create a list of targets to apply random error "gates" to
	targets = [[random.randint(0,bits-1)]]
	for i in range(bits):
		if i in targets:
			continue
		elif random.random() <= errorp:
			targets.append([i])
	
	matrices = []
	for target in targets:
		n_vec = np.random.rand(3)	# Create randomised axis vector for the gate
		for i,component in enumerate(n_vec):
			n_vec[i] = component*((-1)**random.randint(0,1))	# Flip sign of components at random
		n_vec = n_vec/np.linalg.norm(n_vec)		# Ensure the axis vector is normalised
		angle = (4*np.pi*error_size)*random.random()	# Pick a random angle between 0 and pi/8
		matrix = np.cos(angle/2)*I-1j*np.sin(angle/2)*(n_vec[0]*X+n_vec[1]*Y+n_vec[2]*Z)	# Construct the gate
		extended_matrix = extend_unary(targets=target,gate=matrix,bits=bits)	# Extend the gate to the multi-qubit setup
		matrices.append(extended_matrix)

	# Multiply all the error "gates" together to construct the overall error gate
	error_matrix = np.identity(2**bits,dtype=np.float32)
	for matrix in matrices:
		error_matrix = np.matmul(error_matrix,matrix)
	return error_matrix

class Grover:

	def __init__(self, oracle_function, bits, verbose=None):
		if verbose is None:
			self.verbose = False
		else:
			self.verbose = verbose
		self.bitnumber = bits

		if self.verbose: print("Computing Hadamard Network...")
		hadamard_gate = 1/(np.sqrt(2))*np.array([[1,1],[1,-1]],dtype=np.float32)
		self.hadamards = extend_unary(gate=hadamard_gate,bits=self.bitnumber,verbose=self.verbose)
		if self.verbose: print("Done!")

		self.diffuser = self.compute_diffuser()
		self.quantum_oracle = self.compute_oracle(oracle_function)

	def compute_diffuser(self):
		if self.verbose: print("Computing Diffuser...")
		diffuser = -1*np.identity(2**self.bitnumber,dtype=np.float32)
		diffuser[0,0] = 1
		diffuser = np.matmul(self.hadamards,diffuser)
		diffuser = np.matmul(diffuser,self.hadamards)

		if self.verbose: print("Done!")
		return diffuser

	def compute_oracle(self,oracle_function):
		if self.verbose: print("Computing Quantum Oracle...")
		quantum_oracle = np.identity(2**self.bitnumber,dtype=np.float32)
		for i in range(2**self.bitnumber):
			try:	# Use try/except in case the number of bits exceeds original register bitlength
				if oracle_function(i):
					quantum_oracle[i,i] = -1
			except:
				continue

		if self.verbose: print("Done!")
		return quantum_oracle

	def search(self,iterations,errorp=None):
		"""
		Performs a Grover Search for a given number of iterations. Returns a numpy array.
		Iterations: The number of iterations to compute (int)
		"""
		target_state = np.zeros(2**self.bitnumber,dtype=np.float32)
		target_state[0] = 1
		target_state = np.matmul(self.hadamards,target_state)

		circuit = np.identity(2**self.bitnumber,dtype=np.float32)
		for i in range(iterations):
			if errorp is not None:
				if random.random() <= errorp: circuit=np.matmul(circuit,get_error_matrix(self.bitnumber,errorp))
			circuit = np.matmul(circuit,self.quantum_oracle)
			if errorp is not None:
				if random.random() <= errorp: circuit=np.matmul(circuit,get_error_matrix(self.bitnumber,errorp))
			circuit = np.matmul(circuit,self.diffuser)
			if errorp is not None:
				if random.random() <= errorp: circuit=np.matmul(circuit,get_error_matrix(self.bitnumber,errorp))
			if self.verbose: print("Completed {}/{} Grover Iterations...".format(i+1,iterations), end="\r",flush=True)
		if self.verbose: print("\nDone!")

		target = np.matmul(circuit,target_state)
		return target