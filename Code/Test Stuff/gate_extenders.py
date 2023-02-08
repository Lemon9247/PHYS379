import numpy as np
import random
import itertools

def extend_unary(q=None,gate=None,bits=None):
	"""
	Extend unary gate to an N qubit state
	q = indices of qubits the gate is applied to. Indexing of qubits starts from ZERO! (int list)
	gate = unary gate to be extended to a multi-qubit state. (2D complex numpy array, size 2*2)
	bits = number of qubits (int)
	"""
	if q is None:
		raise SyntaxError("Missing qubit index")
	if gate is None:
		raise SyntaxError("No gate specified")
	if bits is None:
		raise SyntaxError("Number of qubits not specified")

	temp_gate = np.array(1)
	for i in range(bits):
		if i in q: # Gates are combined together via tensor/kronecker product
			temp_gate = np.kron(temp_gate,gate) # Insert the gate into the resulting matrix that acts on desired qubit
		else:
			temp_gate = np.kron(temp_gate,np.identity(2,dtype=complex)) # Insert an identity matrix to act on a different qubit
	return temp_gate

def extend_binary(q=None,gate=None,bits=None):
	"""
	Extend binary gate to an N qubit state
	q = indices of adjacent qubits the gate is applied to. Indexing of qubits starts from ZERO! (tuple of 2 integers)
	gate = binary gate to be extended to a multi-qubit state. (2D complex numpy array, size 4*4)
	bits = number of qubits (int)
	"""
	if q is None:
		raise SyntaxError("Input qubit indices not specified")
	if gate is None:
		raise SyntaxError("No gate specified")
	if bits is None:
		raise SyntaxError("Number of qubits not specified")
	if q[1] != q[0]+1:
		raise SyntaxError("Gate must be applied to ajacent qubits")

	temp_gate = np.array(1)
	on_second_bit = False
	for i in range(bits):
		if on_second_bit:
			on_second_bit = False
			continue
		if (i,i+1) == q: # Gates are combined together via tensor/kronecker product
			temp_gate = np.kron(temp_gate,gate) # Insert the gate into the resulting matrix that acts on desired qubit
			on_second_bit = True
		else:
			temp_gate = np.kron(temp_gate,np.identity(2,dtype=complex)) # Insert an identity matrix to act on a different qubit
	return temp_gate

def measure(inputq):
	"""
	Measures the N qubit register, simulating quantum randomness.
	Uses algorithm given in quantum_computer.pdf
	"""
	q=0
	r = random.uniform(0,1)#Random number between 0 and 1
	qbitnum = int(np.log2(len(inputq)))#Number of qubits
	check = list(itertools.product("01",repeat = qbitnum))#Creates a list of every possible combination of 0 and 1
	for i,state_component in enumerate(inputq):
		q += state_component**2#Adds value to existing q
		if q > r:
			out = check[i] #Outputs the corresponding combination of 0 and 1 
			break
	return out

def main():
	# Define input qubit state vector and count number of qubits
	q0 = np.array((0,0,1,0,0,0,0,0),dtype=complex)
	q0 = q0/np.linalg.norm(q0) # Ensure state is normalised
	N_QUBITS = int(np.log2(len(q0)))

	# Define unary quantum gates
	NOT = np.array([[0,1],[1,0]])
	PAULI_Y = np.array([[0,-1j],[1j,0]])
	PAULI_Z = np.array([[1,0],[0,-1]])
	SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])

	# Create circuit
	#gate1 = extend_unary(q=[0],gate=NOT,bits=N_QUBITS)
	#gate2 = extend_unary(q=[1],gate=NOT,bits=N_QUBITS)
	#circuit = np.matmul(gate2,gate1)
	circuit = extend_binary(q=(1,2),gate=SWAP,bits=N_QUBITS)

	# Send qubit through the circuit
	q0 = np.matmul(circuit,q0)
	print(q0)
	print(measure(q0))

if __name__ == "__main__":
	main()
