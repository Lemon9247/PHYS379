import numpy as np
import time

def extend_adjacent_binary(q=None,gate=None,bits=None):
	"""
	Extend binary gate to an N qubit state
	q = indices of adjacents qubit the gate is applied to. Indexing of qubits starts from ZERO! (int)
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
			temp_gate = np.kron(temp_gate,np.identity(2)) # Insert an identity matrix to act on a different qubit
	return temp_gate

def get_swapper(array,q,bits):
	swapper = np.identity(2**bits)
	SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
	for i,element1 in enumerate(array):
		if element1 != q[0]:
			continue
		else:
			# loop to compare array elements
			for j in range(0, len(array) - i - 1):
				if array[j] == q[0]:
					continue
				elif array[j] == q[1]:
					break
				else:
					temp = array[j]
					array[j] = array[j+1]
					array[j+1] = temp
					extended_swap = extend_adjacent_binary(q=(j-1,j),gate=SWAP,bits=bits)
					swapper = np.matmul(extended_swap,swapper)
			break
	return swapper

def extend_binary(q=None,gate=None,bits=None):
	swapper1 = get_swapper([i for i in range(bits)],q,bits)
	print(swapper1)
	extended_gate = extend_adjacent_binary(q=(q[1]-1,q[1]),gate=gate,bits=bits)
	input()
	swapper2 = get_swapper([i for i in range(bits)][::-1],q[::-1],bits)
	
	print(swapper2)
	input()
	final_gate = np.matmul(extended_gate,swapper1)
	final_gate = np.matmul(swapper2,final_gate)
	return final_gate

bits = 3
gate = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
print(gate)
r=extend_binary(q=(1,2),gate=gate,bits=bits)
print(r)
print(np.linalg.det(r))