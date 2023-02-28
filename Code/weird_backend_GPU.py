import numpy as np
import cupy as cp
import random, itertools

def extend_unary(q=None,gate=None,bits=None):
    """
    Extend unary gate to an N qubit state
    q = index of qubit the gate is applied to. Indexing of qubits starts from ZERO! (int)
    gate = unary gate to be extended to a multi-qubit state. (2D complex numpy array, size 2*2)
    bits = number of qubits (int)
    """
    if q is None:
        raise SyntaxError("Missing qubit index")
    if gate is None:
        raise SyntaxError("No gate specified")
    if bits is None:
        raise SyntaxError("Number of qubits not specified")

    temp_gate = cp.array(1)
    for i in range(bits):
        if i == q: # Gates are combined together via tensor/kronecker product
            temp_gate = cp.kron(temp_gate,gate) # Insert the gate into the resulting matrix that acts on desired qubit
        else:
            temp_gate = cp.kron(temp_gate,cp.identity(2,dtype=cp.float32)) # Insert an identity matrix to act on a different qubit
    return temp_gate

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

    temp_gate = cp.array(1)
    on_second_bit = False
    for i in range(bits):
        if on_second_bit:
            on_second_bit = False
            continue
        if (i,i+1) == q: # Gates are combined together via tensor/kronecker product
            temp_gate = cp.kron(temp_gate,gate) # Insert the gate into the resulting matrix that acts on desired qubit
            on_second_bit = True
        else:
            temp_gate = cp.kron(temp_gate,cp.identity(2,dtype=cp.float32)) # Insert an identity matrix to act on a different qubit
    return temp_gate

def get_swapper(array,q,bits):
    swapper = cp.identity(2**bits,dtype=cp.float32)
    SWAP = cp.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
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
                    swapper = cp.matmul(extended_swap,swapper)
            break
    return swapper

def extend_binary(q=None,gate=None,bits=None):
    swapper1 = get_swapper([i for i in range(bits)],q,bits)
    extended_gate = extend_adjacent_binary(q=(q[1]-1,q[1]),gate=gate,bits=bits)
    swapper2 = get_swapper([i for i in range(bits)][::-1],q[::-1],bits)
    final_gate = cp.matmul(extended_gate,swapper1)
    final_gate = cp.matmul(swapper2,final_gate)
    return final_gate

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
            return "{0:b}".format(i) # Returns the measured bit state in its decimal representation
    return "{0:b}".format(len(inputq)-1)    # Necessary due to floating point imprecision for large qubit counts