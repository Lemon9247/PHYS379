import numpy as np
import random
import itertools
import math
import fractions

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
    extended_gate = extend_adjacent_binary(q=(q[1]-1,q[1]),gate=gate,bits=bits)
    swapper2 = get_swapper([i for i in range(bits)][::-1],q[::-1],bits)
    final_gate = np.matmul(extended_gate,swapper1)
    final_gate = np.matmul(swapper2,final_gate)
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

def IQFT(inputq):
    N = len(inputq)
    omega = np.exp(-2*np.pi*1j/N)
    IQFT_matrix = np.array([[0 for j in range(N)] for i in range(N)],dtype=complex)
    for i in range(N):
        for j in range(N):
            IQFT_matrix[i][j] = omega**(i*j)
    result = np.matmul(IQFT_matrix,inputq)
    return result

def shors(bigN):
    """
    Performs quantum Shor's algorithm on a given number
    Variable names taken from the Wikipedia page for Shor's
    """
    HADAMARD = np.array([[1/math.sqrt(2),1/math.sqrt(2)],[1/math.sqrt(2),-1/math.sqrt(2)]])
    a = random.randint(2,bigN-1)
    k = math.gcd(1,bigN)
    initregister = []
    phase = 0
    for i in range(bigN**2,2*bigN**2):#Finds the right number of required qubits
        if np.log2(i)%1 == 0:
            qbitnum = int(np.log2(i))+1#Extra qubit for the U gates
            break
    print(qbitnum)
    for i in range(2**qbitnum):#Creates the initial register. All start at 0 except the extra qubit which starts at 1
        if i == 1:
            initregister.append(1)
        else:
            initregister.append(0)
    initregister = np.array(initregister)
    currq = initregister
    if k != 1:
        return k
    else:
        currq = extend_unary(gate=HADAMARD,bits=qbitnum)
        for i in range(qbitnum-1):#Do the U gates
            UGATE = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,a**2**i%bigN]])
            gate2 = extend_binary(q=(qbitnum,i),gate=UGATE, bits=qbitnum)
            currq = np.matmul(gate2,currq)
        result=measure(IQFT(currq))#Inverse QFT function
        #print(result)
        for i in range(len(result)):
            if result[-(i+1)] == "1":
                phase+=2**(i)
        #print(phase)
        phase = phase/(2**qbitnum)
        frac = fractions.Fraction(phase).limit_denominator(bigN)
        s, r = frac.numerator, frac.denominator
        #if r%2 != 0:
         #   return "run again"
        #if a**(1/2*r)==-1%bigN:
         #   return "run again"
        guesses = [math.gcd(a**(r//2)-1, bigN), math.gcd(a**(r//2)+1, bigN)]
        return guesses
            

def main():
    while True:
        print(shors(21))
    

if __name__ == "__main__":
    main()
