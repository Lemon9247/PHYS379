import numpy as np
import random
import itertools
import math
import fractions

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

    temp_gate = np.array(1)
    for i in range(bits):
        if i == q: # Gates are combined together via tensor/kronecker product
            temp_gate = np.kron(temp_gate,gate) # Insert the gate into the resulting matrix that acts on desired qubit
        else:
            temp_gate = np.kron(temp_gate,np.identity(2,dtype=complex)) # Insert an identity matrix to act on a different qubit
    return temp_gate

def extend_binary(q=None,gate=None,bits=None):
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
    #if q[1] != q[0]+1:
        #raise SyntaxError("Gate must be applied to adjacent qubits")

    temp_gate = np.array(1)
    on_second_bit = False
    for i in range(bits):
        found = False
        if on_second_bit:
            on_second_bit = False
            continue
        for j in range(bits):#potential method for doing non-adjacent qubits. seems to work ok so far?
            if(i,j)==q:
                temp_gate = np.kron(temp_gate,gate) # Insert the gate into the resulting matrix that acts on desired qubit
                on_second_bit = True
                found = True
                break
        if found == False:
                temp_gate = np.kron(temp_gate,np.identity(2,dtype=complex))# Insert an identity matrix to act on a different qubit
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
        q += np.linalg.norm(state_component**2)#Adds value to existing q
        if q > r:
            out = check[i] #Outputs the corresponding combination of 0 and 1 
            break
    return out
def IQFT(inputq):
        """
        Inverse quantum fouier transform function for given N qubit state
        Adjusted so it leaves out the last qubit in register for Shor's
        """
        final = 0
        qbitnum = int(np.log2(len(inputq)))
        SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
        HADAMARD = np.array([[1/math.sqrt(2),1/math.sqrt(2)],[1/math.sqrt(2),-1/math.sqrt(2)]])#Define gates
        currq = inputq
        for i in range(qbitnum,0,-1):
                gate1 = extend_unary(q=i-1,gate=HADAMARD,bits=qbitnum)
                currq = np.matmul(gate1,currq)
                for k in range(i):#Apply CROT gates to every qubit below the current one
                        if i-1 == k:
                                break
                        CROT = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,math.e**(-2*math.pi*1j/(i-k))]])
                        gate2 = extend_binary(q=(k,i-1),gate=CROT,bits=qbitnum)
                        currq = np.matmul(gate2,currq)
                if i == 1:
                        gate1 = extend_unary(q=i-1,gate=HADAMARD,bits=qbitnum)
                        final = np.matmul(gate1,currq)
                        break
        return final
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
        for i in range(qbitnum-1):# Do Hadamards on the 0 qubits
            gate1 = extend_unary(q=i,gate=HADAMARD,bits=qbitnum)
            currq = np.matmul(gate1,currq)
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
            #return "run again"
        #if a**(1/2*r)==-1%bigN:
            #return "run again"
        guesses = [math.gcd(a**(r//2)-1, bigN), math.gcd(a**(r//2)+1, bigN)]
        return guesses
            

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
    #gate1 = extend_unary(q=0,gate=NOT,bits=N_QUBITS)
    #gate2 = extend_unary(q=1,gate=NOT,bits=N_QUBITS)
    #circuit = np.matmul(gate2,gate1)
    #circuit = extend_binary(q=(1,2),gate=SWAP,bits=N_QUBITS)

    # Send qubit through the circuit
    #q0 = np.matmul(circuit,q0)
    while True:
        print(shors(21))
    

if __name__ == "__main__":
    main()
