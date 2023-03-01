from weird_backend import *
import numpy as np
import random, fractions

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

class shor:

    def __init__(self,bigN,verbose=None):
        self.N = bigN
        self.verbose = verbose if verbose != None else False
        
        # For an n-bit integer, there should be 2n bits in the main register and n in the ancillary
        self.bits = int(3*np.ceil(np.log2(bigN)))
        #print(self.bits)
        self.ancillary_bitnumber = self.bits//3
        self.main_bitnumber = self.bits-self.ancillary_bitnumber
        
        self.IQFT = self.get_IQFT_matrix()

    def get_IQFT_matrix(self):
        N = 2**self.main_bitnumber
        #print(N)
        omega = np.exp(-2*np.pi*1j/N)
        IQFT_matrix = np.array([[0 for j in range(N)] for i in range(N)],dtype=complex)
        for i in range(N):
            for j in range(N):
                IQFT_matrix[i][j] = omega**(i*j)
        IQFT_matrix *= 1/(np.sqrt(N))
        for i in range(self.ancillary_bitnumber):
            IQFT_matrix = np.kron(IQFT_matrix,np.identity(2))
        #print(IQFT_matrix.shape)
        #print(IQFT_matrix)
        #print(np.linalg.det(IQFT_matrix))
        return IQFT_matrix

    def construct_CU_matrix(self,a,control_bit):
        if control_bit == 0:
            A = a%self.N
        else:
            A = (a**(2*control_bit-1))%self.N

        CU = np.zeros((2**self.bits,2**self.bits))
        get_bin = lambda x, n: format(x, 'b').zfill(n)
        for column_number in range(2**self.bits):
            k = get_bin(column_number,self.bits)
            if k[control_bit]== "0":
                j = int(k,2)
            else:
                main = k[:self.main_bitnumber]
                ancil = k[self.main_bitnumber:]
                f = int(ancil,2)
                if f >= self.N:
                    j=int(k,2)
                else:
                    f2 = (A*f)%self.N
                    new_ancil = get_bin(f2,self.ancillary_bitnumber)
                    j_binary = main+new_ancil # Combine strings
                    j = int(j_binary,2)

            CU[j][column_number] = 1
        return CU

    def run_algorithm(self):
        HADAMARD = 1/(np.sqrt(2))*np.array([[1,1],[1,-1]],dtype=np.float32)

        a = 7
        k = np.gcd(a,self.N)
        if k != 1:
            return [self.N//k,k]

        circuit = extend_unary(targets=[i for i in range(self.main_bitnumber)],gate=HADAMARD,bits=self.bits)
        for i in reversed(range(self.main_bitnumber)):#Do the U gates
            UGATE = self.construct_CU_matrix(a,i)
            circuit = np.matmul(UGATE,circuit)

        q_vec = np.array([0 for i in range(2**self.bits)])
        q_vec[1] = 1
        q_vec = np.matmul(circuit,q_vec)

        collapsed = measure(q_vec)
        collapsed_statevec = np.array(collapsed)

        states = []
        for i in collapsed:
            if i == "0":
                states.append(np.array([1,0]))
            else:
                states.append(np.array([0,1]))
        collapsed_statevec = states[0].copy()
        #print(states)
        for i,entry in enumerate(states):
            if i == 0:
                continue
            else:
                collapsed_statevec = np.kron(collapsed_statevec,entry)
        collapsed_statevec = collapsed_statevec/np.linalg.norm(collapsed_statevec)
        final_state = np.matmul(self.IQFT,collapsed_statevec)
        result = measure(final_state)
        #print(result)
        
        x_register_result = result[:self.main_bitnumber]
        print(x_register_result)
        int_result = int("".join(x_register_result[::-1]),2)
        phase = int_result/(2**self.main_bitnumber)
        print(int_result,phase)
        frac = fractions.Fraction(phase).limit_denominator(self.main_bitnumber)
        s, r = frac.numerator, frac.denominator
        guesses = [np.gcd(a**(r//2)-1, self.N), np.gcd(a**(r//2)+1, self.N)]
        return guesses

def main():
    while True:
        J=shor(15)
        result = J.run_algorithm()
        print(result)
    
if __name__ == "__main__":
    main()
