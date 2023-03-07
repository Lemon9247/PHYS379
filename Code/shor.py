import numpy as np
import random, fractions, itertools, time

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
    q = indices of adjacent qubits the gate is applied to. Indexing of qubits starts from ZERO! (int)
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

def get_error_matrix(bits,errorp):
    error_size = 0.1
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
        n_vec = np.random.rand(3)   # Create randomised axis vector for the gate
        for i,component in enumerate(n_vec):
            n_vec[i] = component*((-1)**random.randint(0,1))    # Flip sign of components at random
        n_vec = n_vec/np.linalg.norm(n_vec)     # Ensure the axis vector is normalised
        angle = (4*np.pi*error_size)*random.random()    # Pick a random angle between 0 and pi/8
        matrix = np.cos(angle/2)*I-1j*np.sin(angle/2)*(n_vec[0]*X+n_vec[1]*Y+n_vec[2]*Z)    # Construct the gate
        extended_matrix = extend_unary(targets=target,gate=matrix,bits=bits)    # Extend the gate to the multi-qubit setup
        matrices.append(extended_matrix)

    # Multiply all the error "gates" together to construct the overall error gate
    error_matrix = np.identity(2**bits,dtype=np.float32)
    for matrix in matrices:
        error_matrix = np.matmul(error_matrix,matrix)
    return error_matrix

class shor:

    def __init__(self,N,a=None,bits=None,verbose=None):
        """
        Class to handle shor's algorithm
        N = target number to factorise
        a = pivot for shor's algorithm. If not specified, a random number less than N is chosen
        bits = number of qubits in the main register. If not specified, there are 2n qubits for an n-bit value of N
        """
        self.N = N
        self.a = a if a!= None else random.randint(1,N-1)
        self.verbose = verbose if verbose != None else False   

        # For an n-bit integer, there should be n ancillary qubits
        # The number of qubits in the main register can be varied, but it is best to have 2n
        self.main_bitnumber = bits if bits != None else 2*int(np.ceil(np.log2(self.N)))
        self.ancillary_bitnumber = int(np.ceil(np.log2(self.N)))
        self.bits = self.main_bitnumber+self.ancillary_bitnumber
        if verbose: print("Ancillary Bits: {}, Total Bits: {}".format(self.ancillary_bitnumber,self.bits))
        # Construct IQFT matrix
        self.HADAMARD = 1/(np.sqrt(2))*np.array([[1,1],[1,-1]],dtype=np.float32)
        self.IQFT = self.get_IQFT_matrix_v2()

    def get_IQFT_matrix(self):
        """
        Inverse quantum fouier transform function for given N qubit state.
        Adjusted so it does not act on the ancillary qubits, only the main register
        """
        if self.verbose: print("Computing IQFT matrix for {} bits in working register".format(self.main_bitnumber))
        circuit  = np.identity(2**self.main_bitnumber)
        for i in reversed(range(self.main_bitnumber)):
            gate1 = extend_unary(targets=[i-1],gate=self.HADAMARD,bits=self.main_bitnumber)
            circuit = np.matmul(gate1,circuit)
            for k in range(i): #Apply CROT gates to every qubit below the current one
                    if i-1 == k:
                        break
                    CROT = np.array([[1,0,0,0],[0,np.e**(-2*np.pi*1j/(i-k)),0,0],[0,0,1,0],[0,0,0,1]])
                    gate2 = extend_binary(q=(k,i-1),gate=CROT,bits=self.main_bitnumber)
                    circuit = np.matmul(gate2,circuit)
        if self.verbose: print("Applying tensor products to IQFT matrix for ancillary register...")
        for i in range(self.ancillary_bitnumber):
            circuit = np.kron(circuit,np.identity(2))
        return circuit

    def get_IQFT_matrix_v2(self):
        """
        Alternative method to get inverse quantum fourier transform matrix on main register
        Uses mathematical definition of IQFT instead of building out of unary & binary gates
        By default this method remains unused!
        """
        if self.verbose: print("Computing IQFT matrix for {} bits in working register".format(self.main_bitnumber))
        N = 2**self.main_bitnumber
        omega = np.exp(-2*np.pi*1j/N)
        IQFT_matrix = np.array([[0 for j in range(N)] for i in range(N)],dtype=complex)
        for i in range(N):
            for j in range(N):
                IQFT_matrix[i][j] = omega**(i*j)
        IQFT_matrix *= 1/(np.sqrt(N))
        for i in range(self.ancillary_bitnumber):
            IQFT_matrix = np.kron(IQFT_matrix,np.identity(2))
        if self.verbose: print("Done!")
        return IQFT_matrix

    def construct_CU_matrix(self,control_bit):
        """
        CU matrix is constructed according to algorithm outlined in undergrad
        quantum computing projects paper
        """
        index = self.main_bitnumber-(control_bit+1) # Reverse numerical ordering of main register
        if index == 0:
            A = self.a%self.N
        else:
            A = (self.a**(2**index))%self.N
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

    def run_algorithm(self,errorp=None):
        """
        Calculates the output x/2^L ("phase") of Shor's algorithm for a given value of a
        """
        k = np.gcd(self.a,self.N)
        if k != 1:  # If a is already a non-trivial factor of N we are done
            if self.verbose: print("Random value a was already a non-trivial factor!")
            return ([self.N//k,k],True)   # Return True in second argument to flag algorithm was skipped

        circuit = extend_unary(targets=[i for i in range(self.main_bitnumber)],gate=self.HADAMARD,bits=self.bits)
        for i in reversed(range(self.main_bitnumber)):#Do the U gates
            UGATE = self.construct_CU_matrix(i)
            if self.verbose: print("Computed {}/{} controlled U gates".format(self.main_bitnumber-i,self.main_bitnumber),end="\r",flush=True)
            if errorp is not None:
                if random.random() <= errorp: circuit=np.matmul(circuit,get_error_matrix(self.bits,errorp))
            circuit = np.matmul(UGATE,circuit)
        if self.verbose: print("\nMeasuring ancillary qubits...")
        q_vec = np.array([0 for i in range(2**self.bits)])
        q_vec[1] = 1
        q_vec = np.matmul(circuit,q_vec)
        collapsed = measure(q_vec)  # Measure ancillary register as part of Shor's algorithm
        states = []
        for i in collapsed: # Convert measurement into a state vector
            if i == "0":
                states.append(np.array([1,0]))
            else:
                states.append(np.array([0,1]))
        collapsed_statevec = states[0].copy()
        for i,entry in enumerate(states):
            if i == 0:
                continue
            else:
                collapsed_statevec = np.kron(collapsed_statevec,entry)
        collapsed_statevec = collapsed_statevec/np.linalg.norm(collapsed_statevec)
        if self.verbose: print("Applying IQFT to working register")
        new_circuit = np.identity(2**self.bits)
        if errorp is not None:
                if random.random() <= errorp: new_circuit=np.matmul(get_error_matrix(self.bits,errorp),new_circuit)
        new_circuit = np.matmul(self.IQFT,new_circuit)
        if errorp is not None:
                if random.random() <= errorp: new_circuit=np.matmul(get_error_matrix(self.bits,errorp),new_circuit)
        final_state = np.matmul(new_circuit,collapsed_statevec)   # Send main register through IQFT
        result = measure(final_state)
        x_register_result = result[:self.main_bitnumber]
        if self.verbose: print("Measured state:",x_register_result)
        int_result = int("".join(x_register_result[::-1]),2)
        phase = int_result/(2**self.main_bitnumber)
        if self.verbose: print("Done!")
        return (phase,False)    # Return False in second argument to flag algorithm was run

    def get_period(self,phase):
        """
        Calculate the resulting period for a given output of Shor's algorithm.
        """
        phase_frac = contfraction(phase)
        max_trials = 10
        if self.verbose: print("Finding period using output of Shor's algorithm")
        searching = True
        while searching:
            expansion = phase_frac.expand(max_trials)
            trial_period = phase_frac.frac.denominator
            if expansion != []:
                for i in range(1,len(expansion)+1):
                    seq = expansion[:i]
                    d,s = phase_frac.contfrac_to_frac(seq)
                    r = abs(phase-d/s)<1/(2*phase_frac.frac.denominator)
                    if s<self.N and r:
                        trial_period = s
                        break
            i=1
            while True:
                period = trial_period*i
                if self.a**period%self.N == 1:
                    searching = False
                    break
                else:
                    i+=1
                if i > max_trials:
                    max_trials += 10
                    if self.verbose: print("Chosen search size was too small, restarting search with larger radius")
                    break
        if self.verbose: print("Done!")
        return period

    def get_factors(self,p):
        p=int(np.ceil(p))
        guesses = [np.gcd(self.a**(p//2)-1, self.N), np.gcd(self.a**(p//2)+1, self.N)]
        return guesses

class contfraction:

    def __init__(self,N):
        self.N = N
        self.frac = fractions.Fraction(N)

    def expand(self,terms):
        n,d = self.frac.numerator,self.frac.denominator
        res = []
        q, r = divmod(n, d)
        while r!= 0:
            res = res + [q]
            prev_r = r
            q, r = divmod(d, r)
            d = prev_r
            if len(res)==terms:
                break
        return res

    @staticmethod
    def contfrac_to_frac(seq):
        """Convert the simple continued fraction in seq into a fraction, num / den"""
        n, d, num, den = 0, 1, 1, 0
        for u in seq:
            n, d, num, den = num, den, num*u + n, den*u + d
        return num, den

def main():
    while True:
        target = 15
        main_register_bitnumber = 3
        #a=7
        J=shor(target,bits=main_register_bitnumber,verbose=True)
        output = J.run_algorithm(errorp=1)
        if output[1]:   # Check if the algorithm was skipped
            print(output[0],"\n")
        else:
            phase = output[0]
            print(phase)
            p = J.get_period(phase)
            print(phase,J.get_factors(p),"\n")
            time.sleep(0.5)
    
if __name__ == "__main__":
    main()
