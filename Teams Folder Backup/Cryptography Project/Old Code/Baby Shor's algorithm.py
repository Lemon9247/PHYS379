import math
import random
# Given a integer N > 1
print('What is N?')
N = int(input())
# Pick a random number 1 < a < N
a = random.randrange(2, N)
print('a = ' + str(a))
K = math.gcd(a, N)
r = 1
if K != 1:
    print('p = ' + str(K))
    print('q = ' + str(N / K))
    print('These are the non-trivial factors of N used in the RSA algorithm')
else:  # Quantum period-finding sub-routine
    q = math.floor(1 + 2 * math.log(N, 2))  # q is the number of qubits
    Q = 2 ** q  # Ergo N ** 2 <= Q < 2 * N ** 2
    print('The number of qubits is ' + str(q))
    for i in range(1, N):
        if (a ** i) % N == 1:
            r = i
        break
    if r % 2 == 0 and (a ** (r / 2) + 1) % N != 0:
        print('p = ' + str(a ** (r / 2) + 1))
        print('q = ' + str(a ** (r / 2) - 1))
        print('These are the non-trivial factors of N used in the RSA algorithm')
    else:
        while r % 2 != 0 or (a ** (r / 2) + 1) % N == 0:
            a = random.randrange(2, N)
            K = math.gcd(a, N)
            if K != 1:
                print('p = ' + str(K))
                print('q = ' + str(N / K))
                print('These are the non-trivial factors of N used in the RSA algorithm')
                r = 2
                a = 0
            else:  # Quantum period-finding sub-routine
                q = math.floor(1 + math.log(N, 2))  # q is the number of qubits
                Q = 2 ** q  # Ergo N ** 2 <= Q < 2 * N ** 2
                for i in range(1, N):
                    if (a ** i) % N == 1:
                        r = i
                    break
