import random
import math

def inverse(a, m):  # Gives the modular inverse of a modulo m
    for b in range(1, m):
        if (a * b) % m == 1:
            return b
    return -1

def keys(key_size):
    min = 2 ** (key_size - 1)
    max = (2 ** key_size) - 1
    primes = [2]
    start = 1 * 2 ** (key_size // 2 - 1)
    stop = 1 * 2 ** (key_size // 2 + 1)
    if start >= stop:
        return []
    for i in range(3, stop + 1, 2):
        for p in primes:
            if i % p == 0:
                break
        else:
            primes.append(i)  # Building a list of prime numbers up to the stop integer
    while primes and primes[0] < start:  # Getting rid of the prime numbers in the list that aren't large enough
        del primes[0]
    while primes:
        p = random.choice(primes)
        primes.remove(p)
        q_values = [q for q in primes if min <= p * q <= max]
        if q_values:
            q = random.choice(q_values)
            break
    n = p * q
    s = (p - 1) * (q - 1)
    while True:
        e = random.randrange(1, s)
        g = math.gcd(e, s)
        d = inverse(e, s)
        if g == 1 and e != d:
            break
    return (e, n), (d, n)


def encrypt(message, package):
    e, n = package
    cipher = [pow(ord(c), e, n) for c in message]
    return cipher

def decrypt(cipher, package):
    d, n = package
    message = [chr(pow(c, d, n)) for c in cipher]
    return ''.join(message)


def main():
    # Two random large integers p and q for key generation
    bit_length = int(input("Enter bit_length: "))
    while bit_length <= 2:
        print("Invalid bit_length, has to be greater than two")
        bit_length = int(input("Enter new bit_length greater than two: "))

    print("Running RSA...")
    print("Generating public and private keys...")
    public, private = keys(2 ** bit_length)
    print("Public Key: ", public)
    print("Private Key: ", private)

    msg = input("Write message: ")
    print([ord(c) for c in msg])

    encrypted_msg = encrypt(msg, public)
    print("Encrypted message: ")
    print(encrypted_msg)
    print("Decrypted message: ")
    print(decrypt(encrypted_msg, private))


if __name__ == "__main__":
    main()
