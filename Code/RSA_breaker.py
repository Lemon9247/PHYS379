import shor
import RSA
import math,time

def get_private_key(factors,public_key):
	e,n=public_key
	p,q = factors
	s = (p-1)*(q-1)
	g = math.gcd(e, s)
	d = RSA.inverse(e, s)
	return (d,n)

def factorise_modulus(public_key):
	e,n = public_key
	main_register_bitnumber = 5
	while True:
		print("Working...")
		J=shor.shor(n,bits=main_register_bitnumber,verbose=True)
		output = J.run_algorithm()
		if output[1]:   # Check if the algorithm was skipped
			factors=output[0]
			break
		else:
			phase = output[0]	# Get output from Shor's algorithm
			p = J.get_period(phase)	# Compute period
			factors = J.get_factors(p)	# Compute factors
			if 1 in factors:	# Repeat shor's algorithm if resulted in trivial factors
				print("Got bad factors, trying again")
			else:
				break
	return factors

def main():
	while True:
		# Get 3 bit RSA keys
		public, private = RSA.keys(2**3)
		print("Public Key:",public)
		print("Private Key:",private)
		# Find non-trivial prime factors of the modulus
		result = factorise_modulus(public)
		#print(result,result[0]*result[1])
		# Compute private key
		cracked = get_private_key(result,public)
		print("Guessed Private Key:",cracked)
		if cracked == private:
			print("Success")
		else:
			print("Failed")
		print("Restarting in a sec...")
		print()
		time.sleep(2)

if __name__=="__main__":
	main()