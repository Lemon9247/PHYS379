import shor, RSA
import math,time
import numpy as np
import matplotlib.pyplot as plt

def get_private_key(factors,public_key):
	e,n=public_key
	p,q = factors
	s = (p-1)*(q-1)
	g = math.gcd(e, s)
	d = RSA.inverse(e, s)
	return (d,n)

def factorise_modulus(public_key,bits,verbose=None):
	verbose = verbose if verbose != None else False
	e,n = public_key
	main_register_bitnumber = bits if bits!= None else 5
	if verbose: print("Working...")
	J=shor.shor(n,bits=main_register_bitnumber,verbose=verbose)
	output = J.run_algorithm()
	if output[1]:   # Check if the algorithm was skipped
		factors=output[0]
		#break
	else:
		phase = output[0]	# Get output from Shor's algorithm
		p = J.get_period(phase)	# Compute period
		factors = J.get_factors(p)	# Compute factors
		# if 1 in factors:	# Repeat shor's algorithm if resulted in trivial factors
		# 	print("Got bad factors, trying again")
		# else:
		# 	break
	return factors

def crack_key(keys=None,verbose=None,bits=None):
	verbose = verbose if verbose != None else False
	working_bits = bits if bits != None else 5
	# Get 3 bit RSA keys
	public, private = keys if keys!= None else RSA.keys(2**3)
	if verbose: print("Public Key:",public)
	if verbose: print("Private Key:",private)
	# Find non-trivial prime factors of the modulus
	result = factorise_modulus(public,working_bits,verbose=verbose)
	#print(result,result[0]*result[1])
	# Compute private key
	cracked = get_private_key(result,public)
	if verbose: print("Guessed Private Key:",cracked)
	if cracked == private:
		if verbose: print("Success")
		success = True
	else:
		if verbose: print("Failed")
		success = False
	return success

def main():
	public = (23,143)
	private = (47,143)
	keys = None # Can alternatively set this to keys=(public,private) to test against a constant key
	shots = 100
	maximum_bitnumber = 5 # 8-bit RSA requires an 8 qubit ancillary register, so limit this to 5 at max
	bit_sizes = [i+1 for i in range(maximum_bitnumber)]
	results = [[] for i in range(maximum_bitnumber)]
	for bitnumber in bit_sizes:
		print("Testing {} working bits".format(bitnumber))
		#print("Public Key:",public)
		#print("Private Key:",private)
		for i in range(shots):
			try:
				result = crack_key(keys=keys,bits=bitnumber)
				if result:
					results[bitnumber-1].append(1)
				else:
					results[bitnumber-1].append(0)
			except:
				results[bitnumber-1].append(0)
			print("Completed {}/{} shots".format(i+1,shots),end="\r",flush=True)
		print("\n")
	y = []
	errors = []
	for result_list in results:
		y.append(np.mean(result_list))
		errors.append(np.std(result_list)/np.sqrt(shots))
	print(results)
	fig,ax=plt.subplots()
	ax.errorbar(bit_sizes, y, yerr=errors, fmt="o", ecolor="gray", elinewidth=0.75, capsize=3)
	plt.xticks(bit_sizes)
	ax.set_ylim([0, 1])
	plt.xlabel("Number of Working Bits")
	plt.ylabel("Success Probability")
	plt.title("""Measured probability of successfully finding the RSA private key from a ramdom 3-bit public key
for different sizes of the working register size for {} shots""".format(shots)
		)
# 	plt.title(
# 		"""Measured probability of successfully cracking the RSA key e={},d={},n={}
# for different sizes of the working register size for {} shots""".format(public[0],private[0],public[1],shots)
# 		)	# Use this plot title if working with a fixed public, private key pairing!
	plt.show()



if __name__=="__main__":
	main()
	#while True:
	#	crack_key(verbose=True)