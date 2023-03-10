import shor, RSA
import math,time
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# write list to binary file
def write(list_name,file_name):
    # store list in binary file so 'wb' mode
    with open(file_name, 'wb') as fp:
        pickle.dump(list_name, fp)
        print('Done writing list into a binary file')

# Read list to memory
def read(file_name):
    # for reading also binary mode is important
    with open(file_name, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list

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

def test_errorp():
	public = (23,143)
	private = (47,143)
	keys = None # Can alternatively set this to keys=(public,private) to test against a constant key
	y = [[] for i in range(3)]
	errors = [[] for i in range(3)]
	error_size_list = [0.1,0.2,0.3]
	for num,error_size in enumerate(error_size_list):
		print("------------------------------------------")
		print("Error Size = {}".format(error_size))

		bitnumber = 4
		shots = 100
		trials = 5
		errorp_step = 0.5

		errorp_list = np.array([i*errorp_step for i in range(int(1/errorp_step)+1)])
		results = [[] for i in range(int(1/errorp_step)+1)]
		for errorp in errorp_list:
			print("Testing errorp={}".format(errorp))
			for j in range(trials):
				temp = []
				for i in range(shots):
					try:
						result = crack_key(keys=keys,bits=bitnumber)
						if result:
							temp.append(1)
						else:
							temp.append(0)
					except:
						temp.append(0)
					print("Completed {}/{} shots, {}/{} trials".format(i+1,shots,j,trials),end="\r",flush=True)
				success = np.mean(temp)
				results[int(errorp/errorp_step)].append(success)
			print("Completed {}/{} shots, {}/{} trials".format(i+1,shots,trials,trials),end="\r",flush=True)
			print("\n")
		for result_list in results:
			y[num].append(np.mean(result_list))
			errors[num].append(np.std(result_list)/np.sqrt(trials))

	folder = str(int(np.floor(time.time())))+"/"
	os.mkdir(folder)
	write(y,folder+"y")
	write(errors,folder+"errors")
	

def plot_data(y_file,err_file):
	y = read(y_file)
	errors  = read(err_file)
	error_size_list = [0.1,0.2,0.3]
	
	
	bitnumber = 4
	shots = 100
	trials = 5
	errorp_step = 0.5

	errorp_list = np.array([i*errorp_step for i in range(int(1/errorp_step)+1)])
	
	shapes = ["o","v","*"]
	b = [-0.075,0,0.075]
	#print(results)
	fig,ax=plt.subplots()
	for i in range(3):
		ax.errorbar(errorp_list+b[i], y[i], yerr=errors[i], fmt=shapes[i], ecolor="gray", elinewidth=0.75, capsize=3, label=str(error_size_list[i]))
	plt.xticks(errorp_list)
	plt.xlabel("Probability of error on a qubit")
	plt.ylabel("Success Probability")
	plt.legend(title="Error Size")
	plt.title("""Measured probability of successfully
breaking 8-bit RSA encryption for different values
of the error probability,
measured over {} trials of {} shots.
Working Register = {} Qubits""".format(trials,shots,bitnumber)
)
	plt.tight_layout()
	plt.show()

if __name__=="__main__":
	#main()
	#test_errorp()
	plot_data("1678447330/y","1678447330/errors")
	#while True:
	#	crack_key(verbose=True)