import numpy as np
import cupy as cp
import random
import quantum_backend_GPU as quantum
import matplotlib.pyplot as plt

def get_database(bits):
	database = []
	length = 2**bits
	for i in range(length):
		database.append(random.randint(1,800))
	database[0]=0
	return database

def adaptive_oracle(x,x_0,database):
	Y = database[x_0]
	if database[x] < Y:
		return 1
	else:
		return 0

def adaptive_search(database,threshold):
	bits = int(np.ceil(np.log2(len(database))))
	x_0 = random.randint(0,len(database)-1)
	scaling = 1.34
	m = 1
	fails = 0
	while fails < threshold:
		iterations = random.randint(1,np.ceil(m))
		J = quantum.Grover(lambda x: adaptive_oracle(x,x_0,database),bits)
		q = J.search(iterations)
		x_1 = quantum.measure(q)
		#print("x0:",x_0,database[x_0],"x1:",x_1,database[x_1])
		#print(adaptive_oracle(x_1,x_0,database))
		if adaptive_oracle(x_1,x_0,database):
			x_0 = x_1
			fails = 0
		else:
			fails += 1
		m = scaling*m
	#print(x_0)
	#print(database[x_0])
	del J
	cp._default_memory_pool.free_all_blocks()
	return x_0

def main():
	shots = 100
	bigshots = 10
	outputs = [[] for i in range(bigshots)]
	bits = 10
	threshold = 5
	database = get_database(bits)
	for j in range(bigshots):
		for i in range(shots):
			t = adaptive_search(database,threshold)
			outputs[j].append(t)
			print("Completed {}/{} shots, {}/{} bigshots".format(i+1,shots,j,bigshots),end="\r",flush=True)
	print("Completed {}/{} shots, {}/{} bigshots".format(shots,shots,bigshots,bigshots),end="\r",flush=True)	
	print("\nDone!")
	
	freq = [[0 for i in range(bigshots)] for j in range(2**bits)]
	for i in range(len(outputs)):
		for j in outputs[i]:
			freq[j][i] += 1
	x = np.array([i for i in range(2**bits)])
	y,errors = [],[]
	for freq_list in freq:
		y.append(np.mean(freq_list))
		errors.append(np.std(freq_list))

	plt.errorbar(x, y, yerr=errors, fmt="o", ecolor='gray', elinewidth=0.75, capsize=3)
	plt.xlabel("Database Register")
	plt.ylabel("Frequency")
	plt.title(
		"Frequency plot of average output of the Durr-Hoyer Algorithm for {} trials of {} shots".format(bigshots,shots)
		)
	fail_rate=100-y[0]
	fail_rate_error=errors[0]
	print("Fail rate: ({} +/- {})%".format(fail_rate,fail_rate_error))
	cp._default_memory_pool.free_all_blocks()
	plt.show()

if __name__=="__main__":
	main()