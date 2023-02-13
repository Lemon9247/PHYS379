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
	searching = True
	fails = 0
	while searching:
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
			if fails == threshold:
				searching = False
		m = scaling*m
	#print(x_0)
	#print(database[x_0])
	del J
	cp._default_memory_pool.free_all_blocks()
	return x_0

def main():
	shots = 100
	T = []
	bits = 5
	threshold = 10
	database = get_database(bits)
	for i in range(shots):
		t = adaptive_search(database,threshold)
		T.append(t)
		print("Completed {}/{} shots".format(i+1,shots),end="\r",flush=True)
	print("\nDone!")
	readings = []
	freq = []
	for i in range(2**bits):
		readings.append(i)
		freq.append(0)
	for i in T:
		freq[i] += 1
	x = np.array(readings)
	y = np.array(freq)

	plt.scatter(x,y)
	plt.xlabel("Database Register")
	plt.ylabel("Frequency")
	error_rate=100-freq[0]
	print("Error rate: {}%".format(error_rate))
	cp._default_memory_pool.free_all_blocks()
	plt.show()

if __name__=="__main__":
	main()