import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import random, time

def main():
	thingy=input("Use GPU? Y/N\n\t>").strip().lower()
	if thingy == "y":
		GPU = True
	else:
		GPU = False
	if GPU:
		import quantum_backend_GPU as quantum
	else:
		import quantum_backend_GPU as quantum

	database = []
	length = 2**5
	for i in range(length):
		database.append(0)
	targets=1
	for i in range(targets):
		database[0]=1

	def f(x):
		if database[x] == 1:
			return True
		else:
			return False
			
	print(len(database))
	bits = int(np.ceil(np.log2(len(database))))
	iterations = int(np.ceil(np.sqrt(len(database)/targets)))
	print(iterations)
	start_time=time.time()
	J = quantum.Grover(f,bits,verbose=True)
	T=[]
	shots = 1024
	t = J.search(iterations,errorp=0.2)
	print("Starting shots!")
	for i in range(shots):
		T.append(quantum.measure(t))
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
	end_time=time.time()
	print("Runtime: {}s".format(end_time-start_time))
	del J
	if GPU: cp._default_memory_pool.free_all_blocks()
	plt.show()

if __name__=="__main__":
	main()