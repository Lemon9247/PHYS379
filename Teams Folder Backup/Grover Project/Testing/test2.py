import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import random, time
import quantum_backend_GPU as quantumG
import quantum_backend as quantumC

def main(shots):
	bitlengths = []
	for i in range(12):
		bitlengths.append(i+1)
	print(bitlengths)

	databases = []
	for i in bitlengths:
		database = []
		length = 2**i
		for j in range(length):
			database.append(0)
		targets=1
		for j in range(targets):
			database[0]=1
		databases.append(database)

	def f(x,y):
		if y[x] == 1:
			return True
		else:
			return False

	timesG = []
	errG = []
	for database in databases:
		bits = int(np.ceil(np.log2(len(database))))
		iterations = int(np.ceil(np.sqrt(len(database)/targets)))
		times = []
		for i in range(shots):
			start_time=time.time()
			J = quantumG.Grover(lambda x: f(x,y=database),bits,verbose=False)
			t = J.search(iterations)
			times.append(time.time()-start_time)
		timesG.append(np.mean(np.array(times)))
		errG.append(np.std(np.array(times)))
		del J
	cp._default_memory_pool.free_all_blocks()
	print("GPU DONE")

	timesC = []
	errC = []
	for database in databases:
		bits = int(np.ceil(np.log2(len(database))))
		iterations = int(np.ceil(np.sqrt(len(database)/targets)))
		times = []
		for i in range(shots):
			start_time=time.time()
			J = quantumC.Grover(lambda x: f(x,y=database),bits,verbose=False)
			t = J.search(iterations)
			times.append(time.time()-start_time)
		timesC.append(np.mean(np.array(times)))
		errC.append(np.std(np.array(times)))
		del J
	print("CPU DONE")

	Gdata = [bitlengths,timesG]
	Cdata = [bitlengths,timesC]
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.errorbar(
		Gdata[0], Gdata[1], yerr=errG, 
		fmt="o", color="red", ecolor='gray', 
		elinewidth=0.75, capsize=3, label = "GPU - RTX 3060, 6GB GDDR6 VRAM @ 1750 MHz")
	ax.errorbar(
		Cdata[0], Cdata[1], yerr=errC,
		fmt="s", color="blue", ecolor='gray', 
		elinewidth=0.75, capsize=3, label = "CPU - Ryzen 7 5800H, 16GB DDR4 RAM @ 3200 MHz")
	plt.xlabel("Number of qubits")
	plt.ylabel("Average Time for Grover Search, s")
	plt.title("Average time to complete a Grover Search of "+u"\u221A"+"N iterations for n=log2(N) qubits")
	ax.legend()
	plt.show()

if __name__=="__main__":
	main(10)