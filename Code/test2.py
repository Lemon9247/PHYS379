import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import random, time
import quantum_backend_GPU as quantumG
import quantum_backend as quantumC

def main(shots):
	bitlengths = []
	for i in range(14):
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
	for database in databases:
		bits = int(np.ceil(np.log2(len(database))))
		iterations = int(np.ceil(np.sqrt(len(database)/targets)))
		start_time=time.time()
		J = quantumG.Grover(lambda x: f(x,y=database),bits,verbose=False)
		t = J.search(iterations)

		for i in range(shots):
			quantumG.measure(t)
		timesG.append(time.time()-start_time)
		del J
	cp._default_memory_pool.free_all_blocks()
	print("GPU DONE")

	timesC = []
	for database in databases:
		bits = int(np.ceil(np.log2(len(database))))
		iterations = int(np.ceil(np.sqrt(len(database)/targets)))
		start_time=time.time()
		J = quantumC.Grover(lambda x: f(x,y=database),bits,verbose=False)
		t = J.search(iterations)

		for i in range(shots):
			quantumC.measure(t)
		timesC.append(time.time()-start_time)
		del J
	print("CPU DONE")

	Gdata = [bitlengths,timesG]
	Cdata = [bitlengths,timesC]
	print(len(Gdata[0]),len(Gdata[1]))
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(Gdata[0], Gdata[1],color="red")
	ax.plot(Cdata[0], Cdata[1],color="blue")
	plt.show()

if __name__=="__main__":
	main(1024)