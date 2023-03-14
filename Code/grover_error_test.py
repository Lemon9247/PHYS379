import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import random, time
import quantum_backend_GPU as quantum


def main():
	y = [[] for i in range(3)]
	errors = [[] for i in range(3)]
	error_size_list = [0.01,0.05,0.1]

	database = []
	bits = 5
	length = 2**bits
	for i in range(length):
		database.append(0)
	targets=1
	for i in range(targets):
		database[0]=1
	iterations = int(np.ceil(np.sqrt(len(database)/targets)))

	def f(x):
		if database[x] == 1:
			return True
		else:
			return False

	for num,error_size in enumerate(error_size_list):
		print("------------------------------------------")
		print("Error Size = {}".format(error_size))
		shots = 100
		trials = 20
		errorp_step = 0.2
		errorp_list = np.array([i*errorp_step for i in range(int(1/errorp_step)+1)])
		results = [[] for i in range(int(1/errorp_step)+1)]
		for errorp in errorp_list:
			print("Testing errorp={}".format(errorp))
			for j in range(trials):
				temp = []
				for i in range(shots):
					try:
						J = quantum.Grover(f,bits,verbose=False)
						t = J.search(iterations,errorp=errorp,error_size=error_size)
						result = quantum.measure(t)
						if result == 0:
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

	b = [-0.05,0,0.05]
	shapes = ["o","v","*"]
	#print(results)
	fig,ax=plt.subplots()
	for i in range(3):
		ax.errorbar(errorp_list+b[i], y[i], yerr=errors[i], fmt=shapes[i], ecolor="gray", elinewidth=0.75, capsize=3, label=str(error_size_list[i]))
	plt.xticks(errorp_list)
	plt.xlabel("Probability of error on a qubit")
	plt.ylabel("Success Probability")
	plt.legend(title="Error Size")
	plt.title("""Experimental probability of successfully finding the target
measured over {} trials of {} shots.
Working Register = {} Qubits""".format(trials,shots,bits)
)
	plt.tight_layout()
	plt.show()


if __name__=="__main__":
	main()

