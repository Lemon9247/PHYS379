import shor
import numpy as np
import matplotlib.pyplot as plt

def test_factorising_success():
	target = 15
	min_bitnumber = 3
	max_bitnumber = 6
	a = 7
	shots = 100
	trials = 100
	bit_sizes = [i for i in range(min_bitnumber,max_bitnumber+1)]
	results = [[] for i in range(min_bitnumber,max_bitnumber+1)]
	for bitnumber in bit_sizes:
		print("Testing {} working bits".format(bitnumber))
		J=shor.shor(target,a=a,bits=bitnumber,verbose=False)
		for j in range(trials):
			temp = []
			for i in range(shots):
				try:
					phase = J.run_algorithm(errorp=0)[0]
					p = J.get_period(phase)
					result = J.get_factors(p)
					if result == [3,5] or result == [5,3]:
						temp.append(1)
					else:
						temp.append(0)
				except:
					temp.append(0)
				print("Completed {}/{} shots, {}/{} trials".format(i+1,shots,j,trials),end="\r",flush=True)
			success = np.mean(temp)
			results[bitnumber-min_bitnumber].append(success)
		print("\n")
	y = []
	errors = []
	for result_list in results:
		y.append(np.mean(result_list))
		errors.append(np.std(result_list)/np.sqrt(trials))
	print(results)
	fig,ax=plt.subplots()
	ax.errorbar(bit_sizes, y, yerr=errors, fmt="o", ecolor="gray", elinewidth=0.75, capsize=3)
	plt.xticks(bit_sizes)
	plt.xlabel("Number of Working Bits")
	plt.ylabel("Success Probability")
	plt.title("""Experimental probability of successfully factorising 15 into [3,5]
for different sizes of the working qubit register,
measured over {} trials of {} shots""".format(trials,shots)
	)
	plt.show()


def test_errorp():
	y = [[] for i in range(3)]
	errors = [[] for i in range(3)]
	error_size_list = [0.1,0.2,0.3]
	for num,error_size in enumerate(error_size_list):
		print("------------------------------------------")
		print("Error Size = {}".format(error_size))
		target = 15
		bitnumber = 4
		a = 7
		shots = 100
		trials = 20
		errorp_step = 0.2
		errorp_list = np.array([i*errorp_step for i in range(int(1/errorp_step)+1)])
		results = [[] for i in range(int(1/errorp_step)+1)]
		for errorp in errorp_list:
			print("Testing errorp={}".format(errorp))
			J=shor.shor(target,a=a,bits=bitnumber,verbose=False)
			for j in range(trials):
				temp = []
				for i in range(shots):
					try:
						phase = J.run_algorithm(errorp=errorp,error_size=error_size)[0]
						p = J.get_period(phase)
						result = J.get_factors(p)
						if result == [3,5] or result == [5,3]:
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

	b = [-0.01,0,0.01]
	shapes = ["o","v","*"]
	#print(results)
	fig,ax=plt.subplots()
	for i in range(3):
		ax.errorbar(errorp_list+b[i], y[i], yerr=errors[i], fmt=shapes[i], ecolor="gray", elinewidth=0.75, capsize=3, label=str(error_size_list[i]))
	plt.xticks(errorp_list)
	plt.xlabel("Probability of error on a qubit")
	plt.ylabel("Success Probability")
	plt.legend(title="Error Size")
	plt.title("""Experimental probability of successfully factorising {}
for different values of the error probability,
measured over {} trials of {} shots.
Working Register = {} Qubits""".format(target,trials,shots,bitnumber)
)
	plt.tight_layout()
	plt.show()


def test_shor_outputs():
	target = 15
	main_register_bitnumber = 4
	a = 7
	shots = 500
	trials = 30
	errorp=1
	phase_list = [i/(2**main_register_bitnumber) for i in range(2**main_register_bitnumber)]
	outputs = [[0 for i in range(2**main_register_bitnumber)] for i in range(trials)]
	for trial_num in range(trials):
		for i in range(shots):
			J=shor.shor(target,a=a,bits=main_register_bitnumber,verbose=False)
			phase = J.run_algorithm(errorp=errorp)[0]
			print("Completed {}/{} shots, {}/{} trials".format(i+1,shots,trial_num,trials),end="\r",flush=True)
			for i,element in enumerate(phase_list):
				if element == phase:
					outputs[trial_num][i]+=1
	print()
	freq = np.array(outputs).T
	y,errors = [],[]
	for freq_list in freq:
		y.append(np.mean(freq_list)/shots)
		errors.append(np.std(freq_list)/shots/np.sqrt(shots*trials))
	fig,ax=plt.subplots()
	ax.errorbar(phase_list,y,yerr=errors, ecolor="gray", elinewidth=0.75, capsize=3)
	ax.set_ylim([min(y)-0.2*min(y), max(y)+0.2*max(y)])
	plt.xlabel("Output x/(2^L) of Shor's Algorithm")
	plt.ylabel("Measured Probability")
	plt.title("""L={} bits, tested over {} trials of {} shots, errorp = {}""".format(main_register_bitnumber,trials,shots,errorp))
	plt.show()

if __name__=="__main__":
	#test_factorising_success()
	test_errorp()
	#test_shor_outputs()
	