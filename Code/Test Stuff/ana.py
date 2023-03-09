import numpy as np
import matplotlib.pyplot as plt
import random, csv
import quantum_backend as quantum


def get_database(bits):
	database = []
	length = 2**bits
	for i in range(length):
		database.append(random.randint(1,800))
	database[0]=801
	return database

def import_ESI_data():
	with open("esi.csv",newline="") as file:
		data_reader = csv.reader(file,delimiter=",")
		data = [row for row in data_reader]
		del data[0]		# Remove Names/ESI line at the start of the file
		data_length = len(data)
		bits = int(np.ceil(np.log2(data_length)))
		entries = 2**bits
		for i in range(entries-data_length):	# Ensure the database length is a power of 2
			data.append(None)
	return data

def adaptive_oracle(x,x_0,database):
	Y = database[x_0][1]
	try:
		if database[x][1] > Y:
			return True
		else:
			return False
	except:
		return False

def adaptive_oracle2(x,x_0,database):
	Y = database[x_0]
	try:
		if database[x] > Y:
			return True
		else:
			return False
	except:
		return False

def adaptive_search(database,min_val):
	bits = int(np.ceil(np.log2(len(database))))
	while True:
		x_0 = random.randint(0,len(database)-1)
		if database[x_0] is not None:
			break
	scaling = 1.34
	m = 1
	fails = 0
	while database[x_0]<=min_val:
		iterations = random.randint(1,np.ceil(m))
		J = quantum.Grover(lambda x: adaptive_oracle2(x,x_0,database),bits)
		q = J.search(iterations,errorp=1) #this is here where I change the errorp, try putting errorp=0
		x_1 = quantum.measure(q)
		if adaptive_oracle2(x_1,x_0,database):
			x_0 = x_1
		m = min(scaling*m,np.sqrt(2**bits))
	#del J   
#del J was here
	return x_0

def multi_trial_durr_hoyer(shots,trials,database,min_val):
	outputs = [[] for i in range(trials)]
	bits = int(np.ceil(np.log2(len(database))))
	for j in range(trials):
		for i in range(shots):
			t = adaptive_search(database,min_val)
			outputs[j].append(t)
			print("Completed {}/{} shots, {}/{} trials".format(i+1,shots,j,trials),end="\r",flush=True)
	print("Completed {}/{} shots, {}/{} trials".format(shots,shots,trials,trials),end="\r",flush=True)	
	print("\nDone!")
	freq = [[0 for i in range(trials)] for j in range(2**bits)]
	for i in range(len(outputs)):
		for j in outputs[i]:
			freq[j][i] += 1
	return freq

def main1(shots,trials,bits):
	min_val = 799 #here is where i change the range of values... if you equal max_val and min_val it should work as
	database = get_database(bits)

	freq = multi_trial_durr_hoyer(shots,trials,database,min_val)
	x = np.array([i for i in range(2**bits)])
	y,errors = [],[]
	for freq_list in freq:
		y.append(np.mean(freq_list))
		errors.append(np.std(freq_list))

	plt.errorbar(x, y, yerr=errors, fmt="o", ecolor='gray', elinewidth=0.75, capsize=3)
	plt.xlabel("Database Register")
	plt.ylabel("Frequency")
	#plt.title(
	#	"""Frequency plot of average output of the Durr-Hoyer Algorithm
#for {} trials of {} shots""".format(trials,shots)
	#	)
	plt.title(
		"""Frequency plot of the Durr-Hoyer Algorithm for {} shots
when applied to database with no threshold but a range of values (1,9). Error probability = 0.9""".format(shots)
		)
	plt.show()

def main2(shots,trials,bits):
	thresholds = [i for i in range(1,10+1)]
	database = get_database(bits)

	fail_rates, fail_rate_errors = [],[]
	for threshold in thresholds:
		print("Threshold:",threshold)
		freq = multi_trial_durr_hoyer(shots,trials,database,max_val,min_val)
		
		fail=100-np.mean(freq[0])
		fail_error=np.std(freq[0])
		print("Fail rate: ({} +/- {})%".format(fail,fail_error))
		fail_rates.append(fail)
		fail_rate_errors.append(fail_error)
	plt.errorbar(thresholds, fail_rates, yerr=fail_rate_errors, fmt="o", ecolor='gray', elinewidth=0.75, capsize=3)
	plt.xlabel("Termination Threshold")
	plt.ylabel("Failed Searches")
	plt.title(
		"""Mean number of failed shots of the Durr-Hoyer Algorithm
for {} trials of {} shots for different termination thresholds with {} qubits""".format(trials,shots,bits)
		)
	plt.show()

if __name__=="__main__":
	main1(shots=100,trials=1,bits=3)