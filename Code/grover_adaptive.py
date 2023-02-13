import numpy as np
import random
import quantum_backend_GPU as quantum

def get_database():
	database = []
	length = 2**10
	for i in range(length):
		database.append(random.randint(1,800))
	database[random.randint(0,length-1)]=0
	return database

def adaptive_oracle(x,x_0,database):
	Y = database[x_0]
	if database[x] < Y:
		return 1
	else:
		return 0

def main():
	database = get_database()
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
		print("x0:",x_0,database[x_0],"x1:",x_1,database[x_1])
		print(adaptive_oracle(x_1,x_0,database))
		if adaptive_oracle(x_1,x_0,database):
			x_0 = x_1
			fails = 0
		else:
			fails += 1
			if fails == 5:
				searching = False
		m = scaling*m
	print(x_0)
	print(database[x_0])

if __name__=="__main__":
	main()