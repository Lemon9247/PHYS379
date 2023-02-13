import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import random, time
import quantum_backend_GPU as quantumG
import quantum_backend as quantumC

def get_database():
	database = []
	length = 2**2
	for i in range(length):
		database.append(random.randint(1,800))
	targets=1
	for i in range(targets):
		database[random.randint(0,length-1)]=0
	return database

def main():


if __name__=="__main__":
	main()