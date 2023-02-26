import numpy as no
import matplotlib.pyplot as plt
import csv

with open("100shots_1trials_5bits.csv",newline="\n") as file:
	data = csv.reader(file,delimiter=",")
	x,y,yerr=[],[],[]
	for i, line in enumerate(data):
		if i!=0:
			x.append(line[0])
			y.append(line[1])
			yerr.append(line[2])

plt.errorbar(x,y,yerr=yerr,fmt="o")
plt.show()
