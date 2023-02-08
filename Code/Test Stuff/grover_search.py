import numpy as np
from gate_extenders import extend_unary
import csv

def trim_csv_data(filename=None,wanted_indices=None):
	"""
	filename = Name of the .csv file (str)
	wanted_indices = The indices of each entry to include in the trimmed dataset (list)
	"""
	with open(filename,newline="") as f:
		data = csv.reader(f,delimiter=",")
		trimmed_data = []	# Create a new list to store the trimmed lines
		for i,line in enumerate(data):
			if i == 0:	# Skip the first line containting the labels of each column
				continue
			trimmed_line = []
			bad_line=False
			for j in wanted_indices:	# Extract the desired entries from this line
				if line[j]=="":	# If there is a desired entry missing from this line, do not include this line
					bad_line=True
					break
				trimmed_line.append(line[j])	# Otherwise, add this entry to the trimmed line
			if bad_line:
				continue
			trimmed_data.append(trimmed_line)	# Append the trimmed line to the new list
	return trimmed_data

def oracle_function(entry):
	mass = entry[1]
	radius = entry[2]
	temperature = entry[3]
	if 0.8<float(mass)<1.9 and 0.5<float(radius)<4.0 and 0<float(temperature)<600:
		return 1
	else:
		return 0

def main():
	trimmed = trim_csv_data("data.csv",[0,2,8,53])
	print(len(trimmed))
	print(trimmed)
	earthlikes = []
	for entry in trimmed:
		if oracle_function(entry)==1:
			earthlikes.append(entry[0])
	print(earthlikes)
	print(len(earthlikes))

if __name__=="__main__":
	main()