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
			bad_line = False
			for j in wanted_indices:	# Extract the desired entries from this line
				if line[j] == "":	# If there is a desired entry missing in this line, do not include this line
					bad_line=True
					break
				trimmed_line.append(line[j])	# Otherwise, add this entry to the trimmed line
			if bad_line:
				continue
			trimmed_data.append(trimmed_line)	# Append the trimmed line to the new list
		return trimmed_data

def main():
	trimmed = trim_csv_data("data.csv",[0,2]) # Extract 1st and 3rd entry from each line (name and mass of planet)
	with open("trimmed_data.csv","w") as g:
		for line in trimmed:
			g.write(line[0]+","+str(line[1])+"\n")

if __name__=="__main__":
	main()