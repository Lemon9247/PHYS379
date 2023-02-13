import random

def oracle(guess):
	if 0.5 < guess < 5:
		return 1
	else:
		return 0

thingy = []
for i in range(10):
	thingy.append(random.randint(1,5)*random.random())
random.shuffle(thingy)
print(thingy)

fails = 1
for item in thingy:
	if oracle(item):
		print("Found",item,"after",fails,"checks")
		fails += 1
	else:
		fails += 1

