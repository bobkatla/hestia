import json

with open('data.json','r')as infile:
	data = json.load(infile)
	
print(len(data))
for i in range(len(data)):
	data[i]= data[i][0]
	
with open('data_measure_sitting.json','w') as outfile:
	json.dump(data,outfile)
