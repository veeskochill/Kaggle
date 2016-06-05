import csv as csv
import numpy as np

csv_file = csv.reader(open('train.csv', 'rb'))
header = csv_file.next() #skip first line

data = []
for row in csv_file:
	data.append(row)

data = np.array(data)

fare_ceiling = 40

data[data[0::,9].astype(np.float) >=fare_ceiling, 9] = fare_ceiling - 1.0
adata = data

fare_bracket_size = 10
num_price_brackets = fare_ceiling / fare_bracket_size

num_classes = 3
num_classes = len(np.unique(data[0::,2]))


survival_table = np.zeros((2, num_classes, num_price_brackets))

for i in xrange(num_classes):
    for j in xrange(num_price_brackets):
            women_only_stats = adata[                          #Which element           
                         (adata[0::,4] == "female")    #is a female
                       &(adata[0::,2].astype(np.float) #and was ith class
                             == i+1)                        
                       &(adata[0:,9].astype(np.float)  #was greater 
                            >= j*fare_bracket_size)   #than this bin              
                       &(adata[0:,9].astype(np.float)  #and less than
                            < (j+1)*fare_bracket_size)#the next bin    
                          , 1]  
            men_only_stats = adata[                          #Which element           
                         (adata[0::,4] == "male")    #is a male
                       &(adata[0::,2].astype(np.float) #and was ith class
                             == i+1)                        
                       &(adata[0:,9].astype(np.float)  #was greater 
                            >= j*fare_bracket_size)   #than this bin              
                       &(adata[0:,9].astype(np.float)  #and less than
                            < (j+1)*fare_bracket_size)#the next bin    
                          , 1]
            survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float)) 
            survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))

survival_table[ survival_table != survival_table ] = 0.


survival_table[ survival_table < 0.5 ] = 0
survival_table[ survival_table >= 0.5 ] = 1 


test_file = open('test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()
predictions_file = open("genderclassmodel.csv", "wb")
p = csv.writer(predictions_file)
p.writerow(["PassengerId", "Survived"])


for row in test_file_object:
	for j in xrange(num_price_brackets):
		try:
			row[8] = float(row[8])
		except:
			bin_fare = 3 - float(row[1])
			break

		if row[8] > fare_ceiling:
			bin_fare = num_price_brackets-1
			break
		if row[8] >= j * fare_bracket_size and row[8] < (j+1) * fare_bracket_size:
			bin_fare = j
			break
	if row[3] == 'female':
		p.writerow([row[0], "%d" % int(survival_table[0, float(row[1])- 1, bin_fare])])
	else:
		p.writerow([row[0], "%d" % int(survival_table[1, float(row[1])-1, bin_fare])])

test_file.close()
predictions_file.close()

