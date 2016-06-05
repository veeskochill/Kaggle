import pandas as pd
import numpy as np
import csv as csv

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('train.csv', header=0)

df.fare_limited = df.Fare
fare_ceiling = 40
df.fare_limited = df.Fare.map(lambda x : fare_ceiling-1 if x > fare_ceiling else x)

fare_bracket_size = 10
num_price_brackets = fare_ceiling / fare_bracket_size

num_classes = 3

#surviaval_df = pd.DataFrame(columns = "")

survival_table = np.zeros((2, num_classes, num_price_brackets))

df.gender = df.Sex.map({"male":0, "female" :1})
df.price_bracket = df.fare_limited.map(lambda x : int(x/fare_bracket_size))

for i in xrange(num_classes):
	for j in xrange(num_price_brackets):
		women_only = (df.loc[(df.gender == 1) & (df.price_bracket == j) & (df.Pclass == i)])
		print women_only.Survived.mean()
		men_only = (df.loc[(df.gender == 0) & (df.price_bracket == j)&(df.Pclass == i)])
		survival_table[0,i,j] = women_only.Survived.mean()#.astype(np.float)) 
		survival_table[1,i,j] = men_only.Survived.mean()#.astype(np.float))

survival_table[ survival_table != survival_table ] = 0.

survival_table[ survival_table < 0.5 ] = 0
survival_table[ survival_table >= 0.5 ] = 1 

test_file = open('test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()
predictions_file = open("pandas_genderclassmodel.csv", "wb")
p = csv.writer(predictions_file)
p.writerow(["PassengerId", "Survived"])

testdf = pd.read_csv('test.csv', header=0)
#print testdf
testdf['Fare'].fillna(testdf.mean()['Fare'])
testdf.Survived = testdf.Fare
testdf.bin_fare = pd.Series([((fare_ceiling-1)/fare_bracket_size) if fare > fare_ceiling else (fare/fare_bracket_size) for fare in testdf.Fare])

testdf.gender = testdf.Sex
testdf.gender = testdf.Sex.map({"male":0, "female" :1})
testdf.bin_fare = testdf.bin_fare.map(lambda x : int(0) if np.isnan(x) else int(x))


predictions_file = open("pandasclassmodel.csv", "wb")
p = csv.writer(predictions_file)
p.writerow(["PassengerId", "Survived"])

for row in range(len(testdf)):
	p.writerow([
		testdf.PassengerId[row], 
		"%d" % int(
			survival_table
			[
			testdf.gender[row], 
			testdf.Pclass[row]-1, 
			testdf.bin_fare[row]
			])])


#print testdf.Survived
#p.writerow([row[0], "%d" % int(survival_table[0, float(row[1])- 1, bin_fare])])

test_file.close()
predictions_file.close()