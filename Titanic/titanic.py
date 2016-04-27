import csv as csv
import numpy as np

csv_file = csv.reader(open('train.csv', 'rb'))
header = csv_file.next() #skip first line

data = []
for row in csv_file:
	data.append(row)

data = np.array(data)

num_pass = np.size(data[0::,1].astype(np.float))
num_sur = np.sum(data[0::,1].astype(np.float))
sur_ratio = num_sur/num_pass

female_only = data[0::,4] == "female"
male_only = data[0::,4] != "female"

women_onboard = data[fem_only, 1].astype(np.float)
men_onboard = data[male_only,1].astype(np.float)


male_sur_ratio = np.sum(men_onboard) / np.size(men_onboard)
female_sur_ratio = np.sum(women_onboard) / np.size(women_onboard)

print " Proportion of women who survived is %s" % female_sur_ratio
print " Proportion of men who survived is %s" % male_sur_ratio


#TEST FILE
test_file = open('test.csv', 'rb')
test_obj = csv.reader(test_file)

header = test_obj.next()

pred_file = open('genderbasemodel.csv', 'wb')
pred_obj = csv.writer(pred_file)


pred_obj.wwriterow(["PassengerId", "Survived"])
for row in test_obj:
	if row[3] == 'female':
		pred_obj.writerow([row[0], '1'])
	else:
		pred_obj.writerow([row[0],'0'])
test_file.close()
pred_file.close()


