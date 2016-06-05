import pandas as pd
import numpy as np
import csv as csv


# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 


# For .read_csv, always use header=0 when you know row 0 is the header row
train_data = pd.read_csv('train.csv', header=0)
train_data['Fare'].fillna(train_data.mean()['Fare'])
train_data.Age = train_data.Age.fillna(train_data.Age.mean())

train_data.Sex = train_data.Sex.map({"male":0, "female" :1})
del train_data['Cabin']
del train_data['Ticket']
del train_data['Embarked']
del train_data['Name']

test_data = pd.read_csv('test.csv', header=0)
test_data.Fare = test_data.Fare.fillna(test_data.mean()['Fare'])
test_data.Age = test_data.Age.fillna(test_data.mean()['Age'])

test_data.Sex = test_data.Sex.map({"male":0, "female" :1})
del test_data['Cabin']
del test_data['Ticket']
del test_data['Embarked']
del test_data['Name']


# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

#print train_data.drop(train_data.columns[1], axis =1)
# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data.drop(train_data.columns[1], axis =1),train_data.Survived)

# Take the same decision trees and run it on the test data
output = forest.predict(test_data)

predictions_file = open("rfmodel.csv", "wb")
p = csv.writer(predictions_file)
p.writerow(["PassengerId", "Survived"])
for uid, result in enumerate(output):
	p.writerow([test_data.PassengerId[uid], result])
#	print train_data.PassengerId[uid], result

predictions_file.close()
