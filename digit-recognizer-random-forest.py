# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

print('Loading training set...')
dataset = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
y_test = test_data.values[:, :]
X_train = dataset.values[:, 1:]
y_train = dataset.values[:, 0]

print('Start Learning...')
recognizer = RandomForestClassifier(n_estimators = 150, criterion = 'entropy')
recognizer.fit(X_train, y_train)

y_pred = recognizer.predict(y_test) #answer of test data :) :) 



    
df = pd.DataFrame({'ImageId': pd.Series(range(1, 28001)) , 'Label': y_pred})
df.to_csv('Output.csv', index = False)
