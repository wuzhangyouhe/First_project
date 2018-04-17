from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import pylab as pl
import pandas as pd
import csv
from csv import reader
from sklearn.metrics import accuracy_score
from sklearn import datasets, linear_model, neighbors
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
from sklearn.metrics import mean_squared_error, r2_score


# Load a CSV file
def load_csv(filename):
    file = open(filename, "rU")
    lines = reader(file)
    dataset = list(lines)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# cpu_data = "mobile_cpu.csv"
# cpu_price = "mobile_price.csv"
feature_names = ['lithography', 'nbCores', 'nbThreads', 'PBF', 'Cache', 'TDP', 'MMS', 'MMC', 'GBF', 'Theta0']
target_name = ['Recommended_Price']
cpu_data = "all_CPU_dataset.csv"
cpu_price = "all_CPU_price.csv"


def read_lines():
    with open(cpu_data, 'rU') as data:
        data_x = csv.reader(data, delimiter=",")
        for row in data_x:
            yield [float(i) for i in row]


# get cpu_data_array
cpu_data_array = []
for i in read_lines():
    cpu_data_array.extend([i])

# print cpu_data_array
for i in cpu_data_array:
    print i

# get cpu_price_array
cpu_price_array = load_csv(cpu_price)
for j in range(len(cpu_price_array[0])):
    str_column_to_float(cpu_price_array, j)

# print cpu_price_array
for i in cpu_price_array:
    print i

x = np.array([np.concatenate((v, [1])) for v in cpu_data_array])
y = cpu_price_array

# Create linear regression object
regr = linear_model.LinearRegression()
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
# Train the model using the training sets
regr.fit(x, y)
knn.fit(x, y)

print knn.predict(x)
# Compute RMSE on training data
p = regr.predict(x)
pk = knn.predict(x)

# Now we can constuct a vector of errors
err = abs(p - y)
sum_err = np.sum(err)

mae = mean_absolute_error(p, y)
print 'Mean Absolute Error: \n', mae

mae_knn = mean_absolute_error(pk, y)
print 'KNN MAE: \n', mae_knn
# Let's see the error on the first 10 predictions
#print err[:10]

# Dot product of error vector with itself gives us the sum of squared errors
# Compute RMSE
rmse_train = np.sqrt(np.sum((p-y)**2)/len(p))
rmse_knn = np.sqrt(np.sum((pk-y)**2)/len(pk))
#rmse_train = mean_squared_error(p, y)
print 'Root Mean Square Error: \n', rmse_train
print 'KNN Root Mean Square Error: \n', rmse_knn
# We can view the regression coefficients
#print 'Regression Coefficients: \n', pd.DataFrame(regr.coef_, columns=feature_names)


def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] is predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

accuracy = getAccuracy(y, p)
knn_accuracy = accuracy_score(y, pk)
#normalize=False
print 'linear regression: \n', accuracy
print 'KNN Accuracy: \n', knn_accuracy

# matplotlib inline
# pl.plot(p, y, 'ro', label='Tested Price')
# pl.plot([0, 12000], [0, 12000], 'g-', label='Predict Accuracy')
# pl.xlabel('Predicted Price')
# pl.ylabel('Actual Price')
# pl.legend(loc='upper left')
# pl.show()

pl.plot(pk, y, 'ro', label=' KNN Tested Price')
pl.plot([0, 12000], [0, 12000], 'g-', label='KNN Predict Accuracy')
pl.xlabel('Predicted Price')
pl.ylabel('Actual Price')
pl.legend(loc='upper left')
pl.show()