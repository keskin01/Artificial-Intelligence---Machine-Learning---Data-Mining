from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('prices.csv')
col = ["price", "lot_area", "living_area", "num_floors", "num_bedrooms", "num_bathrooms", "waterfront", "year_built",
       "year_renovated"]
dataset = dataset[col]
X = dataset[col[1:]]
y = dataset[col[0]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
lr = LinearRegression()
lr.fit(X_train, y_train)
coefficient_dataset = pd.DataFrame(lr.coef_, X.columns, columns=['Coefficient'])
print(coefficient_dataset)
y_test_predict = lr.predict(X_test)
print(y_test_predict)
print(y_test)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))