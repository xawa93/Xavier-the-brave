import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns 
%matplotlib inline

# import a standard dataset - the Boston house price index
from sklearn.datasets import load_boston
boston_dataset = load_boston()
# show the dataset
boston_dataset

# show the keys
print(boston_dataset.keys())

# convert to data frame using Pandas
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()

# add the target value (Median Valye)
boston['MV'] = boston_dataset.target

# check if any data is null
boston.isnull().sum()

# create a correlation matrix rounding to one decimal point
correlation_matrix = boston.corr().round(1)
# print a correlation heat map
sns.heatmap(data=correlation_matrix, annot=True)

# remove the correlated variable and the Y value
# the Y value is going to be loaded separately
boston = boston.drop(['RAD', 'MV'], axis=1)
boston.head()

# create a separate Y value
boston_Y = boston_dataset.target

# split data into training and test
from sklearn.model_selection  import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(boston, boston_Y, test_size = 0.2, random_state=5)

# print the shapes to check everything is OK
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# build a linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lin_model = LinearRegression()

# fit the model to the training data
lin_model_fit = lin_model.fit(X_train, Y_train)
lin_model_fit

# print the alpha value of the model (intercept)
print(lin_model_fit.intercept_)

# print the beta values of the model (co-efficients)
betas = lin_model_fit.coef_
counter = 0
for col in boston.columns:
    if counter == 0:
        print("Beta weights/co-efficients (unregularised)")
        print("-----------------------------------------")
    print(col + ": " + str(round(betas[counter], 4)))
    counter +=1

# predict the training data
boston_predict = lin_model_fit.predict(X_train)

# calculate RMSE (root mean square error) and R^2 (predictive power)
# training set
rmse = (np.sqrt(mean_squared_error(Y_train, boston_predict)))
r2 = r2_score(Y_train, boston_predict)

# print the performance metrics
print("Training performance (unregularised)")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# predict the test set
boston_predict_test = lin_model_fit.predict(X_test)

# calculate RMSE (root mean square error) and R^2 (predictive power)
# testing set
rmse = (np.sqrt(mean_squared_error(Y_test, boston_predict_test)))
r2 = r2_score(Y_test, boston_predict_test)

print("Testing performance (unregularised)")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# print the coefficients again
betas = lin_model_fit.coef_

counter = 0
for col in boston.columns:
    if counter == 0:
        print("Beta weights/co-efficients (unregularised)")
        print("-----------------------------------------")
    print(col + ": " + str(round(betas[counter], 4)))
    counter +=1

#############################################################
# BREAK
#############################################################



#############################################################
######                                                 ######
######    Expansion to L1 Model                        ######
######                                                 ######
#############################################################

from sklearn.linear_model import Lasso, Ridge, ElasticNet

# fit a L1 model - alpha is a value between 0 and inf where higher
# means more regularisation. Typically we use max = 1
lasso_model = Lasso(alpha = 0.5)

# fit the model to the training data
lasso_model_fit = lasso_model.fit(boston, boston_Y)

# predict the training data
boston_predict_lasso = lasso_model_fit.predict(X_train)

# calculate RMSE (root mean square error) and R^2 (predictive power)
# training set
rmse = (np.sqrt(mean_squared_error(Y_train, boston_predict_lasso)))
r2 = r2_score(Y_train, boston_predict_lasso)

# print the performance metrics
print("Training performance (L1 regularisation)")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# predict the testing data
boston_predict_lasso_test = lasso_model_fit.predict(X_test)

# calculate RMSE (root mean square error) and R^2 (predictive power)
# testing set
rmse = (np.sqrt(mean_squared_error(Y_test, boston_predict_lasso_test)))
r2 = r2_score(Y_test, boston_predict_lasso_test)

# print the performance metrics
print("Testing performance (L1 regularisation)")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# print the coefficients
betas = lasso_model_fit.coef_
counter = 0
for col in boston.columns:
    if counter == 0:
        print("Beta weights/co-efficients (L1 regularisation)")
        print("-----------------------------------------")
    print(col + ": " + str(round(betas[counter], 4)))
    counter +=1