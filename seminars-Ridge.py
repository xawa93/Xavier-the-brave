ridge_model = Ridge(alpha = 0.5)

ridge_model_fit = ridge_model.fit(boston, boston_Y)

boston_predict = ridge_model_fit.predict(X_train)

rmse = (np.sqrt(mean_squared_error(Y_train, boston_predict)))
r2 = r2_score(Y_train, boston_predict)

print("Training performance (L2 regularisation)")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

boston_predict_test = ridge_model_fit.predict(X_test)

rmse = (np.sqrt(mean_squared_error(Y_test, boston_predict_test)))
r2 = r2_score(Y_test, boston_predict_test)

print("Testing performance (L2 regularisation)")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

betas = ridge_model_fit.coef_

counter = 0
for col in boston.columns:
    if counter == 0:
        print("Beta weights/co-efficients (L2 regularisation)")
        print("-----------------------------------------")
    print(col + ": " + str(round(betas[counter], 4)))
    counter +=1
print("\n")