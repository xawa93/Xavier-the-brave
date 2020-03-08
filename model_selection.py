from sklearn.feature_selection import RFE

# Fit using the RFE model
rfe = RFE(lin_model, 7)
X_rfe = rfe.fit_transform(X_train, Y_train)
lin_model.fit(X_rfe, Y_train)
print(rfe.ranking_)

nof_list=np.arange(1,12)            
high_score=0

nof=0           
score_list =[]
for n in range(len(nof_list)):
    rfe = RFE(lin_model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train, Y_train)
    X_test_rfe = rfe.transform(X_test)
    lin_model.fit(X_train_rfe, Y_train)
    score = lin_model.score(X_test_rfe, Y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
        
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))