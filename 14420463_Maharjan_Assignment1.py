import pandas as pd
import numpy as np

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
energy_efficiency = fetch_ucirepo(id=242) 
  
# data (as pandas dataframes) 
X = energy_efficiency.data.features 
y = energy_efficiency.data.targets 
  
# metadata 
print(energy_efficiency.metadata) 
  
# variable information 
print(energy_efficiency.variables) 
def train_valid_test_split(X, y):
    ''' Split the data into train, validation and test datasets using pandas'''
    train=int(X.shape[0]*0.7)
    valid=int(X.shape[0]*0.1)
    test=int(X.shape[0]*0.2)
    print("Data amounts to training data {}, validation data {} and testing data {}. ".format(train, valid, test))

    X_df=pd.DataFrame(X, dtype=float)
    X_df.insert(0, 'W0', 1)
    X_train=X_df.iloc[0:train,:].values
    X_valid=X_df.iloc[train:train+valid,:].values
    X_test=X_df.iloc[train+valid:,:].values
    y_df=pd.DataFrame(y, dtype=float)
    y_train=y_df.iloc[0:train,:].values
    y_valid=y_df.iloc[train:train+valid,:].values
    y_test=y_df.iloc[train+valid:,:].values
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test
def w_hat_lin_calc(X_train, y_train):
    w_hat=np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    return w_hat

def w_hat_lin_calc_pseudo(X_train, y_train):
    w_hat=np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
    return w_hat

def w_hat_ridge_calc(X_train, y_train,lamb):
    n_features=X_train.shape[1]
    w_hat=np.linalg.inv(X_train.T @ X_train + lamb*np.eye(n_features)) @ X_train.T @ y_train
    return w_hat

def predict(X, w_hat):
    y_predict=X @ w_hat
    return y_predict

def compare(y_test, y_predict,n):
    mae_i = np.mean(np.abs(y_test - y_predict))
    mse_i = np.mean((y_test - y_predict) ** 2)
    rmse_i = np.sqrt(mse_i)
    print("For the Y{}, the MAE is {:.3f}, MSE is {:.3f} and RMSE is {:.3f}."
        .format(n, mae_i, mse_i, rmse_i))
    return

def standardize(X):
    X_std=(X - np.mean(X)) / np.std(X)
    return X_std

X_train, X_valid, X_test, y_train, y_valid, y_test=train_valid_test_split(X,y)
w_hat_rid1=w_hat_ridge_calc(X_train, y_train[:, 0],1)
w_hat_rid2=w_hat_ridge_calc(X_train, y_train[:, 1],1)
y_predict_ridge_train1=predict(X_train, w_hat_rid1)
y_predict_ridge_train2=predict(X_train, w_hat_rid2)
print("###TRAINING DATASET RESULTS###")
print("-------------------------")
print("Ridge Regression Results on Training Data:")
print("-------------------------")
compare(y_train[:, 0], y_predict_ridge_train1,1)
compare(y_train[:, 1], y_predict_ridge_train2,2)

print("-------------------------")
print("###VALIDATION DATASET RESULTS###")
# validation dataset calculation and results

w_hat_lin1=w_hat_lin_calc(X_train, y_train[:, 0])
w_hat_lin2=w_hat_lin_calc(X_train, y_train[:, 1])
y_predict_linear1=predict(X_valid, w_hat_lin1)
y_predict_linear2=predict(X_valid, w_hat_lin2)
print("-------------------------")
print("Linear Regression Results using np.linalg.inv:")
print("-------------------------")
compare(y_valid[:, 0], y_predict_linear1,1)
compare(y_valid[:, 1], y_predict_linear2,2)
print("-------------------------")

X_train, X_valid, X_test, y_train, y_valid, y_test=train_valid_test_split(X,y)
w_hat_lin_std1=w_hat_lin_calc(standardize(X_train), y_train[:, 0])
w_hat_lin_std2=w_hat_lin_calc(standardize(X_train), y_train[:, 1])
y_predict_linear1=predict(standardize(X_valid), w_hat_lin_std1)
y_predict_linear2=predict(standardize(X_valid), w_hat_lin_std2)
print("-------------------------")
print("Linear Regression Results using standardized data:")
print("-------------------------")
compare(y_valid[:, 0], y_predict_linear1,1)
compare(y_valid[:, 1], y_predict_linear2,2)
print("-------------------------")

w_hat_lin_calc_pseudo1=w_hat_lin_calc_pseudo(X_train, y_train[:, 0])
w_hat_lin_calc_pseudo2=w_hat_lin_calc_pseudo(X_train, y_train[:, 1])
y_predict_linear1=predict(X_valid, w_hat_lin_calc_pseudo1)
y_predict_linear2=predict(X_valid, w_hat_lin_calc_pseudo2)
print("-------------------------")
print("Linear Regression Results using np.linalg.pinv:")
print("-------------------------")
compare(y_valid[:, 0], y_predict_linear1,1)
compare(y_valid[:, 1], y_predict_linear2,2)
print("-------------------------") 


print("-------------------------")
y_predict_ridge1=predict(X_valid, w_hat_rid1)
y_predict_ridge2=predict(X_valid, w_hat_rid2)

print("Ridge Regression Results of validation data:")
print("-------------------------")
compare(y_valid[:, 0], y_predict_ridge1,1)
compare(y_valid[:, 1], y_predict_ridge2,2)
print("-------------------------")
print("-------------------------")
print("###TEST DATASET RESULTS###")
# testing dataset calculation and results   
y_predict_linear1=predict(X_test, w_hat_lin1)
y_predict_linear2=predict(X_test, w_hat_lin2)
print("-------------------------")
print("Linear Regression Results using np.linalg.inv for test dataset:")
print("-------------------------")
compare(y_test[:, 0], y_predict_linear1,1)
compare(y_test[:, 1], y_predict_linear2,2)
print("-------------------------")
y_predict_linear1=predict(standardize(X_test), w_hat_lin_std1)
y_predict_linear2=predict(standardize(X_test), w_hat_lin_std2)
print("-------------------------")
print("Linear Regression Results using standardized data for test dataset:")
print("-------------------------")
compare(y_test[:, 0], y_predict_linear1,1)
compare(y_test[:, 1], y_predict_linear2,2)
print("-------------------------")
y_predict_linear1=predict(X_test, w_hat_lin_calc_pseudo1)
y_predict_linear2=predict(X_test, w_hat_lin_calc_pseudo2)
print("-------------------------")
print("Linear Regression Results using np.linalg.pinv for test dataset:")
print("-------------------------")
compare(y_test[:, 0], y_predict_linear1,1)
compare(y_test[:, 1], y_predict_linear2,2)
print("-------------------------")
print("-------------------------")
y_predict_ridge1=predict(X_test, w_hat_rid1)
y_predict_ridge2=predict(X_test, w_hat_rid2)
print("Ridge Regression Results for test dataset:")
print("-------------------------")
compare(y_test[:, 0], y_predict_ridge1,1)
compare(y_test[:, 1], y_predict_ridge2,2)
print("-------------------------")

from sklearn.linear_model import LinearRegression, Ridge
X_df=pd.DataFrame(X)
train=int(X.shape[0]*0.7)
valid=int(X.shape[0]*0.1)
test=int(X.shape[0]*0.2)
X_train=X_df.iloc[0:train,:].values
X_valid=X_df.iloc[train:train+valid,:].values
X_test=X_df.iloc[train+valid:,:].values
y_df=pd.DataFrame(y)
y_train=y_df.iloc[0:train,:].values
y_valid=y_df.iloc[train:train+valid,:].values
y_test=y_df.iloc[train+valid:,:].values
LR_model=LinearRegression()
LR_model.fit(X_train, y_train)
y_predict_sklearn=LR_model.predict(X_test)
RR_model=Ridge(alpha=1)
RR_model.fit(X_train, y_train)
y_predict_ridge_sklearn=RR_model.predict(X_test)

print("--------------")
print("Sklearn Linear Regression Results:")
print("--------------")
compare(y_test, y_predict_sklearn,1)

print("--------------")
print("Sklearn Ridge Regression Results:")
print("--------------")
compare(y_test, y_predict_ridge_sklearn,2)
print("-------------------------")