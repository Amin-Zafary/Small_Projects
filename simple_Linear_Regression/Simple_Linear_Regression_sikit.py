import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor , LinearRegression
from sklearn.preprocessing import StandardScaler
np.set_printoptions(precision=2)
#plt.style.use('./deeplearning.mplstyle')

data = np.loadtxt("data/houses.txt", delimiter=',', skiprows=1)

X_train = data[:,:4]
y_train = data[:,4]
X_features = ['size(sqft)','bedrooms','floors','age']

print(X_train.shape)
print(y_train.shape)

#Scale/normalize the training data
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

#create and fit regression model
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

#view parameters
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")
#print( "model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")


# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)
# make a prediction using w,b. 
y_pred = np.dot(X_norm, w_norm) + b_norm  
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")


print("////////////////////////////////////////////////////////////////////////////////////////")
#example for normal linear regression in sikit learn

data = np.loadtxt("data/houses.txt", delimiter=',', skiprows=1)

X_train = data[:,:4]
y_train = data[:,4]
X_features = ['size(sqft)','bedrooms','floors','age']

scaler = StandardScaler()
X_train_norm =  scaler.fit_transform(X_train)


linear_model = LinearRegression()
linear_model.fit(X_train, y_train) 

b = linear_model.intercept_
w = linear_model.coef_
print(f"w = {w:}, b = {b:0.2f}")

print(f"Prediction on training set:\n {linear_model.predict(X_train)[:4]}" )
print(f"prediction using w,b:\n {(X_train @ w + b)[:4]}")
print(f"Target values \n {y_train[:4]}")

print(f"Prediction on training set:\n {linear_model.predict(X_train_norm)[:4]}" )
print(f"prediction using w,b:\n {(X_train_norm @ w + b)[:4]}")
print(f"Target values \n {y_train[:4]}")

x_house = np.array([1200, 3,1, 40]).reshape(-1,4)
x_house_predict = linear_model.predict(x_house)[0]
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.2f}")