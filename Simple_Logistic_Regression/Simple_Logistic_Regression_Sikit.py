import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

#fit the model
lr_model = LogisticRegression()
lr_model.fit(X, y)

#make prediction
y_pred = lr_model.predict(X)
print("Prediction on training set:", y_pred)

#calculate accuracy
print("Accuracy on training set:", lr_model.score(X, y))


print("/////////////////////////////////////////////////////////////////////////////////////")
# using sikit learn logistic regression on another dataset

data = np.loadtxt("data/ex2data1.txt", delimiter=',')
X_train = data[:,:2]
y_train = data[:,2]

#fit the model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

#make prediction
y_pred = lr_model.predict(X_train)
print("Prediction on training set:", y_pred)

print("Accuracy on training set:", lr_model.score(X_train, y_train))


print("/////////////////////////////////////////////////////////////////////////////////////")
# using sikit learn logistic regression on another dataset

data = np.loadtxt("data/ex2data2.txt", delimiter=',')
X_train = data[:,:2]
y_train = data[:,2]

#for have complex features
def map_feature(X1, X2):
    """
    Feature mapping function to polynomial features    
    """
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    degree = 6
    out = []
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((X1**(i-j) * (X2**j)))
    return np.stack(out, axis=1)

mapped_X = map_feature(X_train[:, 0], X_train[:, 1])

#fit the model
lr_model = LogisticRegression()
lr_model.fit(mapped_X, y_train)

#make prediction
y_pred = lr_model.predict(mapped_X)
print("Prediction on training set:", y_pred)

#calculate accuracy
print("Accuracy on training set:", lr_model.score(mapped_X, y_train))
