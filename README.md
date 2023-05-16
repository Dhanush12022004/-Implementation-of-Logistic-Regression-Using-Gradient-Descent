# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: DHANUSH GR
RegisterNumber:  212221040038
*/
```

import numpy as np

import matplotlib.pyplot as plt

from scipy import optimize 

data=np.loadtxt("/content/ex2data1 (1).txt",delimiter=',')

x=data[:,[0,1]]

y=data[:,2]

print("Array value of x:")

x[:5]


print("Array value of y:")

y[:5]

print("Exam 1-score graph:")

plt.figure()

plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted")

plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not admitted")

plt.xlabel("Exam 1 score")

plt.ylabel("Exam 2 score")

plt.legend()

plt.show()

def sigmoid(z):

  return 1/(1+np.exp(-z))

print("Sigmoid function graph: ")

plt.plot()

x_plot=np.linspace(-10,10,100)

plt.plot(x_plot,sigmoid(x_plot))

plt.show()

def costFunction(theta,x,y):

  h=sigmoid(np.dot(x,theta))
  
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  
  grad=np.dot(x.T,h-y)/x.shape[0]
  
  return J,grad

x_train=np.hstack((np.ones((x.shape[0],1)),x))

theta=np.array([0,0,0])

J,grad=costFunction(theta,x_train,y)

print("x_train_grad value:")

print(J)

print(grad)

x_train=np.hstack((np.ones((x.shape[0],1)),x))

theta=np.array([-24,0.2,0.2])

J,grad=costFunction(theta,x_train,y)

print("y_train_grad value:")

print(J)

print(grad)

def cost(theta,x,y):

  h=sigmoid(np.dot(x,theta))
  
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  
  return J
  
def gradient(theta,x,y):

  h=sigmoid(np.dot(x,theta))
  
  grad=np.dot(x.T,h-y)/x.shape[0]
  
  return grad
  
x_train=np.hstack((np.ones((x.shape[0],1)),x))

theta=np.array([0,0,0])

res=optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method='Newton-CG',jac=gradient)

print("res.x:")

print(res.fun)

print(res.x)


def plotDecisionBoundary(theta,x,y):

  x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
  
  y_min,y_max=x[:,1].min()-1,x[:,1].max()+1
  
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  
  x_plot=np.c_[xx.ravel(),yy.ravel()]
  
  x_plot=np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  
  y_plot=np.dot(x_plot,theta).reshape(xx.shape)
  

  plt.figure()
  
  plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted")
  
  plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Admitted")
  
  plt.contour(xx,yy,y_plot,levels=[0])
  
  plt.xlabel("Exam 1 score")
  
  plt.ylabel("Exam 2 score")
  

  plt.legend()
  
  plt.show()
  
print("Descision Boundary - graph for exam score:")

plotDecisionBoundary(res.x,x,y)


print("probability value:")

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))

print(prob)


def predict(theta, x):

  x_train = np.hstack((np.ones((x.shape[0], 1)), x))
  
  prob = sigmoid(np.dot(x_train, theta))
  
  return (prob >= 0.5).astype(int)
  
print("Prediction value of mean:")

np.mean(predict(res.x, x)  == y)

## Output:
![image](https://github.com/Dhanush12022004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128135558/4b9e0818-c880-4ade-bb6b-3ffd09721dbe)

![image](https://github.com/Dhanush12022004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128135558/f385a053-5614-4c1e-a966-6698b3e64eaa)

![image](https://github.com/Dhanush12022004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128135558/ab5d0f3c-b4d5-4454-9b27-27664b351bc7)

![image](https://github.com/Dhanush12022004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128135558/320fdb9f-8281-4685-a483-6a7d325551d5)

![image](https://github.com/Dhanush12022004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128135558/35478086-ae23-4aae-92a4-9c2ed4f37ed8)

![image](https://github.com/Dhanush12022004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128135558/ce4cb4eb-98c3-4a72-b3b7-67475d71f68e)

![image](https://github.com/Dhanush12022004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128135558/3fa8fd68-600a-4ee2-b462-5412f74687d7)

![image](https://github.com/Dhanush12022004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128135558/5cfb4f66-3ee4-4ebe-8ee5-d3fedce06586)

![image](https://github.com/Dhanush12022004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128135558/757fe62f-80d3-44fa-96f2-c6028d55cb4a)

![image](https://github.com/Dhanush12022004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128135558/560624f2-7a96-47d5-8e91-23c5b94469ca)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

