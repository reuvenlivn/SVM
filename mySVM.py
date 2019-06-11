# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:51:22 2019

@author: reuve
"""

import matplotlib.pyplot as plt
from sklearn import datasets,svm,model_selection
import numpy as np

#4
iris = datasets.load_iris()

#5
data = iris.data[:, :2]  # we only take the first two features.
labels = iris.target

#question 6
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data,labels)
my_svm = svm.SVC()
my_svm.fit(X_train,Y_train)

#7
artificial_data = np.array([[4.8,3.9],[5,2.7],[6,2.9],[7,3]])
artificial_labels = my_svm.predict(artificial_data)

#8
plt.scatter(artificial_data[:,0],artificial_data[:,1],c=artificial_labels)
plt.title('artificial data')
plt.show()
#9
support_vectors = my_svm.support_vectors_
plt.scatter(support_vectors[:,0],support_vectors[:,1],c='r')
plt.title('support vectors')
plt.show()
#10
X_0_2 = iris.data[:, (0,2)]  # we only take the first and third features.
X_train_0_2, X_test_0_2, Y_train, Y_test = model_selection.train_test_split(X_0_2,labels)

#11
#a. linear
svm_linear = svm.SVC(kernel='linear',gamma='scale')
svm_linear.fit(X_train_0_2,Y_train)
support_vectors_lin = svm_linear.support_vectors_
plt.scatter(support_vectors_lin[:,0],support_vectors_lin[:,1],c='r')
plt.title('support_vectors_linear')
plt.show()

score = svm_linear.score(X_test_0_2, Y_test)
print('\n svm_linear score={}\n'.format(score))

#b. plynomial 3
svm_poly3 = svm.SVC(kernel = 'poly',degree=3,gamma='scale')
svm_poly3.fit(X_train_0_2,Y_train)
suppor_vectors_poly3 = svm_poly3.support_vectors_
plt.scatter(suppor_vectors_poly3[:,0],suppor_vectors_poly3[:,1],c='r')
plt.title('support_vectors_poly 3')
plt.show()

score = svm_poly3.score(X_test_0_2, Y_test)
print('\n svm_poly3 score={}\n'.format(score))

#c radial kernel (rbg)
svm_raidal = svm.SVC(kernel = 'rbf',gamma='scale')
svm_raidal.fit(X_train_0_2,Y_train)
support_vectors_raidal = svm_raidal.support_vectors_
plt.scatter(support_vectors_raidal[:,0],support_vectors_raidal[:,1],c='r')
plt.title('support_vectors_raidal')
plt.show()

score = svm_raidal.score(X_test_0_2, Y_test)
print('\n svm_raidal score={}\n'.format(score))

#12
digits = datasets.load_digits()
digits_data = digits.data
digits_labels = digits.target

#13
for i in range(10):
    plt.imshow(digits.images[i])
    #plt.show() is in order to make sure that the next 
    #figure is not above the currnt one but a new one
    plt.show()  
   
#14

x_train_digits, x_test_digits, y_train_digits, y_test_digits = model_selection.train_test_split(digits_data,digits_labels,test_size=0.49)
 
my_svm = svm.SVC(kernel='poly',gamma='scale')
my_svm.fit(x_train_digits,y_train_digits)

#15
score = my_svm.score(x_test_digits,y_test_digits)

#question 16
#C is Penalty parameter of the error term.
my_svm_1 = svm.SVC(C=1,kernel='poly',gamma='scale')
my_svm_1.fit(x_train_digits,y_train_digits)
score_1 = my_svm_1.score(x_test_digits,y_test_digits)
print('\n svm_digits score={} C=1\n'.format(score_1))

my_svm_2 = svm.SVC(C=0.0001,kernel='poly',gamma='scale')
my_svm_2.fit(x_train_digits,y_train_digits)
score_2 = my_svm_2.score(x_test_digits,y_test_digits)
print('\n svm_digits score={} C=0.0001\n'.format(score_2))

my_svm_3 = svm.SVC(C=0.001,kernel='poly',gamma='scale')
my_svm_3.fit(x_train_digits,y_train_digits)
score_3 = my_svm_3.score(x_test_digits,y_test_digits)
print('\n svm_digits score={} C=0.001\n'.format(score_3))


