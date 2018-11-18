

#Importing libraries(There are some unwanted modules , you can use it to increase the accuracy)

import numpy as np
from sklearn import preprocessing,model_selection,svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectPercentile
import  pandas as pd

#Reading our data

df=pd.read_csv('E:/Nandhu/Datasets/boston.csv',index_col=0)

'''
#THIS IS USED TO FIND THE CORRELATION BETWEEN THE VARIABLES

corelation=df.corr()
sns.heatmap(corelation)
plt.show()
'''

#Our feature
X=np.array(df.drop(['MV'],1))

#Our label
y=np.array(df['MV'])

#splitting testing and training data
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2)

#Choosing linear regression
clf=LinearRegression()


#Training our data
clf.fit(X_train,y_train)

#Finding the accuracy
accuracy=clf.score(X_test,y_test)
print("Accuracy =",accuracy)

#Plotting the our result on the graph
y_predict=clf.predict(X_test)
plt.scatter(y_test,y_predict)
plt.xlabel("Prices")
plt.ylabel("Predicted Prices")
plt.show()