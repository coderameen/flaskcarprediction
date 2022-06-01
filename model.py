import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

#load the dataset
df = pd.read_csv('carprediction.csv')
#print(df)

#divide the dataset in x and y axis
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

#split the dataset into training and testing
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.1)

#Import the model
model = LinearRegression()

#train the model
model.fit(x_train,y_train)

'''
#predict
prediction = model.predict([[46000,4]])
print("the care price is ",prediction)
'''

#to write the trained model into pickle file 
file = open('model_plk','wb')
pickle.dump(model,file)