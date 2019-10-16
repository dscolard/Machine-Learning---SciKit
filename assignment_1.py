

import math 
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt  
import seaborn as sns 
from scipy import stats
from datetime import datetime
from sklearn import preprocessing
from matplotlib.pyplot import *
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import TransformedTargetRegressor

from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn import metrics



def run():
	# -- LOAD FILE -- #
	filename = 'data/tcd ml 2019-20 income prediction training (with labels).csv'
	df = pd.read_csv(filename)


	# -- Select relevant Data -- #
	df = df[['Year of Record', 'Gender', 'Profession','Age', 'Country', 'Size of City', 'University Degree', 'Body Height [cm]', 'Income in EUR']]


	df["Age"].fillna(method="ffill", inplace= True)
	# dataset["Profession"].fillna(method="ffill", inplace= True)
	df["Year of Record"].fillna(method="ffill", inplace= True)
	df["Gender"].fillna(method="ffill", inplace= True)
	df.dropna(inplace=True)
	y = df['Income in EUR']


	# --- One Hot Encoding --- #
	df = pd.get_dummies(df, prefix_sep='_', drop_first=True)


	# --- Label Encoder --- #
	encoder = preprocessing.LabelEncoder();
	df = df.apply(encoder.fit_transform);


	# -- Drop rows with NaN (<0.01% of rows)-- #
	# df = df.dropna(how='any')


	# -- Get relevant columns -- #
	# cor = df.corr()
	# cor_target = abs(cor["Income in EUR"])
	# relevant_features = cor_target[cor_target>0.1]
	# print(relevant_features.index)
	

	# --- Multivariate Linear Regression --- #

	transformer = QuantileTransformer(output_distribution='normal')


	X = (df.drop(["Income in EUR"],1))


	regressor = LinearRegression()
	regr = TransformedTargetRegressor(regressor=regressor,transformer=transformer)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
	regr.fit(X_train, y_train)


	# coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  


	# -- Test Model -- #
	y_pred = regr.predict(X_test)
	df = pd.DataFrame({'Actual':y_test, 'Predicted': y_pred})
	df1 = df.head(25)
	print(df1)

	file = 'assignment_1.sav'
	pickle.dump(regr, open(file, 'wb'))

	print(np.sqrt(mean_squared_error(y_test,y_pred)))


	# # ---------------------------------


	filename = 'data/tcd ml 2019-20 income prediction test (without labels).csv'
	df = pd.read_csv(filename)

	df = df[["Year of Record","Gender","Age","Country","Size of City","Profession","University Degree","Body Height [cm]"]]

	df["Age"].fillna( method ='ffill', inplace = True)
	df["Gender"].fillna(method="ffill", inplace= True)
	df["Year of Record"].fillna( method ='ffill', inplace = True)
	# df["Profession"].fillna( method ='ffill', inplace = True)
	

	df = pd.get_dummies(df, prefix_sep='_', drop_first=True)
	df = df.apply(encoder.fit_transform)

	X, df = X.align(df, join='left', axis=1)

	# df = pd.get_dummies(df, prefix_sep='_', drop_first=True)

	df.fillna(value=0, inplace=True)

	# print(df.head(10))

	model = pickle.load(open('assignment_1.sav', 'rb'))

	result = model.predict(df)

	np.savetxt('newresult.csv', result, delimiter=',')



if __name__ == '__main__':
	run()












