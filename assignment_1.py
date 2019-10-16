import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

filename = 'tcd ml 2019-20 income prediction training (with labels).csv'
df = pd.read_csv(filename)

df = df[['Year of Record','Gender','Age','Country','Profession','University Degree','Income in EUR']]

print(df.head())


