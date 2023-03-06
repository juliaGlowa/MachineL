import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

df_train = pd.read_csv("train.csv", sep = "," , encoding= 'utf-8')
df_train.shape
df_train.info()

df_train.head()
df_train.drop_duplicates(subset="ID", inplace=True)
df_train.describe()
#for i in df_train.columns:
#  print(df_train[i].value_counts())
#  print('*'*50)
sns.countplot(x = 'Credit_Score', data = df_train)
plt.show()
df_train.info()
FeaturesToConvert = ['Age','Annual_Income','Num_of_Loan','Num_of_Delayed_Payment','Changed_Credit_Limit','Outstanding_Debt','Amount_invested_monthly','Monthly_Balance']
for feature in FeaturesToConvert:
    uniques = df_train[feature].unique()
    df_train[feature] = df_train[feature].str.strip('-_')
    df_train[feature] = df_train[feature].replace({'':np.nan})
    df_train[feature] = df_train[feature].astype('float64')

df_train['Monthly_Inhand_Salary']= df_train['Monthly_Inhand_Salary'].fillna(method='pad')
le = LabelEncoder()
df_train.Occupation = le.fit_transform(df_train.Occupation)
print(df_train.head())
columns = ['Payment_of_Min_Amount', 'Credit_Score', 'Credit_Mix', 'Payment_Behaviour', 'Type_of_Loan']
df_train[columns] = df_train[columns].apply(le.fit_transform)
print(df_train.head())