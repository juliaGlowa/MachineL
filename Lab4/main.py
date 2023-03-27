import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

df_train = pd.read_csv("cleaned_data.csv", sep=",", encoding='utf-8')
print(df_train)
print(df_train.columns)
exclude_filter = ~df_train.columns.isin(['Unnamed: 0', 'Credit_Score'])

pca = PCA().fit(df_train.loc[:, exclude_filter])
plt.plot(np.cumsum(pca.explained_variance_ratio))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.gcf().set_size_inches(3, 4)
plt.show()

pca = PCA(svd_solver='full', n_components=0.95)
principal_components = pca.fit_transform(df_train.loc[:, exclude_filter])
df_principal = pd.DataFrame(data=principal_components)
print(df_principal.head())

X_train, X_test, Y_train, Y_test = train_test_split(df_principal, df_train['Credit_Score'], test_size=0.3, random_state=33)
