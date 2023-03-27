import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

df_train = pd.read_csv("cleaned_data.csv", sep=",", encoding='utf-8')
#print(df_train)
#print(df_train.columns)
exclude_filter = ~df_train.columns.isin(['Unnamed: 0', 'Credit_Score'])

pca = PCA().fit(df_train.loc[:, exclude_filter])
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.gcf().set_size_inches(3, 4)
#plt.show()

pca = PCA(svd_solver='full', n_components=0.95)
principal_components = pca.fit_transform(df_train.loc[:, exclude_filter])
df_principal = pd.DataFrame(data=principal_components)
#print(df_principal.head())

X_train, X_test, Y_train, Y_test = train_test_split(df_principal, df_train['Credit_Score'], test_size=0.3, random_state=33)

#LOGREG zaczyna się tutaj

logReg = LogisticRegression(random_state = 64)
logReg.fit(X = X_train, y = Y_train)
predictY = logReg.predict(X_test)
print("tekst")
#print(len(predictY))
cm = confusion_matrix(Y_test, predictY)
cm_display = ConfusionMatrixDisplay(cm,display_labels=logReg.classes_).plot()
cm_display.plot()
#plt.show()

classRepo = classification_report(Y_test,predictY)
print(classRepo)

#Obliczanie label 1
TP = 2309
FN = 3590
FP = 1800
TN = 9144
resztamacierzydolicznika = 430
resztamacierzydomianownika = 90+408+430 +3118 +63

accuracy = (TP + TN+resztamacierzydolicznika)/(TP+TN+FP+FN+resztamacierzydomianownika)
recall = TP/(TP+FN)
precision = TP/(TP+FP)
F1score = 2*((precision*recall)/(precision+recall))
#tekst
print(accuracy, recall, precision, F1score)
