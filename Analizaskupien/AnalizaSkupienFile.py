import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

#Czytanie pliku
Df_data = pd.read_csv("countries of the world.csv", sep=",", encoding='utf-8')

# Naprawianie indexów
Convert = ['Population', 'Area (sq. mi.)', 'Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)',
           'Net migration', 'Infant mortality (per 1000 births)', 'GDP ($ per capita)', 'Literacy (%)', 'Phones (per 1000)',
           'Arable (%)', 'Crops (%)', 'Other (%)', 'Climate', 'Birthrate', 'Deathrate', 'Agriculture', 'Industry', 'Service']
dane = Df_data[Convert]

#Usuwanie pustych, zmienianie typu z string na float, zamiana znaków
dane = dane.replace(',', '.', regex=True)
dane = dane.astype(float)
dane= dane.dropna()

#Uporządkowywanie danych i indexów
data.reset_index(drop=True, inplace=True)
Df_data = Df_data.loc[data.index].reset_index(drop=True)
#Standaryzacja
standar = StandardScaler()
standar_dane = standar.fit_transform(dane)

#Dla k = 3
k = 3
km = KMeans(n_clusters=k,random_state=217)
km.fit(standar_dane)

#Wyszukanie optymalnego k

max_score = -1
best_k = 2
for k in range(2, 25):
    kmeans = KMeans(n_clusters=k, random_state=684)
    kmeans.fit(standar_dane)
    labels = kmeans.labels_
    score = silhouette_score(standar_dane, labels)
    if score > max_score:
        max_score = score
        best_k = k
print("Optymalne k to: " + str(best_k))

#Kontynuacja dla k = 3
cl_labels = kmeans.labels_
Df_data['Cluster'] = cl_labels

#Dendrogram
linkage_matrix = linkage(scaled_data, method='ward')
#Plotting
plt.figure(figsize=(12, 12))
dendrogram(linkage_matrix, truncate_mode="level", labels=cluster_labels, color_threshold=k)
plt.title('Plot dendrogram')
plt.xlabel('Cluster')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()