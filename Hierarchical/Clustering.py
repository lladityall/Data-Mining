import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('whole_sale_customers_data.csv')
df.head()

data_clean = df.drop(['Channel', 'Region'], axis=1)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_clean)

linked = linkage(data_scaled, method='complete',metric='euclidean')
plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode='level',p=5)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Customer Index')
plt.ylabel('Distance')
plt.show()

hc = AgglomerativeClustering(n_clusters=3, linkage='complete') # Remove affinity parameter
cluster_labels = hc.fit_predict(data_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=df['Grocery'], y=df['Milk'], hue=cluster_labels, palette='Set1', s=100)
plt.title("Hierarchical Clustering Result")
plt.xlabel("Grocery")
plt.ylabel("Milk")
plt.legend(title='Cluster')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=0)
df['KMeans_Cluster'] = kmeans.fit_predict(data_scaled)

# Compare cluster counts
print(cluster_labels,df[['KMeans_Cluster']].value_counts())
