

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Mall_Customers.csv')
df

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

kmeans = KMeans(n_clusters=3,random_state=0)
kmeans.fit(df_scaled)

plt.figure(figsize=(10, 5))
sns.scatterplot(x=df['Age'], y=df['Annual Income (k$)'], hue=kmeans.labels_,palette='Set2',s=100)
plt.title('Clusters of Customers')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.show()

cluster_info = df.copy()
cluster_info['Cluster'] = kmeans.labels_
cluster_info
