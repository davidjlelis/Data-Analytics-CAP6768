'''
Project Title: The Impact Artificial Intelligence has on Future Jobs
Project step: Preprocessing
Purpose: Preprocessing AI impact on Jobs dataset to a structure so we can effectively mine and analyze the data.

Creators: Group 4
    Marina Lombardo (N01578798)
    Andrea Garcia Fernandez (N01561009)
    David Lelis (N00957151)
    Stephanie El-Bahri (N01404796)


Problem: Dataset doesn't contain enough data to analyze effectively as job titles vary for roles that share very similar 
    responsibilities. Cases may involve the following:
        1. Job titles are very short like Chef, Teacher, Manager, etc.
        2. Job titles that share very similar responsibilies but have different titles like Data Analyst vs Data Scientist
        3. Job titles that share similart titles but have different responsibilities like Chief Data Office vs Chief Executive Officer

Solutions:
    1. Short Text Clustering Algorithms
        - Since the data is limited, finding an algorithm that maximizes the data given would be a good shot to group these roles 
            based on the given data.
    2. Web Scraping Job Descriptions and Clustering
        - Add more data by looking up the well-known responsilibities of these roles and clustering based on this data.
'''

# Import libraries
import pandas as pd
import numpy as np
import opendatasets as od
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# -- use to check for library versions
#print(pd.__version__)
#print(np.__version__)

# -- use to download dataset, if not downloaded. Comment out if downloaded.
# od.download("https://www.kaggle.com/datasets/manavgupta92/from-data-entry-to-ceo-the-ai-job-threat-index")

file = ('from-data-entry-to-ceo-the-ai-job-threat-index//My_Data.csv')
data = pd.read_csv(file)

print(data.head())

sentence = data['Job titiles']

vectorizer = TfidfVectorizer(stop_words='english')

vectorized_documents = vectorizer.fit_transform(sentence)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(vectorized_documents.toarray())

num_clusters= 5
kmeans = KMeans(n_clusters = num_clusters,
                n_init = 45,
                max_iter=500,
                random_state=42)
kmeans.fit(vectorized_documents)

results = pd.DataFrame()
results['document'] = sentence
results['cluster'] = kmeans.labels_

# print(results.sample(5))

results.to_csv('Clusters.csv', index=False)

'''
# visual graph of the clusters

colors = ['red', 'green', 'blue', 'yellow', 'black']
cluster = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']

for i in range(num_clusters):
    plt.scatter(reduced_data[kmeans.labels_ == i, 0],
                reduced_data[kmeans.labels_ == i, 1],
                s = 10, color=colors[i],
                label=f' {cluster[i]}')

plt.legend()
plt.show()
'''