import pandas as pd
from sklearn import metrics

# training = pd.read_csv("training_kmeans.csv")
# heirarchical_training = pd.read_csv("training_Heirarchical.csv")
# em_training = pd.read_csv("training_EM.csv")

# training['em_cluster'] = em_training['EM_cluster']
# training['hierarchical_cluster'] = heirarchical_training['Heirarchical']

training = pd.read_csv('training_result.csv')

em_kmeans = metrics.cohen_kappa_score(
    training['em_cluster'], training['kmean_cluster'])
em_hierarcihcal = metrics.cohen_kappa_score(
    training['em_cluster'], training['hierarchical_cluster'])
kmeans_hierarchical = metrics.cohen_kappa_score(
    training['kmean_cluster'], training['hierarchical_cluster'])


kappa = {}
kappa['y1'] = ['em_cluster', 'em_cluster', 'kmeans_cluster']
kappa['y2'] = ['kmeans_cluster', 'hierarchical_cluster', 'hierarchical_cluster']
kappa['kappa'] = [em_kmeans, em_hierarcihcal, kmeans_hierarchical]

pd.DataFrame(kappa).to_csv('kappa_results.csv', index=False)
