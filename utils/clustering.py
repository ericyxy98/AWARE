import torch
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import sklearn.metrics as M
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import seaborn as sns

def evaluate(model, train_loader, val_loader, test_loader):
    N_clusters = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = np.zeros((0,36))
    y = np.zeros(0)
    z = np.zeros(0)
    for batch_idx, (inputs, masks, labels, info) in enumerate(train_loader):
        inputs = inputs.to(device)
        X_train = model.encoder(inputs.unsqueeze(1)).cpu().detach().numpy()
        X_train = np.concatenate((X_train, info[:, 2:6]),axis=1)
        y_train = np.full_like(labels[:,0], 0)
        X = np.concatenate((X, X_train),axis=0)
        y = np.concatenate((y, y_train),axis=0)
        z = np.concatenate((z, labels[:,0]),axis=0)
    
    for batch_idx, (inputs, masks, labels, info) in enumerate(val_loader):
        inputs = inputs.to(device)
        X_val = model.encoder(inputs.unsqueeze(1)).cpu().detach().numpy()
        X_val = np.concatenate((X_val, info[:, 2:6]),axis=1)
        y_val = np.full_like(labels[:,0], 1)
        X = np.concatenate((X, X_val),axis=0)
        y = np.concatenate((y, y_val),axis=0)
        z = np.concatenate((z, labels[:,0]),axis=0)
    
    for batch_idx, (inputs, masks, labels, info) in enumerate(test_loader):
        inputs = inputs.to(device)
        X_test = model.encoder(inputs.unsqueeze(1)).cpu().detach().numpy()
        X_test = np.concatenate((X_test, info[:, 2:6]),axis=1)
        y_test = np.full_like(labels[:,0], 2)
        X = np.concatenate((X, X_test),axis=0)
        y = np.concatenate((y, y_test),axis=0)
        z = np.concatenate((z, labels[:,0]),axis=0)
        
    tsne = TSNE(n_components=2, verbose=1, n_iter=300)
    tsne_results = tsne.fit_transform(X)

    results = {'tsne-2d-one': tsne_results[:,0],
           'tsne-2d-two': tsne_results[:,1],
           'y': pd.Categorical(y),
           'z': pd.Categorical(z)}
    
    plt.figure(figsize=(6,4))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="z",
        data=results,
        palette=sns.color_palette("Set2"),
        legend="full",
    )
    plt.legend(["Healthy","Asthma"])
    plt.title("Labels")
    plt.show()
    
    metrics = np.zeros((3,3))
    
    # K-means
    clustering = KMeans(n_clusters=N_clusters).fit(X)
    results['cluster'] = pd.Categorical(clustering.labels_)
    
#     plt.figure(figsize=(6,4))
#     sns.scatterplot(
#         x="tsne-2d-one", y="tsne-2d-two",
#         hue="cluster",
#         data=results,
#         palette=sns.color_palette("Set2"),
#         legend="full",
#     )
#     plt.title("K-means++")
#     plt.show()
    
    results_df = pd.DataFrame(results)
    f_train = results_df['cluster'][results_df['y']==0].value_counts(sort=False)
    f_val = results_df['cluster'][results_df['y']==1].value_counts(sort=False)
    f_test = results_df['cluster'][results_df['y']==2].value_counts(sort=False)
    
    metrics[0,0] = M.silhouette_score(X,results['cluster'])
    metrics[0,1] = stats.chisquare(f_obs=f_test, f_exp=f_train/f_train.sum()*f_test.sum()).pvalue
    metrics[0,2] = stats.chisquare(f_obs=f_test, f_exp=f_val/f_val.sum()*f_test.sum()).pvalue
#     print("SS = ", metrics[0,0])
#     print("Train vs. test")
#     print(metrics[0,1])
#     print("Val vs. test")
#     print(metrics[0,2])
    
    # Agglomerative
    clustering = AgglomerativeClustering(n_clusters=N_clusters, distance_threshold=None).fit(X)
    results['cluster'] = pd.Categorical(clustering.labels_)
    
#     plt.figure(figsize=(6,4))
#     sns.scatterplot(
#         x="tsne-2d-one", y="tsne-2d-two",
#         hue="cluster",
#         data=results,
#         palette=sns.color_palette("Set2"),
#         legend="full",
#     )
#     plt.title("Hierarchical (Agglomerative)")
#     plt.show()
    
    results_df = pd.DataFrame(results)
    f_train = results_df['cluster'][results_df['y']==0].value_counts(sort=False)
    f_val = results_df['cluster'][results_df['y']==1].value_counts(sort=False)
    f_test = results_df['cluster'][results_df['y']==2].value_counts(sort=False)
    
    metrics[1,0] = M.silhouette_score(X,results['cluster'])
    metrics[1,1] = stats.chisquare(f_obs=f_test, f_exp=f_train/f_train.sum()*f_test.sum()).pvalue
    metrics[1,2] = stats.chisquare(f_obs=f_test, f_exp=f_val/f_val.sum()*f_test.sum()).pvalue
#     print("SS = ", metrics[1,0])
#     print("Train vs. test")
#     print(metrics[1,1])
#     print("Val vs. test")
#     print(metrics[1,2])
    
    # GaussianMixture
    clustering = GaussianMixture(n_components=N_clusters, random_state=0).fit(X)
    results['cluster'] = pd.Categorical(clustering.predict(X))
    
#     plt.figure(figsize=(6,4))
#     sns.scatterplot(
#         x="tsne-2d-one", y="tsne-2d-two",
#         hue="cluster",
#         data=results,
#         palette=sns.color_palette("Set2"),
#         legend="full",
#     )
#     plt.title("Gaussian Mixture")
#     plt.show()
    
    results_df = pd.DataFrame(results)
    f_train = results_df['cluster'][results_df['y']==0].value_counts(sort=False)
    f_val = results_df['cluster'][results_df['y']==1].value_counts(sort=False)
    f_test = results_df['cluster'][results_df['y']==2].value_counts(sort=False)
    
    metrics[2,0] = M.silhouette_score(X,results['cluster'])
    metrics[2,1] = stats.chisquare(f_obs=f_test, f_exp=f_train/f_train.sum()*f_test.sum()).pvalue
    metrics[2,2] = stats.chisquare(f_obs=f_test, f_exp=f_val/f_val.sum()*f_test.sum()).pvalue
#     print("SS = ", metrics[2,0])
#     print("Train vs. test")
#     print(metrics[2,1])
#     print("Val vs. test")
#     print(metrics[2,2])
    print("SS  p-value(T) p-value(V)")
    print(metrics)
    
    return metrics
    
def visualize(X, y, perplexity=30.0, legend=None):
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=300)
    tsne_results = tsne.fit_transform(X)

    results = {'tsne-2d-one': tsne_results[:,0],
           'tsne-2d-two': tsne_results[:,1],
           'y': pd.Categorical(y)}
    
    plt.figure(figsize=(6,4))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        data=results,
        palette=sns.color_palette("Set2"),
        legend="full",
    )
    if legend != None:
        plt.legend(legend)
    plt.show()