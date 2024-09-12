import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.linear_model import SGDOneClassSVM
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

def novelty_detection(model, train_loader, val_loader, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #     algorithm = SGDOneClassSVM(
#         nu=0.5,
#         shuffle=True,
#         tol=1e-6,
#     )
#     algorithm = OneClassSVM(nu=0.3, kernel="rbf", gamma=0.1)
    algorithm = LocalOutlierFactor()
#     algorithm = IsolationForest()
    
    
    X = np.zeros((0,36))
    y = np.zeros(0)
    for batch_idx, (inputs, masks, labels, info) in enumerate(train_loader):
        inputs = inputs.to(device)
        X_val = model.encoder(inputs.unsqueeze(1)).cpu().detach().numpy()
        X_val = np.concatenate((X_val, info[:, 2:6]),axis=1)
        y_val = np.full_like(labels[:,0], 0)
        X = np.concatenate((X, X_val),axis=0)
        y = np.concatenate((y, labels[:,0]),axis=0)
    
    y_pred = algorithm.fit_predict(X)
    
#     tsne = TSNE(n_components=2, verbose=1, n_iter=300)
#     tsne_results = tsne.fit_transform(X)

#     results = {'tsne-2d-one': tsne_results[:,0],
#            'tsne-2d-two': tsne_results[:,1],
#            'y': pd.Categorical(y_pred)}
    
#     plt.figure(figsize=(6,4))
#     sns.scatterplot(
#         x="tsne-2d-one", y="tsne-2d-two",
#         hue="y",
#         data=results,
#         palette=sns.color_palette("Set2"),
#         legend="full",
#     )
    print("Train Outliers")
    print((y_pred==1).sum())
    print((y_pred!=1).sum())
    
    
    X = np.zeros((0,36))
    y = np.zeros(0)
    for batch_idx, (inputs, masks, labels, info) in enumerate(train_loader):
        inputs = inputs.to(device)
        X_train = model.encoder(inputs.unsqueeze(1)).cpu().detach().numpy()
        X_train = np.concatenate((X_train, info[:, 2:6]),axis=1)
        y_train = np.full_like(labels[:,0], 0)
        X = np.concatenate((X, X_train),axis=0)
        y = np.concatenate((y, labels[:,0]),axis=0)
    
    y_pred = algorithm.fit_predict(X)
    
#     tsne = TSNE(n_components=2, verbose=1, n_iter=300)
#     tsne_results = tsne.fit_transform(X)

#     results = {'tsne-2d-one': tsne_results[:,0],
#            'tsne-2d-two': tsne_results[:,1],
#            'y': pd.Categorical(y_pred)}
    
#     plt.figure(figsize=(6,4))
#     sns.scatterplot(
#         x="tsne-2d-one", y="tsne-2d-two",
#         hue="y",
#         data=results,
#         palette=sns.color_palette("Set2"),
#         legend="full",
#     )
    
    print("Val Outliers")
    print((y_pred==1).sum())
    print((y_pred!=1).sum())
    
    
    X = np.zeros((0,36))
    y = np.zeros(0)
    for batch_idx, (inputs, masks, labels, info) in enumerate(test_loader):
        inputs = inputs.to(device)
        X_test = model.encoder(inputs.unsqueeze(1)).cpu().detach().numpy()
        X_test = np.concatenate((X_test, info[:, 2:6]),axis=1)
        y_test = np.full_like(labels[:,0], 2)
        X = np.concatenate((X, X_test),axis=0)
        y = np.concatenate((y, labels[:,0]),axis=0)
        
    y_pred = algorithm.fit_predict(X)
    
    tsne = TSNE(n_components=2, verbose=1, n_iter=300)
    tsne_results = tsne.fit_transform(X)

    results = {'tsne-2d-one': tsne_results[:,0],
           'tsne-2d-two': tsne_results[:,1],
           'y': pd.Categorical(y_pred)}
    
    plt.figure(figsize=(6,4))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        data=results,
        palette=sns.color_palette("Set2"),
        legend="full",
    )
    
    print("Test Outliers")
    print((y_pred==1).sum())
    print((y_pred!=1).sum())
    