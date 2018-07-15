import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.sparse.linalg import svds
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

from util.word2vec_as_MF import Word2vecMF

def corr_experiment(model, benchmark, from_folder, ITER=range(10000,5000), plot_corrs=False):
    """
    Aggregator for word similarity correlation experiment.
    """
    
    # Load dataset and model dictionary

    dataset = benchmark.values
    model_vocab = model.vocab

    # Choose only pairs of words which exist in model dictionary
    ind1 = []
    ind2 = []
    vec2 = []
    chosen_pairs = []
    for i in range(dataset.shape[0]):
        try:
            word1 = dataset[i, 0].lower()
            word2 = dataset[i, 1].lower()
        except:
            print(dataset[i,0])
        if (word1 in model_vocab and word2 in model_vocab):
            ind1.append(int(model_vocab[word1]))
            ind2.append(int(model_vocab[word2]))
            vec2.append(np.float64(dataset[i, 2]))
            chosen_pairs.append((word1, word2))
            
    # Calculate correlations
    corrs = []
    vecs = []
    for it in ITER:
    #for it in range(MAX_ITER):
        vec1 = []
        C, W = model.load_CW(from_folder, it)
        
        G = (C.T).dot(W)
        pc = (model.D).sum(axis=1) / (model.D).sum()
        vec1 = (pc.reshape(-1, 1)*G[:,ind1]*G[:,ind2]).sum(axis=0)
        vec1 = list(vec1)
        
        #W = W / np.linalg.norm(W, axis=0)
        #vec1 = (W[:,ind1]*W[:,ind2]).sum(axis=0)
        #vec1 = list(vec1)
        corrs.append(spearmanr(vec1, vec2)[0])
        vecs.append(vec1)
    corrs = np.array(corrs)  
    vecs = np.array(vecs)
    
    # Plot correlations
    if (plot_corrs):
        plots = [corrs, vecs.mean(axis=1), vecs.std(axis=1)]
        titles = ['Correlation', 'Mean', 'Standard deviation']

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for i in xrange(3):
            ax[i].plot(plots[i])
            ax[i].set_title(titles[i], fontsize=14)
            ax[i].set_xlabel('Iterations', fontsize=14)
            ax[i].grid()
    
    return corrs, vecs, vec2, chosen_pairs

def datasets_corr(model, datasets_path, from_folder, MAX_ITER=100, plot_corrs=False, matrix='W', train_ratio=1.0):
    """
    Calculate correlations for all datasets in datasets_path
    """
    
    indices = np.load(open(datasets_path+'/indices.npz', 'rb'))
    sorted_names = ['mc30', 'rg65', 'verb143', 'wordsim_sim', 'wordsim_rel', 'wordsim353', 
                    'mturk287', 'mturk771', 'simlex999', 'rw2034', 'men3000']
    
    # Calculate correlations
    corrs_dict = {}
    #for filename in os.listdir(datasets_path):
        #if filename[-4:]=='.csv':
        
    for name in sorted_names:
        
        corrs = []
        
        pairs_num = indices['0'+name].size
        idx = np.arange(pairs_num)
        np.random.shuffle(idx)
        idx = idx[:int(train_ratio * pairs_num)]
        
        ind1 = indices['0'+name][idx]
        ind2 = indices['1'+name][idx]
        scores = indices['2'+name][idx]

        for it in xrange(MAX_ITER):
            W, C = model.load_CW(from_folder, it)
            if (matrix == 'W'):
                G = W
            else:
                G = (C.T).dot(W)
            G = G / np.linalg.norm(G, axis=0)
            cosines = (G[:,ind1]*G[:,ind2]).sum(axis=0)
            corrs.append(spearmanr(cosines, scores)[0])

        corrs = np.array(corrs)  
        corrs_dict[name] = corrs
            
    # Plot correlations
    if (plot_corrs):
        
        #w2v_ds_corrs = datasets_corr(model, datasets_path, 'enwik-200/w2v/factors', MAX_ITER=1, plot_corrs=False)
        #svd_ds_corrs = datasets_corr(model, datasets_path, 'enwik-200/PS800iter_fromSVD_factors', MAX_ITER=1, plot_corrs=False)
        
        fig, ax = plt.subplots(4, 3, figsize=(15, 20))
        for num, name in enumerate(sorted_names):
            x = ax[num/3,num%3]

            x.plot(corrs_dict[name], lw=2, label='opt method')
            x.set_title(name, fontsize=14)

            # plot original word2vec correlation
            #w2v_corr = w2v_ds_corrs[name][0]
            #x.plot((0, MAX_ITER), (w2v_corr, w2v_corr), 'k-', color='red', lw=2, label='SGNS')

            # plot SVD correlation
            #svd_corr = svd_ds_corrs[name][0]
            #x.plot((0, MAX_ITER), (svd_corr, svd_corr), 'k-', color='green', lw=2, label='SVD')

            x.legend(loc='best')
            x.grid()
    
    
    return corrs_dict