import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import glob
from scipy.sparse.linalg import svds
from numpy.linalg import svd
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

from util.word2vec_as_MF import Word2vecMF

################################## Spearman correlation ########################################
def correlation(model, benchmark, from_folder, index, plot_corrs=False):
    """
    Aggregator for word similarity correlation experiment.
    """
    
    # Load dataset and model dictionary

    dataset = pd.read_csv(benchmark,header=None, delimiter=';').values
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
            
    vec1 = []
    C, W = model.load_CW(from_folder, index)
    
    
    #U, s, V = svd(C.T@W)
    #C = (U@np.diag(s)).T
    #W = (np.diag(s)@V)
    #G = (C.T).dot(W)
    
    #pc = (model.D).sum(axis=1) / (model.D).sum()
    #vec1 = (pc.reshape(-1, 1)*G[:,ind1]*G[:,ind2]).sum(axis=0)
    #vec1 = [1-cosine(W[:, ind1[idx]], W[:, ind2[idx]])for idx in range(len(ind1))]
    #vec1 = list(vec1)
    W = C.T @ W
    W = W / np.linalg.norm(W, axis=0)
    vec1 = (W[:,ind1]*W[:,ind2]).sum(axis=0)
    #vec1 = list(vec1)
    
    '''vec1 = []
    for pair in chosen_pairs:
        word1, word2 = pair
        vec1.append((W[:,int(model_vocab[word1])]*W[:,int(model_vocab[word2])]).sum())'''
    corr = spearmanr(vec1, vec2)[0]
     
    return corr, vec1, vec2, chosen_pairs


def corr_word2vec(skip ,benchmark, model_vocab):
    """
    Aggregator for word similarity correlation experiment.
    """
     
    
    # Load dataset and model dictionary

    dataset = benchmark.values

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
            
    vec1 = []
    for pair in chosen_pairs:
        word1, word2 = pair
        vec1.append(skip.similarity(word1, word2))
        '''v1=skip.wv[word1]    
        v2=skip.wv[word2]
        v1=v1/np.linalg.norm(v1)
        v2=v2/np.linalg.norm(v2)
        vec1.append((v1*v2).sum())'''
    corr = spearmanr(vec1, vec2)[0]
    
    return corr, vec1, vec2, chosen_pairs

def datasets_corr(model, from_folder, MAX_ITER=1000, plot_corrs=False, matrix='W', train_ratio=1.0):
    """
    Calculate correlations for all datasets in datasets_path
    """
    
    sorted_names = ['men3000','rw2034', 'MTURK-771', 'rg65', 'verb143', 'wordsim_sim', 'wordsim_rel', 'wordsim353', 'mturk287',  'simlex999', 'mc30']
    #sorted_names = ['men3000', 'MTURK-771']
    prefix='datasets/'
    corrs_dict = {}
    filelist = glob.glob(from_folder+'/W*.npz')
    steps = sorted([int(file.split('/')[-1][1:-4]) for file in filelist])
    steps = [step for step in steps if step<MAX_ITER]
    for name in sorted_names:
        
        corrs = []
        for idx, step in enumerate(steps):
            try:
                corrs.append(correlation(model=model,
                             benchmark=prefix+name+'.csv',
                             from_folder=from_folder,
                             index=step)[0])
            except:
                corrs.append(np.inf)
                print('Step', idx, ' invalid.', end='')

        steps = [steps[i] for i,x in enumerate(corrs) if not np.isinf(x)]
        corrs = [x for x in corrs if not np.isinf(x)]
        corrs_dict[name]=(steps, corrs)
        
        
    #print(corrs_dict)        
    # Plot correlations
    column = 2
    row = 6
    fig, axarr = plt.subplots(row, column)
    fig.tight_layout()
    fig.set_figheight(10)
    if (plot_corrs):
        for idx, name in enumerate(sorted_names):
            i=idx//column
            j=idx%column
            axarr[i,j].plot(corrs_dict[name][0], corrs_dict[name][1])
            axarr[i,j].set_title(name)
            
    
    
    
    return corrs_dict