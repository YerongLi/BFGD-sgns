import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import glob
from scipy.sparse.linalg import svds
from numpy.linalg import svd
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from sklearn.preprocessing import normalize
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

    
    normInv = np.array([1./n if n else 0. for n in np.linalg.norm(W, axis=0)])
    W = W*normInv
    vec1 = (W[:,ind1]*W[:,ind2]).sum(axis=0)

    corr = spearmanr(vec1, vec2)[0]
    #corr=0.7
    '''
    print('debug')
    print(vec1)
    print(np.isfinite(C).all())
    print(np.isfinite(W).all())
    print(np.isfinite(vec1).all())
    print(np.isfinite(vec2).all())
    '''
    return corr, vec1, vec2, chosen_pairs

def corr_word2vec(skip ,benchmark, model_vocab):
    """
    Calculate for word2vec model.
    model_vocab, vocabulary dictionary
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

    corr = spearmanr(vec1, vec2)[0]
    
    return corr, vec1, vec2, chosen_pairs

def datasets_corr(from_folder, MAX_ITER=1000, plot_corrs=False, matrix='W', train_ratio=1.0):
    """
    Calculate correlations for all datasets in datasets_path
    """
    
    sorted_names = ['wordsim_sim', 'wordsim_rel', 'wordsim353','men3000','simlex999', 'rw2034', 'MTURK-771', 'rg65', 'verb143', 'mturk287', 'mc30']
    #sorted_names = ['mc30', 'rg65']
    
    prefix='benchmark/'
    corrs_dict = {}
    filelist = glob.glob(from_folder+'/W*.npz')
    steps = sorted([int(file.split('/')[-1][1:-4]) for file in filelist])
    steps = [step for step in steps if step<MAX_ITER]
    
    model =  Word2vecMF()
    model.vocab = model.load_vocab(from_folder+'/vocab.txt')[1]
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
                print('Step', idx, 'invalid.', end=' ')

        steps = [steps[i] for i,x in enumerate(corrs) if not np.isinf(x)]
        corrs = [x for x in corrs if not np.isinf(x)]
        corrs_dict[name]=(steps, corrs)
        
    column = 2
    row = 6
    fig, axarr = plt.subplots(row, column, sharex=True, figsize=(10, 10))
    fig.tight_layout()
    #fig.set_figheight(10)
    if (plot_corrs):
        for idx, name in enumerate(sorted_names):
            i=idx//column
            j=idx%column
            axarr[i,j].plot(corrs_dict[name][0], corrs_dict[name][1])
            axarr[i,j].set_title(name)
    fig=axarr[0,0].figure
    fig.text(0.01,0.5, "Linguistic scores (Spearman Correlation Scores)", ha="center", va="center",  rotation=90)
    fig.text(0.5,0.0, "Number of Iterations", ha="center", va="center")
    return corrs_dict

def load_sentences(mode='debug'):
    """
    Load training corpus sentences/
    """
    
    if (mode == 'imdb'):
        sentences = pickle.load(open('data/sentences_all.txt', 'rb'))
    elif (mode == 'debug'):
        sentences = pickle.load(open('data/sentences1k.txt', 'rb'))
    elif (mode == 'enwik9'):
        sentences = pickle.load(open('data/enwik9_sentences.txt', 'rb'))
    return sentences

def plot_MF(MFs, x=None, xlabel='Iterations', ylabel='MF'):
    """
    Plot given MFs.
    """
    
    fig, ax = plt.subplots(figsize=(15, 5))
    if not x:
        ax.plot(MFs)
    else:
        ax.plot(x, MFs)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True)



def plot_dynamics(vecs, vec2, n=5, MAX_ITER=100):
    """
    Plot how the distances between pairs change with each n
    iterations of the optimization method.
    """
    
    for i in xrange(MAX_ITER):
        if (i%n==0):
            plt.clf()
            plt.xlim([-0.01, 1.2])
            plt.plot(vecs[i], vec2, 'ro', color='blue')
            plt.grid()
            display.clear_output(wait=True)
            display.display(plt.gcf())
    plt.clf()
    
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for i in xrange(2):
        ax[i].set_xlim([-0.01, 1.2])
        ax[i].plot(vecs[-i], vec2, 'ro', color='blue')
        ax[i].set_title(str(i*MAX_ITER)+' iterations')
        ax[i].set_xlabel('Cosine distance', fontsize=14)
        ax[i].set_ylabel('Assesor grade', fontsize=14)
        ax[i].grid()
    
    
def dist_change(vecs, vec2, chosen_pairs, n=5, dropped=True, from_iter=0, to_iter=-1):
    """
    Get top pairs which change distance between words the most.
    """
    
    vecs_diff = vecs[to_iter, :] - vecs[from_iter, :]
    args_sorted = np.argsort(vecs_diff)

    for i in range(n):
        if (dropped):
            idx = args_sorted[i]
        else:
            idx = args_sorted[-1-i]
        print("Words:", chosen_pairs[idx])
        print("Assesor score:", vec2[idx])
        print("Distance change:", vecs[from_iter, idx], '-->', vecs[to_iter, idx])
        print('\n')


def AR_experiment(model, dataset, from_folder, MAX_ITER=100, step_size=5, plot_accs=False):
    """
    Aggregator for analogical reasoning accuracy experiment.
    """
    
    # Calculate accuracies 
    accs = []
    num_points = MAX_ITER/step_size + 1
    for i in xrange(num_points):
        acc, miss = analogical_reasoning(model, dataset, from_folder, i*step_size)
        accs.append(acc)
    accs = np.array(accs)
    
    # Plot accuracies
    if (plot_accs):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(np.arange(num_points)*step_size, accs)
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.set_xlabel('Iterations', fontsize=14)
        ax.grid()    
        
    return accs, miss