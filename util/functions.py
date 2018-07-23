import numpy as np
import os

from scipy.sparse.linalg import svds
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

from util.word2vec_as_MF import Word2vecMF
from numpy.linalg import norm, svd

def load(from_file):
    data = []
    with open(from_file) as file:
        for line in file:
            data+= [line[:-1]]

    sentences = []  # Initialize an empty list of sentences

    print("Parsing sentences from training set")
    #data=['Yes','This is a test','.','   ']
    for sentence in data:
        sentences += [sentence.split()]

    real_sentences = [sentence for sentence in sentences if sentence]
    
    return real_sentences
            
            
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
    
def norm_p2(A):
    _, s, _ = svd(A)
    return np.sqrt(sum(np.square(s)))

################################## Projector splitting experiment ##################################
def opt_experiment(model,
                   mode='PS',
                   min_count=200,
                   d = 100,
                   eta = 5e-6,
                   reg = 0.,
                   batch_size = 1000,
                   MAX_ITER = 100,
                   from_iter = 0,
                   start_from = 'RAND',
                   display = False,
                   init = (False, None, None),
                   itv_print=100,
                   itv_save=5000           
                  ):
    """
    Aggregator for projector splitting experiment.
    """ 
    
    # Start projector splitting from svd of SPPMI or random initialization
    
    def folder_path(num_iter):
        #return 'enwik-'+str(min_count)+'/'+mode+str(num_iter)+'iter_from'+start_from+'_dim'+str(d)+'_step'+str(eta)+'_factors'
        return 'enwik-'+str(min_count)+'/'+mode+'iter_from'+start_from+'_dim'+str(d)+'_step'+str(eta)+'_'+str(reg)

    
    from_folder = folder_path(from_iter+MAX_ITER)
    if (from_iter > 0):
        os.rename(folder_path(from_iter), from_folder)  
        C, W = model.load_CW(from_folder, from_iter)
        init_ = (True, C, W)
    else:
        init_ = init
    print(from_folder)
    
    if (mode == 'PS'):
        model.projector_splitting(eta=eta, d=d, MAX_ITER=MAX_ITER, from_iter=from_iter, display=display,
                                  init=init_, save=(True, from_folder), itv_print=itv_print, itv_save=itv_save)
        
    if (mode == 'SPS'):
        model.stochastic_ps(eta=eta, batch_size=batch_size, d=d, MAX_ITER=MAX_ITER, from_iter=from_iter, display=display,
                            init=init_, save=(True, from_folder))
        
    if (mode == 'AM'):
        model.alt_min(eta=eta, d=d, MAX_ITER=MAX_ITER, from_iter=from_iter,
                      init=init_, save=(True, from_folder))
        
    if (mode == 'BFGD'):

        model.bfgd(eta=eta, d=d, reg=reg, MAX_ITER=MAX_ITER, from_iter=from_iter, display=display,
                      init=init_, save=(True, from_folder), itv_print=itv_print, itv_save=itv_save)
    


################################## SPPMI decomposition initialization ##################################

def SPPMI_init(model, dimension):
    SPPMI = np.maximum(np.log(model.D) - np.log(model.B), 0)
    
    u, s, vt = svds(SPPMI, k=dimension)
    C_svd = u.dot(np.sqrt(np.diag(s))).T
    W_svd = np.sqrt(np.diag(s)).dot(vt)
    
    print('Initial loss', model.MF(C_svd, W_svd))
    
    return C_svd, W_svd

################################## Bi-Factorized Gradient Descent initialization ##################################
def BFGD_init(model, dimension, reg=0):
    L=norm((model.B+model.D)/4, 'fro')
    X0 = 1/L*model.grad_MF(
    np.zeros([dimension,model.B.shape[0]]), np.zeros([dimension,model.B.shape[0]]))
    print(X0.shape)
    u, s, vt = svds(X0, k=dimension)
    C0 = u.dot(np.sqrt(np.diag(s))).T
    W0 = np.sqrt(np.diag(s)).dot(vt)
    
    step_size=None
    
    if reg==0:
        norm1=norm_p2(np.concatenate((C0.T, W0.T), axis=0))
        norm2=norm_p2(model.grad_MF(C0, W0))
        step_size = 1/(20*L*(norm1**2)+3*norm2)
        
    print('Initial loss', model.MF(C0, W0), 'theoretical step size', step_size)
    
    
    return C0, W0, step_size



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

    for i in xrange(n):
        if (dropped):
            idx = args_sorted[i]
        else:
            idx = args_sorted[-1-i]
        print("Words:", chosen_pairs[idx])
        print("Assesor score:", vec2[idx])
        print("Distance change:", vecs[from_iter, idx], '-->', vecs[to_iter, idx])
        print('\n')
    
    
def nearest_words_from_iter(model, word, from_folder, top=20, display=False, it=1):
    """
    Get top nearest words from some iteration of optimization method.
    """
    C, W = model.load_CW(from_folder=from_folder, iteration=it)

    model.W = W.copy()
    model.C = C.copy()

    nearest_sum = model.nearest_words(word, top, display)
    
    return nearest_sum

################################## Analogical reasoning experiments ##################################

def argmax_fun(W, indices, argmax_type='levi'):
    """
    cosine: b* = argmax cosine(b*, b - a + a*) 
    levi: b* = argmax cos(b*,a*)cos(b*,b)/(cos(b*,a)+eps)
    """
    
    if (argmax_type == 'levi'):
        W = W / np.linalg.norm(W, axis=0)
        words3 = W[:, indices]
        cosines = ((words3.T).dot(W) + 1) / 2
        obj = (cosines[1] * cosines[2]) / (cosines[0] + 1e-3)
        pred_idx = np.argmax(obj)
        
    elif (argmax_type == 'cosine'):
        words3_vec = W[:, indices].sum(axis=1) - 2*W[:, indices[0]]
        W = W / np.linalg.norm(W, axis=0)
        words3_vec = words3_vec / np.linalg.norm(words3_vec)
        cosines = (words3_vec.T).dot(W)
        pred_idx = np.argmax(cosines)
        
    return pred_idx

def analogical_reasoning(model, dataset, from_folder, it=0):
    """
    Calculate analogical reasoning accuracy for given dataset.
    """
    dic = model.dictionary
    
    _, W = model.load_CW(from_folder, iteration=it)
    W = W / np.linalg.norm(W, axis=0)

    good_sum = 0
    miss_sum = 0

    for words in dataset.values:

        a, b, a_, b_ = words

        if (a in dic and b in dic and a_ in dic and b_ in dic):

            indices = [dic[a], dic[b], dic[a_]]
            
            words3 = W[:, indices]
            cosines = ((words3.T).dot(W) + 1) / 2
            obj = (cosines[1] * cosines[2]) / (cosines[0] + 1e-3)
            pred_idx = np.argmax(obj)
            
            if (model.inv_dict[pred_idx] == b_):
                good_sum += 1
        else: 
            miss_sum += 1

    # calculate accuracy
    acc = (good_sum) / float(dataset.shape[0]-miss_sum)
    
    return acc, miss_sum

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
