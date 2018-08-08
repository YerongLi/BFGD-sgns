import numpy as np
import os, glob

from scipy.sparse.linalg import svds
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

from util.word2vec_as_MF import Word2vecMF
from numpy.linalg import norm, svd
           

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
                   autostop = False,
                   init = (False, None, None),
                   itv_print=100,
                   itv_save=5000,
                   tol=80,           
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
        C, W = model.load_CW(from_folder, from_iter)
        init_ = (True, C, W)
    else:
        init_ = init
    print(from_folder)
    
    if (mode == 'PS'):
        model.projector_splitting(eta=eta, d=d, MAX_ITER=MAX_ITER, from_iter=from_iter, display=display,
                                  init=init_, save=(True, from_folder), itv_print=itv_print, itv_save=itv_save, 
                                  autostop= autostop, tol=tol)
        
    if (mode == 'SPS'):
        model.stochastic_ps(eta=eta, batch_size=batch_size, d=d, MAX_ITER=MAX_ITER, from_iter=from_iter, display=display,
                            init=init_, save=(True, from_folder))
        
    if (mode == 'AM'):
        model.alt_min(eta=eta, d=d, MAX_ITER=MAX_ITER, from_iter=from_iter,
                      init=init_, save=(True, from_folder))
        
    if (mode == 'BFGD'):

        model.bfgd(eta=eta, d=d, reg=reg, MAX_ITER=MAX_ITER, from_iter=from_iter, display=display,
                      init=init_, save=(True, from_folder), itv_print=itv_print, itv_save=itv_save,
                      tol=tol)

def RAND_init(model, dimension):
    
    
    C0 = (np.random.rand(dimension,model.D.shape[0])-0.5)/dimension
    W0 = (np.random.rand(dimension,model.D.shape[1])-0.5)/dimension
    
    RAND = C0.T @ W0
    u, s, vt = svds(RAND, k=dimension)
    C0 = u.dot(np.sqrt(np.diag(s))).T
    W0 = np.sqrt(np.diag(s)).dot(vt)    
    
    
    return C0, W0        


################################## SPPMI decomposition initialization ##################################

def SPPMI_init(model, dimension, negative):
    
    SPPMI = np.maximum(np.nan_to_num(np.log(model.D) - np.log(model.B)),0)
    # SPPMI = np.log(model.D) - np.log(model.B)
    
    np.savez(open(str(negative)+'debug.npz', 'wb'), Sr1=SPPMI[0], Sc1=SPPMI[:,0])
    print(np.count_nonzero(SPPMI)/SPPMI.shape[0]**2)
    print(norm(SPPMI, 'fro'))
    '''
    print(SPPMI[0], 'SPPMI')
    print(np.log(model.D)[0], 'logD')
    print(np.log(model.B)[0], 'logB')'''
    
    u, s, vt = svds(SPPMI, k=dimension)
    C0 = u.dot(np.sqrt(np.diag(s))).T
    W0 = np.sqrt(np.diag(s)).dot(vt)
    
    print('Initial loss', model.MF(C0, W0))
    
    return C0, W0

def SPPMI_biased_init(model, dimension, negative):
    
    SPPMI = np.maximum(np.nan_to_num(np.log(model.D) - np.log(model.B)),0)
    # SPPMI = np.log(model.D) - np.log(model.B)
    
    #np.savez(open(str(negative)+'debug.npz', 'wb'), Sr1=SPPMI[0], Sc1=SPPMI[:,0])
    print(np.count_nonzero(SPPMI)/SPPMI.shape[0]**2)
    print(norm(SPPMI, 'fro'))
    '''
    print(SPPMI[0], 'SPPMI')
    print(np.log(model.D)[0], 'logD')
    print(np.log(model.B)[0], 'logB')'''
    
    u, s, vt = svds(SPPMI, k=dimension)
    C0 = u.dot(np.sqrt(np.diag(s))).T
    W0 = np.sqrt(np.diag(s)).dot(vt)
    
    print('Initial loss', model.MF(C0, W0))
    
    return C0, W0

################################## Bi-Factorized Gradient Descent initialization ##################################
def BFGD_init(model, dimension):

    
    L=norm((model.B+model.D)/4, 'fro')
    '''
    X0=C0.T @ W0,  Vc x Vw
    ''' 
    X0 = 1/L*model.grad_MF(
    np.zeros([dimension,model.B.shape[0]]), np.zeros([dimension,model.B.shape[1]]))
    
    print(X0.shape)
    
    u, s, vt = svds(X0, k=dimension)
    
    '''
    C0, context matrix, d x Vc
    W0,    word matrix, d x Vw
    '''
    C0 = u.dot(np.sqrt(np.diag(s))).T
    W0 = np.sqrt(np.diag(s)).dot(vt)
    
    print('Initial loss', model.MF(C0, W0))
    return C0, W0

def calculate_step_size(model, reg=0):
    step_size=None
    C0=model.C; W0=model.W
    L=norm((model.B+model.D)/4, 'fro')
    if reg==0:
        norm1=norm(np.concatenate((C0.T, W0.T), axis=0), ord=2)
        norm2=norm(model.grad_MF(C0, W0), ord=2)
        step_size = 1/(20*L*(norm1**2)+3*norm2)
    
    print('theoretical step size', step_size)
    return step_size

    
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

def svd_rev(from_folder, MAX_ITER=10, plot_corrs=False, matrix='W', train_ratio=1.0):
    """
    Calculate correlations for all datasets in datasets_path
    """
    
    filelist = glob.glob(from_folder+'/W*.npz')
    
    directory =  from_folder.split('/')[0]+'/rev'+from_folder.split('/')[1]
    model =  Word2vecMF()
    model.vocab = model.load_vocab(from_folder+'/vocab.txt')[1]
    try:
        os.stat(directory)
        print(directory+' exists.')

    except:
        os.mkdir(directory)
        print(directory+' created.')
    
        
    steps = [int(Wfile.split('/')[2][1:-4]) for Wfile in filelist if int(Wfile.split('/')[2][1:-4])<MAX_ITER ]

        
    model.save_vocab(directory+'/vocab.txt')
    for step in steps:
        try:
            C, W = model.load_CW(from_folder, step)
        except:
            print('Step', step, 'invalid.')

        u, s, vt = svds(C.T @ W , k=C.shape[0])
        model.C = u.dot(np.sqrt(np.diag(s))).T
        model.W = np.sqrt(np.diag(s)).dot(vt)
        model.save_CW(directory, step)

def biased_svd_rev(from_folder, MAX_ITER=10, plot_corrs=False, matrix='W', train_ratio=1.0):
    """
    Calculate correlations for all datasets in datasets_path
    """
    
    filelist = glob.glob(from_folder+'/W*.npz')
    
    directory =  from_folder.split('/')[0]+'/biasedRev'+from_folder.split('/')[1]
    model =  Word2vecMF()
    model.vocab = model.load_vocab(from_folder+'/vocab.txt')[1]
    try:
        os.stat(directory)
        print(directory+' exists.')

    except:
        os.mkdir(directory)
        print(directory+' created.')
    
        
    steps = [int(Wfile.split('/')[2][1:-4]) for Wfile in filelist if int(Wfile.split('/')[2][1:-4])<MAX_ITER ]

        
    model.save_vocab(directory+'/vocab.txt')
    
    for step in steps:
        try:
            C, W = model.load_CW(from_folder, step)
        except:
            print('Step', step, 'invalid.')

        u, s, vt = svds(C.T @ W , k=C.shape[0])
        model.C = u.T
        model.W = (np.diag(s)).dot(vt)
        model.save_CW(directory, step)

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
