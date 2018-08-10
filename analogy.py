import numpy as np
from sklearn.metrics import pairwise_distances
import time
from util.word2vec_as_MF import Word2vecMF
from util.functions import *
from util.visualize import *
start_time = time.time()
def nearest_neighbors(model, word, exclude=[], metric="cosine"):
    #print(model.W.T.shape, 'model.W.T.shape')
    #print(word.reshape(1, -1).shape, 'word.reshape(1, -1)')
    D = pairwise_distances(model.W.T, word.reshape(1, -1), metric=metric)
    #print(D.shape, 'D.shape')
    for w in exclude:
        D[w] = D.max()
    #print(model.inv_vocab[D.argmin(axis=0)[0]])
    return D.argmin(axis=0)[0]
        
def analogy_score(from_folder, index, evalX):
    from_folder='enwik-200/'+from_folder
    
    model =  Word2vecMF()
    model.vocab = model.load_vocab(from_folder+'/vocab.txt')[1]
    vocab = model.vocab    
    model.C, model.W = model.load_CW(from_folder, index)
    '''normInv = np.array([1./n if n else 0. for n in np.linalg.norm(W, axis=0)])
    W = W*normInv'''

    predict = model.W.T[evalX[:,1]]-model.W.T[evalX[:,0]]+model.W.T[evalX[:,2]] # (:, dim)
    
    closest = np.array([nearest_neighbors(model, p, exclude=[evalX[i,0], evalX[i,1], evalX[i,2]])
                        for i,p in enumerate(predict)])
    
    closest = closest.reshape(-1,)
    #print(model.vocab[evalX[:,:]])
    #eturn predict, (predict == evalX[:,3]).mean(), predict.shape, evalX[:,3].shape
    print(evalX[:,3].shape)
    print(closest.shape)
    print((closest == evalX[:,3]))
    return (closest == evalX[:,3]).mean()

def analogy_dict(from_folder,MAX_ITER=100, plot_corrs=False):
    model =  Word2vecMF()
    model.vocab = model.load_vocab('enwik-200/'+from_folder+'/vocab.txt')[1]
    vocab = model.vocab
    model.C, model.W = model.load_CW('enwik-200/'+from_folder, 20)
    X = []
    with open('benchmark/EN-GOOGLE.txt') as file:
        count =0
        for line in file:
            tokens = line.split()
            if tokens[0] in vocab and tokens[1] in vocab and tokens[2] in vocab and tokens[3] in vocab:
                #print(tokens[0], tokens[1], tokens[2], tokens[3])
                count=count+1
                X.append([vocab[tokens[0]], vocab[tokens[1]], vocab[tokens[2]], vocab[tokens[3]]])
                word = model.W.T[vocab[tokens[1]]]-model.W.T[vocab[tokens[0]]]+model.W.T[vocab[tokens[2]]]
                #print(model.inv_vocab[nearest_neighbors(model, word, exclude=[vocab[tokens[0]], vocab[tokens[1]], vocab[tokens[2]], vocab[tokens[3]]])])
            #if count > 50:break
        X=np.array(X)
        #print(X.shape)
        
    filelist = glob.glob('enwik-200/'+from_folder+'/W*.npz')
    steps = sorted([int(file.split('/')[-1][1:-4]) for file in filelist])
    steps = [step for step in steps if step<MAX_ITER]
    
    analogy= [analogy_score(from_folder=datapath, index=step, evalX=X) for step in steps]
    print(steps, analogy)
    #plt.plot(steps, analogy)
    
    return analogy, steps 
  

label_list= {'PSiter_from9PPM_dim100_step1e-05_0.0': 'BFGD 1.0-8'}
for datapath in label_list.keys():
    print(analogy_dict(datapath, MAX_ITER=100))
print("--- %s seconds ---" % (time.time() - start_time))
