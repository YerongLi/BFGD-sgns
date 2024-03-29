import os
import operator

import numpy as np
from numpy.linalg import svd, qr, norm
from scipy.spatial.distance import cosine
from scipy.sparse.linalg import svds

class Word2vecMF(object):
    
    def __init__(self):
        """
        Main class for working with word2vec as MF.
        
        D -- word-context co-occurrence matrix;
        B -- such matrix that B_cw = k*(#c)*(#w)/|D|;
        C, W -- factors of matrix D decomposition;
        vocab -- vocabulary of words from data;
        inv_vocab -- inverse of dictionary.
        """
        
        self.D = None
        self.B = None 
        self.C = None
        self.W = None
        self.vocab = None
        self.inv_vocab = None

    ############ Create training corpus from raw sentences ################
    
    def create_vocabulary(self, data, r):
        """
        Create a vocabulary from a list of sentences, 
        eliminating words which occur less than r times.
        """

        prevocabulary = {}
        for sentence in data:
            for word in sentence:
                if not word in prevocabulary:
                    prevocabulary[word] = 1
                else:
                    prevocabulary[word] += 1

        vocabulary = {}
        idx = 0
        for word in prevocabulary:
            if (prevocabulary[word] >= r):
                vocabulary[word] = idx
                idx += 1
        
        return vocabulary

    def create_matrix_D(self, data, window_size=5):
        """
        Create a co-occurrence matrix D from training corpus.
        """

        dim = len(self.vocab)
        D = np.zeros((dim, dim))
        s = window_size//2
            
        for sentence in data:
            l = len(sentence)
            for i in range(l):
                for j in range(max(0,i-s), min(i+s+1,l)):
                    if (i != j and sentence[i] in self.vocab
                        and sentence[j] in self.vocab):
                        c = self.vocab[sentence[j]]
                        w = self.vocab[sentence[i]]
                        D[c][w] += 1                  
        return D        
    
    def create_matrix_B(self, k):
        """
        Create matrix B (defined in init).
        """
        
        c_ = self.D.sum(axis=1)
        w_ = self.D.sum(axis=0)
        P = self.D.sum()

        w_v, c_v = np.meshgrid(w_, c_)
        B = k*(w_v*c_v)/P
        return B
        
    ######################### Necessary functions #########################
    
    def sigmoid(self, X):
        """
        Sigmoid function sigma(x)=1/(1+e^{-x}) of matrix X.
        """
        Y = X.copy()
        
        Y[X>20] = 1-1e-6
        Y[X<-20] = 1e-6
        Y[(X<20)&(X>-20)] = 1 / (1 + np.exp(-X[(X<20)&(X>-20)]))
        
        return Y
    
    def sigma(self, x):
        """
        Sigmoid function of element x.
        """
        if (x>20):
            return 1-1e-6
        if (x<-20):
            return 1e-6
        else:
            return 1 / (1 + np.exp(-x))
    
    def MF(self, C, W):
        """
        Objective MF(D,C^TW) we want to minimize.
        """
        
        X = C.T.dot(W)
        MF = self.D*np.log(self.sigmoid(X)) + self.B*np.log(self.sigmoid(-X))
        return -MF.sum(), norm(C.dot(C.T)-W.dot(W.T), 'fro')

    def grad_MF(self, C, W):
        """
        Gradient of the functional MF(D,C^TW) over C^TW.
        """
        
        X = C.T.dot(W)
        #print(self.D.shape, X.shape, self.B.shape)
        grad = self.D*self.sigmoid(-X) - self.B*self.sigmoid(X)
        return grad

    ################# Alternating minimization algorithm ##################
    
    def alt_min(self, eta=1e-7, d=100, MAX_ITER=1, from_iter=0, display=0,
                init=(False, None, None), save=(False, None)):
        """
        Alternating mimimization algorithm for word2vec matrix factorization.
        """
        
        # Initialization
        if (init[0]):
            self.C = init[1]
            self.W = init[2]
        else:
            self.C = np.random.rand(d, self.D.shape[0])
            self.W = np.random.rand(d, self.D.shape[1])  
            
        if (save[0] and from_iter==0):
                self.save_CW(save[1], 0)
                
        for it in range(from_iter, from_iter+MAX_ITER):    
            
            if (display):
                print("Iter #:", it+1)
                
            gradW = (self.C).dot(self.grad_MF(self.C, self.W))
            self.W = self.W + eta*gradW
            gradC = self.W.dot(self.grad_MF(self.C, self.W).T)
            self.C = self.C + eta*gradC
                
            if (save[0]):
                self.save_CW(save[1], it+1)    
    
    def bfgd(self, eta=1e-7, d=100, reg=0.0 ,MAX_ITER=1, from_iter=0, display=False,
                init=(False, None, None), save=(False, None), itv_print=100, itv_save=5000,
                autostop =False, tol=100):
        """
        Alternating mimimization algorithm for word2vec matrix factorization.
        """
        # Initialization
        if (init[0]):
            self.C = init[1]
            self.W = init[2]
        else:
            self.C = np.random.rand(d, self.D.shape[0])
            self.W = np.random.rand(d, self.D.shape[1])
        
        
        if autostop:
            Xt1 = (self.C).T.dot(self.W)
        
        print("Iter #:", from_iter, "loss", self.MF(self.C, self.W))
        
        if (save[0] and from_iter==0):
                self.save_CW(save[1], 0)
                self.save_vocab(save[1]+'/vocab.txt')
                
        for it in range(from_iter, from_iter+MAX_ITER):    
            
 
            G=-reg*0.25*(self.C.dot(self.C.T)-self.W.dot(self.W.T))
            # grad = np.zeros([self.C.shape[1], self.C.shape[1]]) # self.grad_MF(self.C, self.W)
            grad = self.grad_MF(self.C, self.W)
            gradW =  self.C.dot(grad)-G.dot(self.W)
            gradC = self.W.dot(grad.T)+G.dot(self.C)
            print(norm(grad, 'fro'), 'grad')
            print(norm(gradW, 'fro'), 'gradW')
            print(norm(gradC, 'fro'), 'gradC')
            print(norm(self.W, 'fro'), 'C')
            
            self.W = self.W + eta*gradW
            self.C = self.C + eta*gradC
            
            if autostop:
                X = (self.C).T.dot(self.W)
                if norm(X-Xt1,'fro')/norm(X, 'fro')<tol*eta:
                    print("Iter #:", it+1, "loss", self.MF(self.C, self.W))
                    break
                else:
                    Xt1=np.array(X)

            if display and 0==(it+1)%itv_print:
                print("Iter #:", it+1, "loss", self.MF(self.C, self.W))
                
            if save[0] and 0==(it+1)%itv_save:
                self.save_CW(save[1], it+1)     
    #################### Projector splitting algorithm ####################
            
            
    def projector_splitting(self, eta=5e-6, d=100, 
                            MAX_ITER=1, from_iter=0, display=0, 
                            init=(False, None, None), save=(False, None), itv_print=100, itv_save=1000,
                            autostop =False, tol=100):
        """
        Projector splitting algorithm for word2vec matrix factorization.
        """
        
        # Initialization
        if (init[0]):
            self.C = init[1]
            self.W = init[2]
        else:
            self.C = np.random.rand(d, self.D.shape[0])
            self.W = np.random.rand(d, self.D.shape[1])

        print("Iter #:", from_iter, "loss", self.MF(self.C, self.W)) 
        if (save[0] and from_iter==0):
                self.save_CW(save[1], 0)
                self.save_vocab(save[1]+'/vocab.txt')

            
        X = (self.C).T.dot(self.W)

        if autostop:
            Xt1=np.array(X)

        for it in range(from_iter, from_iter+MAX_ITER):
            U, S, V = svds(X, d)
            S = np.diag(S)
            V = V.T
            
            self.C = U.dot(np.sqrt(S)).T
            self.W = np.sqrt(S).dot(V.T)

                     
            F = self.grad_MF(self.C, self.W)
            #mask = np.random.binomial(1, .5, size=F.shape)
            #F = F * mask
            
            U, _ = qr((X + eta*F).dot(V))
            V, S = qr((X + eta*F).T.dot(U))
            V = V.T
            S = S.T
            
            X = U.dot(S).dot(V)

            if autostop:
                if norm(X-Xt1,'fro')/norm(X, 'fro')<tol*eta:
                    print("Iter #:", it+1, "loss", self.MF(self.C, self.W))
                    break
                else:
                    Xt1=np.array(X)

            if display and 0==(it+1)%itv_print:
                print("Iter #:", it+1, "loss", self.MF(self.C, self.W))
           
            if save[0] and 0==(it+1)%itv_save:
                self.save_CW(save[1], it+1)

    
            
    def stochastic_ps(self, eta=5e-6, batch_size=100, d=100, 
                      MAX_ITER=1, from_iter=0, display=False,
                      init=(False, None, None), save=(False, None)):
        """
        Stochastic version of projector splitting."
        """
        if (init[0]):
            self.C = init[1]
            self.W = init[2]
        else:
            self.C = np.random.rand(d, self.D.shape[0])
            self.W = np.random.rand(d, self.D.shape[1])
            
        if (save[0] and from_iter==0):
                self.save_CW(save[1], 0)
                
                
        pw = self.D.sum(axis=0) / self.D.sum()
        pc_w = self.D / self.D.sum(axis=0)
        
        X = (self.C).T.dot(self.W)
        for it in range(from_iter, from_iter+MAX_ITER):
            
            if (display):
                print("Iter #:", it+1)
            
            U, S, V = svds(X, d)
            S = np.diag(S)
            V = V.T
            
            self.C = U.dot(np.sqrt(S)).T
            self.W = np.sqrt(S).dot(V.T)
            
            if (save[0]):
                self.save_CW(save[1], it+1)
                
                
            # Calculate stochastic gradient matrix
            F = np.zeros_like(self.D)
            
            words = np.random.choice(self.D.shape[1], batch_size, p=pw)
            for w in words:
                
                contexts = np.random.choice(self.D.shape[0], 4, p=pc_w[:,w])
                for c in contexts:
                    F[c,w] += self.sigma(X[c, w])
                    
                negatives = np.random.choice(self.D.shape[0], 5, p=pw)
                for c in negatives:
                    F[c,w] -= 0.2 * self.sigma(X[c, w])
                    
            U, _ = qr((X + eta*F).dot(V))
            V, S = qr((X + eta*F).T.dot(U))
            V = V.T
            S = S.T
            
            X = U.dot(S).dot(V)       

    def save_vocab(self, to_file):
        
        sorted_vocab = sorted(self.vocab.items(), key=operator.itemgetter(1))
        vocab_to_save = np.array([item[0] for item in sorted_vocab])
        with open(to_file, 'w') as filehandle:
            # Save context vocabulary
            for listitem in vocab_to_save:
                filehandle.write('%s ' % listitem)
            filehandle.write('\n')
            # Save word vocabulary
            for listitem in vocab_to_save:
                filehandle.write('%s ' % listitem)
                
    def load_vocab(self, from_file):
        file = open(from_file, 'r')
    
        Cvocab = file.readline().split(' ')[:-1]
        Wvocab = file.readline().split(' ')[:-1]
    
        Cvocab = {key: index for index, key in enumerate(Cvocab)}
        Wvocab = {key: index for index, key in enumerate(Wvocab)}
    
        return Cvocab, Wvocab    
    #######################################################################
    ############################## Data flow ##############################
    #######################################################################
    
    ########################## Data to Matrices ###########################
    
    def data_to_matrices(self, from_file, r, k, DB_to_file=False, vocab_to_file=False):
        """
        Process raw sentences, create word dictionary, matrix D and matrix B
        then save them to file.
        """
        sentences = []
        with open(from_file) as file:
            for line in file:
                sentences+= [line[:-1]]

        print("Parsing sentences from training set")
        sentences = [sentence.split() for sentence in sentences]

        sentences = [sentence for sentence in sentences if sentence]      
        self.vocab = self.create_vocabulary(sentences, r)
        self.D = self.create_matrix_D(sentences)
        del sentences
        self.B = self.create_matrix_B(k)
        
        sorted_vocab = sorted(self.vocab.items(), key=operator.itemgetter(1))
        vocab_to_save = np.array([item[0] for item in sorted_vocab])
        
        if not False == DB_to_file:
            np.savez(open(DB_to_file, 'wb'), vocab=vocab_to_save, D=self.D, B=self.B)
        
        if not False == vocab_to_file:
            self.save_vocab(vocab_to_file)
    
    ######################### Matrices to Factors ##########################
 
    def load_matrices(self, from_file):
        """
        Load word dictionary, matrix D and matrix B from file.
        """
        print('Loading matrices from training set')
        matrices = np.load(open(from_file, 'rb'))
        self.D = matrices['D']
        self.B = matrices['B']
        
        self.vocab = {}
        for i, word in enumerate(matrices['vocab']):
            self.vocab[word] = i
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
    def save_CW(self, to_folder, iteration):
        """
        Save factors C and W (from some iteration) to some folder.
        """
        
        if not os.path.exists(to_folder):
            os.makedirs(to_folder)
       
        pref = str(iteration)

        np.savez(open(to_folder+'/C'+pref+'.npz', 'wb'), C=self.C)
        np.savez(open(to_folder+'/W'+pref+'.npz', 'wb'), W=self.W) 
    
    ########################### Factors to MF #############################

    def load_CW(self, from_folder, iteration):
        """
        Load factors C and W (from some iteration) from folder.
        """        
           
        if not os.path.exists(from_folder):
            raise NameError('No such directory')
        
        pref = str(iteration)
        
        C = np.load(open(from_folder+'/C'+pref+'.npz', 'rb'))['C']
        W = np.load(open(from_folder+'/W'+pref+'.npz', 'rb'))['W']
        
        return C, W
    
    def factors_to_MF(self, from_folder, to_file, MAX_ITER, from_iter=0):
        """
        Calculate MF for given sequence of factors C and W
        and save result to some file.
        """
        
        MFs = np.zeros(MAX_ITER)
        
        for it in range(from_iter, from_iter+MAX_ITER):
            C, W = self.load_CW(from_folder, it)
            MFs[it-from_iter] = self.MF(C, W)
        
        np.savez(open(to_file, 'wb'), MF=MFs) 
   
    ############################ MF to Figures ############################
    
    def load_MF(self, from_file):
        """
        Load MFs from file.
        """
        
        MFs = np.load(open(from_file), 'rb')['MF']
        
        return MFs
    
    #######################################################################
    ######################### Linquistic metrics ##########################
    #######################################################################

    def word_vector(self, word, W):
        """
        Get vector representation of a word.
        """
        
        if word in self.vocab:
            vec = W[:,int(self.vocab[word])]
        else:
            print("No such word in vocabulary.")
            vec = None
            
        return vec
    
    def nearest_words(self, word, top=20, display=False):
        """
        Find the nearest words to the word 
        according to the cosine similarity.
        """

        W = self.W / np.linalg.norm(self.W, axis=0)   
        if (type(word)==str):
            vec = self.word_vector(word, W)
        else:
            vec = word / np.linalg.norm(word)
 
        cosines = (vec.T).dot(W)
        args = np.argsort(cosines)[::-1]       
            
        nws = []
        for i in range(1, top+1):
            nws.append(self.inv_vocab[args[i]])
            if (display):
                print(self.inv_vocab[args[i]], round(cosines[args[i]],3))

        return nws