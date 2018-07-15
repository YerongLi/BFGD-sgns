
# coding: utf-8

# ## Imports

# In[1]:


import matplotlib.pyplot as plt
from IPython import display

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from numpy.linalg import norm, svd

import itertools
import pickle
import math
import re

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models import word2vec, Word2Vec
import logging

from word2vec_as_MF import Word2vecMF
from functions import *

import time
dimension = 100


# ## Read and preprocess enwik9

# In[14]:


'''# %%time
data = []
#with open('data/enwik8.txt') as file:
with open('data/x1') as file:
    for line in file:
        data+= [line[:-1]]'''


# In[15]:


'''def wiki_to_wordlist(sentence, remove_stopwords=False ):
        # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.

    # 3. Convert words to lower case and split them
    words = sentence.split()
    #
    # 4. Optionally remove stop words (false by default)
    
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)'''


# In[16]:


'''# %%time
# sentences = []  # Initialize an empty list of sentences

print("Parsing sentences from training set")

# sentences += [wiki_to_wordlist(sentence)]
sentences = [sentence.split() for sentence in data]
indices = []'''


# In[17]:


'''# %%time
for i, sentence in enumerate(sentences):
    if not sentence:
        pass
    else:
        indices.append(i)

real_sentences = np.array(sentences)[indices]'''


# In[18]:


'''print('real_sentences', real_sentences[-30:])
print('indices', indices[-30:])
# print('sentences', sentences)'''


# # Gensim

# In[19]:


'''%%time
skip = Word2Vec(real_sentences, size = dimension, compute_loss=True)'''


# In[20]:


'''skip.get_latest_training_loss()'''


# ## Train ro_sgns model starting from SVD of SPPMI

# In[22]:


'''# Create word2vec as matrix factorization model
model_enwik = Word2vecMF()
model_enwik.data_to_matrices(real_sentences, dimension, 5, 'enwik-200/matrices.npz')'''


# In[2]:


# If the model has been already created, load it from file
model_enwik = Word2vecMF()
model_enwik.load_matrices(from_file='enwik-200/matrices1.npz')


# In[3]:


# SVD initialization
SPPMI = np.maximum(np.log(model_enwik.D) - np.log(model_enwik.B), 0)
# print SPPMI
u, s, vt = svds(SPPMI, k=dimension)
C_svd = u.dot(np.sqrt(np.diag(s))).T
W_svd = np.sqrt(np.diag(s)).dot(vt)
print(model_enwik.MF(C_svd, W_svd))


# In[ ]:


'''model_enwik.C = C_svd
model_enwik.W = W_svd
#model_enwik.save_CW('enwik-200/initializations/SVD_dim200', 0)
print(model_enwik.C.shape, model_enwik.W.shape, model_enwik.B.shape, model_enwik.D.shape)'''


# In[11]:


X0 = 1/norm((model_enwik.B+model_enwik.D)/4, 'fro')*model_enwik.grad_MF(
    np.zeros([dimension,model_enwik.B.shape[0]]), np.zeros([dimension,model_enwik.B.shape[0]]))
print(X0.shape)
u, s, vt = svds(X0, k=dimension)
C0 = u.dot(np.sqrt(np.diag(s))).T
W0 = np.sqrt(np.diag(s)).dot(vt)
print(model_enwik.MF(C0, W0))
model_enwik.C = C0
model_enwik.W = W0


# In[ ]:


'''# Train the model
start_time = time.time()
opt_experiment(model_enwik,
               mode='PS', 
               d=dimension,
               eta = 5e-5,
               MAX_ITER=10,
               from_iter=0,
               start_from='SVD',
               init=(True, C_svd, W_svd), display=True)
print("--- %s seconds ---" % (time.time() - start_time))'''


# In[ ]:


'''# Train the model
start_time = time.time()
opt_experiment(model_enwik,
               mode='AM', 
               d=dimension,
               eta = 5e-6,
               lbd = 1.0,
               MAX_ITER=189000,
               from_iter=0,
               start_from='SVD',
               init=(True, C_svd, W_svd), display=True)
print("--- %s seconds ---" % (time.time() - start_time))'''


# In[5]:


# Train the model
start_time = time.time()
opt_experiment(model_enwik,
               mode='PS', 
               d=200,
               eta = 5e-5,
               MAX_ITER=190000,
               from_iter=0,
               start_from='SVD',
               init=(True, C_svd, W_svd), display=True)
print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:


'''model = Word2Vec(real_sentences, size=200, window=5, min_count=5, workers=4)
fname = 'original'
model.save(fname)
model = Word2Vec.load(fname)  # you can continue training with the loaded model!'''

