
# coding: utf-8

# ## Imports

# In[1]:


'''%matplotlib inline
%load_ext autoreload
%autoreload 2'''
import matplotlib.pyplot as plt
from IPython import display

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

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


# ## Read and preprocess enwik9

# In[2]:


# %%time
data = []
with open('data/xaa') as file:
# with open('data/xaa') as file:
    for line in file:
        data+= [line[:-1]]


# In[3]:


def wiki_to_wordlist(sentence, remove_stopwords=False ):
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
    return(words)


# In[5]:


# %%time
# sentences = []  # Initialize an empty list of sentences

print("Parsing sentences from training set")

# sentences += [wiki_to_wordlist(sentence)]
sentences = [sentence.split() for sentence in data]
indices = []


# In[6]:


# %%time
for i, sentence in enumerate(sentences):
    if not sentence:
        pass
    else:
        indices.append(i)

real_sentences = np.array(sentences)[indices]


# In[ ]:


'''print('real_sentences', real_sentences[-30:])
print('indices', indices[-30:])
# print('sentences', sentences)'''


# In[7]:


# Create word2vec as matrix factorization model
model_enwik = Word2vecMF()
model_enwik.data_to_matrices(real_sentences, 200, 5, 'enwik-200/matrices.npz')


# In[ ]:


'''# If the model has been already created, load it from file
model_enwik = Word2vecMF()
model_enwik.load_matrices(from_file='enwik-200/matrices.npz')'''


# ## Train ro_sgns model starting from SVD of SPPMI

# In[8]:


# SVD initialization
SPPMI = np.maximum(np.log(model_enwik.D) - np.log(model_enwik.B), 0)
# print SPPMI
u, s, vt = svds(SPPMI, k=200)
C_svd = u.dot(np.sqrt(np.diag(s))).T
W_svd = np.sqrt(np.diag(s)).dot(vt)


# In[10]:


model_enwik.C = C_svd
model_enwik.W = W_svd

model_enwik.save_CW('enwik-200/initializations/SVD_dim200', 0)


# In[ ]:


'''# Train the model
start_time = time.time()
opt_experiment(model_enwik,
               mode='PS', 
               d=200,
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
               d=200,
               eta = 5e-6,
               MAX_ITER=10000,
               from_iter=10000,
               start_from='SVD',
               init=(True, C_svd, W_svd), display=True)
print("--- %s seconds ---" % (time.time() - start_time))'''


# In[ ]:


model_enwik.C = C_svd
model_enwik.W = W_svd
start_time = time.time()
model_enwik.bfgd(d=200,from_iter=10000, MAX_ITER=10000, eta=5e-6, display=True,
                 init=(True, C_svd, W_svd), 
                 save=[True, 'dataset'])
print("--- %s seconds ---" % (time.time() - start_time))


# In[16]:


get_ipython().run_cell_magic('time', '', 'model = Word2Vec(real_sentences, size = 200, compute_loss=True, min_count= 10)')


# In[17]:


model.get_latest_training_loss()


# In[ ]:


'''model = Word2Vec(real_sentences, size=200, window=5, min_count=5, workers=4)
fname = 'original'
model.save(fname)
model = Word2Vec.load(fname)  # you can continue training with the loaded model!'''

