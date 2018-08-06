
# coding: utf-8

# ## Imports

# In[18]:


'''%matplotlib inline
%load_ext autoreload
%autoreload 2'''
import numpy as np
from scipy.sparse.linalg import svds
from numpy.linalg import norm, svd

import itertools
import math
import re

#from bs4 import BeautifulSoup
#from nltk.corpus import stopwords
#from gensim.models import word2vec, Word2Vec

from util.word2vec_as_MF import Word2vecMF
from util.functions import *

import time
dimension = 100
negative = 3
regularization =0.0


# ## Read and preprocess enwik9

# In[19]:
# # Gensim

# In[20]:


'''%%time
skip = Word2Vec(real_sentences, size = dimension, compute_loss=True)'''


# In[21]:


'''skip.get_latest_training_loss()'''


# ## Train ro_sgns model starting from SVD of SPPMI

# In[22]:


# Create word2vobject has no attributeec as matrix factorization model
model_enwik = Word2vecMF()
'''model_enwik.data_to_matrices(real_sentences, dimension, 5, 
                             DB_to_file='enwik-200/matrices.npz',
                            indices_to_file='enwik-200/vocab.txt')'''
model_enwik.data_to_matrices('data/x1', dimension, k=negative, 
                             DB_to_file=False,
                            vocab_to_file=False)
#print(model_enwik.load_vocab('enwik-200/vocab.txt')[0])


# In[23]:


# If the model has been already created, load it from file
#model_enwik.load_matrices(from_file='enwik-200/matrices2.npz')


# In[24]:


#C0, W0, step_size = BFGD_init(model_enwik, dimension=dimension, reg=regularization); ini = 'X0'
#del C0, W0
C0, W0, step_size = SPPMI_init(model_enwik, k=negative, dimension=dimension); ini = 'PPM'
model_enwik.C = C0; model_enwik.W = W0

# In[25]:
folder='initializations/SPPMI3'
model_enwik.save_CW(folder, 0)
model_enwik.save_vocab(folder+'/vocab.txt')
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


# In[26]:


'''
# In[ ]:
start_time = time.time()
opt_experiment(model_enwik,
               mode='BFGD',
               d=dimension,
               eta=1e-5,
               #eta = step_size,
               reg = regularization,
               MAX_ITER=20,
               from_iter=0,
               start_from='9'+ini,
               itv_print=1,
               itv_save=1,
               init=(True, C0, W0), 
               autostop=False,
               display=True)
print("--- %s seconds ---" % (time.time() - start_tim'''




# In[ ]:


