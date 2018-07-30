from until.word2vec_as_MF import Word2vecMF
import numpy as np
data = []

with open('enwik9.txt') as file:
    for line in file:
        data+= [line[:-1]]

def wiki_to_wordlist(sentence, remove_stopwords=True ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.

    # 3. Convert words to lower case and split them
    words = sentence.split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = ['.']
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)
sentences = []  # Initialize an empty list of sentences

print("Parsing sentences from training set")
#data=['Yes','This is a test','.','   ']
for sentence in data:
    sentences += [wiki_to_wordlist(sentence)]

indices = []


real_sentences = [sentence for sentence in sentences if sentence]

model_enwik = Word2vecMF()
model_enwik.data_to_matrices(real_sentences, 100, 5, 'matrices.npz')
