from string import punctuation
from nltk.corpus import stopwords
import os
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from collections import OrderedDict
import unidecode
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.models import KeyedVectors
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
import pandas as pd
from numpy.linalg import svd
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
from gensim.models.word2vec import PathLineSentences
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import multiprocessing
import gensim.downloader as api
import re





cores = multiprocessing.cpu_count() # Count the number of cores in a computer

punctuation = list(punctuation)
stop_words = stopwords.words('english')


class Data():

    def __getitem__(self, path=None):
        if path != None:
            with open(path, 'r') as f:
                self.doc = f.read()
            return self.doc

    def _preprocess(self, targets, corpus):
        self.index = []
        self.t_index = OrderedDict()
        for target in targets:

            for _, item in enumerate(corpus):
                if target in item:
                    count_target = item.count(target)
                    # Avoiding the sentences with multiple occurrences of the target term for the time being###
                    if count_target == 1:
                        if target not in self.t_index.keys():
                            self.t_index[target] = [_]
                        else:
                             self.t_index[target].append(_)
                        self.index.append(_)
        return self.index, self.t_index


'''
LOAD & EXTRACT DATA
'''
root_dir = os.getcwd()


p1 = os.path.join(root_dir, 'ccoha1.txt')
p2 = os.path.join(root_dir, 'ccoha2.txt')
t = os.path.join(root_dir, 'targets.txt')
r = os.path.join(root_dir, 'targets_results_.txt')


datasets = Data()  # initialization
doc1 = datasets.__getitem__(p1).split('\n')
doc2 = datasets.__getitem__(p2).split('\n')
t1 = datasets.__getitem__(t).split('\n')
results = datasets.__getitem__(r).split('\n')
results
target_act = [x for x in t1 if len(x) > 1]
t1 = [x.lower() for x in t1 if len(x) > 1]
index1 = datasets._preprocess(t1, doc1)
index2 = datasets._preprocess(t1, doc2)
index_t1 = index1[1]
index_t2 = index2[1]
print('The target words are:', t1)
target_words = t1

# conversions
target_uni = [unidecode.unidecode(m) for m in t1]


# Clean corpus 
def listcorpus(doc):
    corpus = []
    for sentence in doc:
        corpus.append([token for token in word_tokenize(sentence)
                          if token not in stop_words and token not in punctuation])
    return corpus



corpus1 = listcorpus(doc1)
corpus2 = listcorpus(doc2)


print(len(doc1))

# Initialiaze SGNS model
model_1 = Word2Vec(vector_size=100, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                   sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=3)
# build vocabulary
model_1.build_vocab(corpus1, progress_per=1000)

# save the model for later use
model_1.save("word2vec_SGNS_1.model")
# Load the model
model_1 = Word2Vec.load("word2vec_SGNS_1.model")
# Train the model
model_1.train(corpus1, total_examples=model_1.corpus_count, epochs=5)
# save after train
model_1.save("word2vec_SGNS_1.model")




# Initialiaze SGNS model
model_2 = Word2Vec(vector_size=100, window=3, hs=0, min_alpha=0.0007, alpha=0.03,
                   sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=3)
# build vocabulary
model_2.build_vocab(corpus2, progress_per=1000)

# save the model for later use
model_2.save("word2vec_SGNS_2.model")
# Load the model
model_2 = Word2Vec.load("word2vec_SGNS_2.model")
# Train the model
model_2.train(corpus2, total_examples=model_2.corpus_count, epochs=5)
# save after train
model_2.save("word2vec_SGNS_2.model")


# Create a matrix with the word embeddings
def matrix(words, model_1):
    x = {}
    for word in words:
        print(word)
        x[word] = model_1.wv[word]

    p = pd.DataFrame(x)
    return p


embeddings_corpus_1 = matrix(target_uni, model_1)

embeddings_corpus_2 = matrix(target_uni, model_2)


def OP(embeddings_corpus_1, embeddings_corpus_2):

    # Orthogonal Procrustes
    A = embeddings_corpus_1.to_numpy()
    B = embeddings_corpus_2.to_numpy()

    # Orthogonal Procrustes is used to find a matrix R which will allow for a mapping from A to B
    M = np.dot(B, A.transpose())
    u, s, vh = svd(M)
    R = np.dot(u, vh)

    # Transform A using R to map it to B. The transformed matrix A_new = RA
    # Mapped matrix
    new_A = np.dot(R, A)

    return new_A, B


new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)


def cosine_similarity(target_words, new_A, B):

    t = target_words
    output = {}

    for i, word in enumerate(t):
        output[word] = dot(new_A[:, i].transpose(), B[:, i].transpose(
        ))/(norm(new_A[:, i].transpose())*norm(B[:, i].transpose()))

    return output


output = cosine_similarity(target_words, new_A, B)


output


def classify(output):
    s = []
    for i, j in output.items():
        if j > 0.92:
            s.append(0)
        else:
            s.append(1)
    return s


s = classify(output)


def accuracy(s, results):
    count = 0
    for i, word in enumerate(output):
        if s[i] == int(results[i]):
            count += 1
    acc = count/len(results)

    return acc


accuracy(s, results)



# Find common words in two corpuses 

corpus1_unique = set().union(*corpus1)
corpus2_unique = set().union(*corpus2)


def common(s0, s1):
    s0 = s0
    s1 = s1
    return len(list(set(s0)&set(s1))), list(set(s0)&set(s1))


num, common_words = common(corpus1_unique,corpus2_unique)


num

common_words



# Pre-trained word2vec Incremental 

# Load pre-trained word2vec model
pretrained_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Initialize new word2vec model with same parameters as pre-trained model
model = Word2Vec(vector_size=300, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                   sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=3)

# Build vocabulary from your data
model.build_vocab(corpus1, progress_per=1000)

# Set weights of new model to pre-trained model
#model.intersect_word2vec_format('pretrained_model.bin', binary=True, lockf=1.0)
model.wv.vectors = pretrained_model.vectors
model.wv.init_sims(replace=True)

# Fine-tune model on your data
model.train(corpus1, total_examples=model.corpus_count, epochs=10)

# Save fine-tuned model
model.save('fine_tuned_model_1.model')


embeddings_corpus_1 = matrix(target_uni, model)



# Incremental training 


# Build vocabulary from your data
model.build_vocab(corpus2, progress_per=1000)

# Fine-tune model on your data
model.train(corpus2, total_examples=model.corpus_count, epochs=10)

# Save fine-tuned model
model.save('fine_tuned_model_2.model')

embeddings_corpus_2 = matrix(target_uni, model)



# results without alignment 

output = cosine_similarity(target_words, embeddings_corpus_1.to_numpy(), embeddings_corpus_2.to_numpy())

s = classify(output)

accuracy(s, results)


# use alignment 

new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)

output = cosine_similarity(target_words, new_A, B)



# Pretrained with different 

# FOR CORPUS 1

# Load pre-trained word2vec model
pretrained_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Initialize new word2vec model with same parameters as pre-trained model
model = Word2Vec(vector_size=300, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                   sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=3)

# Build vocabulary from your data
model.build_vocab(corpus1, progress_per=1000)

# Set weights of new model to pre-trained model
#model.intersect_word2vec_format('pretrained_model.bin', binary=True, lockf=1.0)
model.wv.vectors = pretrained_model.vectors
model.wv.init_sims(replace=True)

# Fine-tune model on your data
model.train(corpus1, total_examples=model.corpus_count, epochs=10)

embeddings_corpus_1 = matrix(target_uni, model)


# FOR CORPUS 2

# Load pre-trained word2vec model
pretrained_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Initialize new word2vec model with same parameters as pre-trained model
model = Word2Vec(vector_size=300, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                   sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=3)


# Set weights of new model to pre-trained model
#model.intersect_word2vec_format('pretrained_model.bin', binary=True, lockf=1.0)
model.wv.vectors = pretrained_model.vectors
model.wv.init_sims(replace=True)

# Build vocabulary from your data
model.build_vocab(corpus2, progress_per=1000)

# Fine-tune model on your data
model.train(corpus2, total_examples=model.corpus_count, epochs=10)

embeddings_corpus_2 = matrix(target_uni, model)


new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)
output = cosine_similarity(target_words, new_A, B)
s = classify(output)
accuracy(s, results)


output


######################
# Test apple word 
target_words.append('apple')

# Load the model
model_1 = Word2Vec.load("fine_tuned_model_1.model")
model_2 = Word2Vec.load("fine_tuned_model_2.model")

embeddings_corpus_1 = matrix(target_words, model_1)
embeddings_corpus_2 = matrix(target_words, model_2)

# Calculate cosine similarity for all common words 
new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)
output = cosine_similarity(target_words, new_A, B)

output
#######################



#######################
# Visualize words and cosine similarity 

fig, ax = plt.subplots()  # Create a figure containing a single axes.

ax.scatter(output.values(),output.values())
# add labels to all points
for (x, y) in output.items():
    ax.text(y, y, x, va='bottom', ha='center')
ax.set_xlabel('Cosine_similarity')
ax.set_ylabel('Words')
ax.legend()

########################





########################
#Clean dataset from POS

def remove_pos(corpus): 
    clean_corpus = []
    for sentence in corpus: 
        sent = [] 
        for word in sentence: 
            if word.endswith('_nn') or word.endswith('_vb'):
                sent.append(word.split('_')[0])
            else:
                sent.append(word) 
        clean_corpus.append(sent)
    return clean_corpus
   

corpus1_ = remove_pos(corpus1)

corpus2_ = remove_pos(corpus2)


#Clean dataset from POS
def clean_targets(target_words):
    targets = []
    
    for word in target_words: 
        targets.append(word.split('_')[0])
    return targets 

targets = clean_targets(target_words)

########################



##################
# Check only pretrained results 

# Initialize new word2vec model with same parameters as pre-trained model
word2vec_model = Word2Vec(size=300, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                   sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=3)

# Load pre-trained word2vec model
model_wv = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
word2vec_model.build_vocab([list(model_wv.vocab.keys())], update=True)
word2vec_model.intersect_word2vec_format('GoogleNews-vectors-negative300.bin', binary=False, lockf=1.0)





word2vec_model.train(sentences, total_examples=total_examples, epochs=word2vec_model.epochs)





# Load pre-trained word2vec model
pretrained_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Initialize new word2vec model with same parameters as pre-trained model
model = Word2Vec(vector_size=300, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                   sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=3)

# Set weights of new model to pre-trained model
#model.intersect_word2vec_format('pretrained_model.bin', binary=True, lockf=1.0)

model.wv.vectors = pretrained_model.vectors
model.wv.init_sims(replace=True)

embeddings_pretrained_corpus_1 = matrix(targets, model)


model.wv['apple']


targets



corpus1['word_nn']


'word' in corpus1_unique




index1

type(index_t1)

len(index_t1['word_nn'])



target_words


s="word_nn all the information you are apple_nn"
 
re.findall(r'\b(\w+_nn)\b',s)
 
[w for w in s.split() if w.endswith('_nn')]


 
 
 
 
 
 
 
 
 
 




