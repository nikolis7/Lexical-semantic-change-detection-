from string import punctuation
import os
import torch
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
from nltk.tokenize import punkt 
from nltk.corpus import stopwords
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



#############################
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
# Useful function 

# Create a matrix with the word embeddings
def matrix(words, model):
    x = {}
    for word in words:
        print(word)
        x[word] = model.wv[word]

    p = pd.DataFrame(x)
    return p


# Orthogonal Procrustes 
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


# Cosine Similarity 
def cosine_similarity(target_words, new_A, B):

    output = {}

    for i, word in enumerate(target_words):
        output[word] = dot(new_A[:, i].transpose(), B[:, i].transpose(
        ))/(norm(new_A[:, i].transpose())*norm(B[:, i].transpose()))

    return output


def classify(output):
    s = []
    for i, j in output.items():
        if j > 0.92:
            s.append(0)
        else:
            s.append(1)
    return s


def accuracy(s, results):
    count = 0
    for i, word in enumerate(output):
        if s[i] == int(results[i]):
            count += 1
    acc = count/len(results)

    return acc


# Find common words in two corpuses 

#corpus1_unique = set().union(*corpus1_)
#corpus2_unique = set().union(*corpus2_)


def common(s0, s1):
    s0 = set().union(*s0)
    s1 = set().union(*s1)
    return len(list(set(s0)&set(s1))), list(set(s0)&set(s1))


num, common_words = common(corpus1_,corpus2_)

#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################



########################
# 1) Run only with own train model  with alignment 

# Initialiaze SGNS model
model_1 = Word2Vec(vector_size=100, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                   sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=3)
# build vocabulary
model_1.build_vocab(corpus1_, progress_per=1000)

# save the model for later use
#model_1.save("word2vec_SGNS_1.model")
# Load the model
#model_1 = Word2Vec.load("word2vec_SGNS_1.model")
# Train the model
model_1.train(corpus1_, total_examples=model_1.corpus_count, epochs=10)

# save after train
model_1.save("word2vec_SGNS_1.model")

# Load the model
model_1 = Word2Vec.load("word2vec_SGNS_1.model")


# Extract embeddings 
embeddings_corpus_1 = matrix(targets, model_1)

embeddings_corpus_all_1 = matrix(common_words,model_1)


model_2 = Word2Vec(vector_size=100, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                   sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=3)
# build vocabulary
model_2.build_vocab(corpus2_, progress_per=1000)

# Train the model
model_2.train(corpus1_, total_examples=model_2.corpus_count, epochs=10)

# save after train
model_2.save("word2vec_SGNS_2.model")

# Load the model
model_2 = Word2Vec.load("word2vec_SGNS_2.model")


# Extract embeddings 
embeddings_corpus_2 = matrix(targets, model_2)

embeddings_corpus_all_2 = matrix(common_words,model_2)


# Alignment 
new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)
output = cosine_similarity(targets, new_A, B)

#Results 
s = classify(output)
accuracy(s, results)

# Alignment  for all common words 
new_A, B = OP(embeddings_corpus_all_1, embeddings_corpus_all_2)
output_all = cosine_similarity(targets, new_A, B)


# Visualize words and cosine similarity 

#1
fig, ax = plt.subplots()  # Create a figure containing a single axes.

ax.scatter(output.values(),output.values())
# add labels to all points
for (x, y) in output.items():
    ax.text(y, y, x, va='bottom', ha='center')
ax.set_xlabel('Cosine_similarity')
ax.set_ylabel('Words')
#hide y-axis 
ax.get_yaxis().set_visible(False)
ax.legend()


#2
fig = plt.figure(figsize = (14, 5))
 
# creating the bar plot
plt.bar(targets, output.values(), color ='maroon',
        width = 0.4)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Words")
plt.ylabel("Cosine similarity")
#plt.title("Students enrolled in different courses")
plt.show()


########################
# 2) Run only with own train model  with incremental learning 

model_1 = Word2Vec(vector_size=100, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                   sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=3)
# build vocabulary
model_1.build_vocab(corpus1_, progress_per=1000)

# Train the model
model_1.train(corpus1_, total_examples=model_1.corpus_count, epochs=10)
# save after train
model_1.save("word2vec_SGNS_1.model")

#Load the model
model_1 = Word2Vec.load("word2vec_SGNS_1.model")


embeddings_corpus_1 = matrix(targets, model_1)

embeddings_corpus_all_1 = matrix(common_words,model_1)


# Build vocabulary from your data for corpus 2 
model_1.build_vocab(corpus2_, progress_per=1000, update=True)

# Fine-tune model on your data
model_1.train(corpus2_, total_examples=model_1.corpus_count, epochs=10)

# Save fine-tuned model
model_1.save('word2vec_SGNS_2_incremental.model')

#Load the model
model_1 = Word2Vec.load("word2vec_SGNS_2_incremental.model")

embeddings_corpus_2 = matrix(targets, model_1)

embeddings_corpus_all_2 = matrix(common_words,model_1)


# results without alignment 
output = cosine_similarity(targets, embeddings_corpus_1.to_numpy(), embeddings_corpus_2.to_numpy())
s = classify(output)
accuracy(s, results)





# Visualize words and cosine similarity 

#1
fig, ax = plt.subplots()  # Create a figure containing a single axes.

ax.scatter(output.values(),output.values())
# add labels to all points
for (x, y) in output.items():
    ax.text(y, y, x, va='bottom', ha='center')
ax.set_xlabel('Cosine_similarity')
ax.set_ylabel('Words')
#hide y-axis 
ax.get_yaxis().set_visible(False)
ax.legend()


#2
fig = plt.figure(figsize = (14, 5))
 
# creating the bar plot
plt.bar(targets, output.values(), color ='maroon',
        width = 0.4)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Words")
plt.ylabel("Cosine similarity")
#plt.title("Students enrolled in different courses")
plt.show()

########################################





######################################
######################################
# 3) Use pretrained model with incremental learning 

# Load pre-trained word2vec model
pretrained_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Initialize new word2vec model with same parameters as pre-trained model
model = Word2Vec(vector_size=300, min_count=1, workers=cores-1, sg=1, negative=3)

# Build vocabulary from your data
model.build_vocab(corpus1_, progress_per=1000)

# Set weights of new model to pre-trained model
#model.intersect_word2vec_format('pretrained_model.bin', binary=True, lockf=1.0)
model.wv.vectors = pretrained_model.vectors
#model.wv.init_sims(replace=True)

# Fine-tune model on your data
model.train(corpus1_, total_examples=model.corpus_count, epochs=5)

# Save fine-tuned model
model.save('pre_trained_fine_tuned_model_1.model')

#Load the model
model = Word2Vec.load("pre_trained_fine_tuned_model_1.model")

embeddings_corpus_1 = matrix(targets, model)


# incremental

# Build vocabulary from your data for corpus 2 
model.build_vocab(corpus2_, progress_per=1000)

# Fine-tune model on your data
model.train(corpus2_, total_examples=model.corpus_count, epochs=5)

# Save fine-tuned model
model.save('pre_trained_fine_tuned_model_2_incremental.model')

# Load the model
model = Word2Vec.load("pre_trained_fine_tuned_model_2_incremental.model")

embeddings_corpus_2 = matrix(targets, model)

# Results
output = cosine_similarity(targets, embeddings_corpus_1.to_numpy(), embeddings_corpus_2.to_numpy())
s = classify(output)
accuracy(s, results)


# Visualize words and cosine similarity 

#1
fig, ax = plt.subplots()  # Create a figure containing a single axes.

ax.scatter(output.values(),output.values())
# add labels to all points
for (x, y) in output.items():
    ax.text(y, y, x, va='bottom', ha='center')
ax.set_xlabel('Cosine_similarity')
ax.set_ylabel('Words')
#hide y-axis 
ax.get_yaxis().set_visible(False)
ax.legend()


#2
fig = plt.figure(figsize = (14, 5))
 
# creating the bar plot
plt.bar(targets, output.values(), color ='maroon',
        width = 0.4)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Words")
plt.ylabel("Cosine similarity")
#plt.title("Students enrolled in different courses")
plt.show()

######################################


######################################
######################################
# 4) Pre-trained model with alignment 

# Load the model
model_1 = Word2Vec.load("pre_trained_fine_tuned_model_1.model")

embeddings_corpus_1 = matrix(targets, model_1)

#####
#train independently on corpus_2

# Load pre-trained word2vec model
pretrained_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Initialize new word2vec model with same parameters as pre-trained model
model_2 = Word2Vec(vector_size=300, min_count=1, workers=cores-1, sg=1, negative=3)

# Build vocabulary from your data
model_2.build_vocab(corpus2_, progress_per=1000)

# Set weights of new model to pre-trained model
#model.intersect_word2vec_format('pretrained_model.bin', binary=True, lockf=1.0)
model_2.wv.vectors = pretrained_model.vectors
model_2.wv.init_sims(replace=False)

# Fine-tune model on your data
model_2.train(corpus2_, total_examples=model_2.corpus_count, epochs=10)

embeddings_corpus_2 = matrix(targets, model_2)

# Alignment 
new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)
output = cosine_similarity(targets, new_A, B)

#Results 
s = classify(output)
accuracy(s, results)



# Visualize words and cosine similarity 

#1
fig, ax = plt.subplots()  # Create a figure containing a single axes.

ax.scatter(output.values(),output.values())
# add labels to all points
for (x, y) in output.items():
    ax.text(y, y, x, va='bottom', ha='center')
ax.set_xlabel('Cosine_similarity')
ax.set_ylabel('Words')
#hide y-axis 
ax.get_yaxis().set_visible(False)
ax.legend()


#2
fig = plt.figure(figsize = (14, 5))
 
# creating the bar plot
plt.bar(targets, output.values(), color ='maroon',
        width = 0.4)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Words")
plt.ylabel("Cosine similarity")
#plt.title("Students enrolled in different courses")
plt.show()









#####################################
#####################################
# Check whether the fine-tuning in pre-trained model moves the embeddings 

# Load the model
model_1 = Word2Vec.load("pre_trained_fine_tuned_model_1.model")

embeddings_corpus_1 = matrix(targets, model_1)


# Load pre-trained word2vec model
pretrained_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


# Set weights of new model to pre-trained model
#model.intersect_word2vec_format('pretrained_model.bin', binary=True, lockf=1.0)
model_2.wv.vectors = pretrained_model.vectors
model_2.wv.init_sims(replace=False)


embeddings_corpus_1_pre = matrix(targets, model_2)


output = cosine_similarity(targets, embeddings_corpus_1.to_numpy(), embeddings_corpus_1_pre.to_numpy())


# Alignment 
new_A, B = OP(embeddings_corpus_1, embeddings_corpus_1_pre)
output2 = cosine_similarity(targets, new_A, B)

print(output2)

#####################################
#####################################
# Trial 2

# Load pre-trained word2vec model
pretrained_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Initialize new word2vec model with same parameters as pre-trained model
model = Word2Vec(vector_size=300, min_count=1, workers=cores-1, sg=1, negative=3)

# Build vocabulary from your data
model.build_vocab(corpus1_, progress_per=1000)

# Set weights of new model to pre-trained model
#model.intersect_word2vec_format('pretrained_model.bin', binary=True, lockf=1.0)
model.wv.vectors = pretrained_model.vectors
model.wv.init_sims(replace=False)


embeddings_corpus_1 = matrix(targets, model)



# Fine-tune model on your data
model.train(corpus1_, total_examples=model.corpus_count, epochs=10)

embeddings_corpus_2 = matrix(targets, model)


output = cosine_similarity(targets, embeddings_corpus_1.to_numpy(), embeddings_corpus_2.to_numpy())



#####################################
#####################################














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



s="word_nn all the information you are apple_nn"
 
re.findall(r'\b(\w+_nn)\b',s)
 
[w for w in s.split() if w.endswith('_nn')]


 
 
 
 
 
 
 
 
 
 




