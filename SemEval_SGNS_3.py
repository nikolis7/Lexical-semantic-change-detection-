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
from sklearn.metrics.pairwise import cosine_similarity as cos 
from numpy import dot
from numpy.linalg import norm
from gensim.models.word2vec import PathLineSentences
from nltk.tokenize import word_tokenize
import nltk
from nltk.tokenize import punkt 
from nltk.corpus import stopwords
#nltk.download('punkt')
#nltk.download('stopwords')
import multiprocessing
import gensim.downloader as api
import re
from scipy.linalg import orthogonal_procrustes
from gensim.models import Word2Vec
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from scipy.spatial.distance import jensenshannon
from gensim.similarities import WmdSimilarity





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
#############################
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


# Orthogonal Procrustes scipy 
def OP_2(embeddings_corpus_1, embeddings_corpus_2):

    # Orthogonal Procrustes
    A = embeddings_corpus_1.to_numpy()
    B = embeddings_corpus_2.to_numpy()

    R, sca = orthogonal_procrustes(A, B)
    
    new_A = np.dot(A, R.T)
    
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

def common(s0, s1):
    s0 = set().union(*s0)
    s1 = set().union(*s1)
    return len(list(set(s0)&set(s1))), list(set(s0)&set(s1))


num, common_words = common(corpus1_,corpus2_)



#######################################################################################
#######################################################################################
# PIPELINE 

def sgns_pipeline(targets=targets, pretrained=False, alignment= "OP", epochs=10, plot=True):
    
    if pretrained==False:
        # Initialiaze SGNS model
            model_1 = Word2Vec(vector_size=300, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                            sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=3)
            # build vocabulary
            model_1.build_vocab(corpus1_, progress_per=10000)
            # Train the model
            model_1.train(corpus1_, total_examples=model_1.corpus_count, epochs=epochs)
            # Get the embeddings 
            embeddings_corpus_1 = matrix(targets, model_1)
                
            
            if alignment == "OP": 
                # Initialiaze SGNS model
                model_2 = Word2Vec(vector_size=300, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                                sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=3)
                # build vocabulary
                model_2.build_vocab(corpus2_, progress_per=10000)
                # Train the model
                model_2.train(corpus2_, total_examples=model_2.corpus_count, epochs=epochs)
                # Get the embeddings 
                embeddings_corpus_2 = matrix(targets, model_2)
                
                
                # Alignment 
                new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)
                output = cosine_similarity(targets, new_A, B)
                #Results 
                s = classify(output)
                accuracy(s, results)
                
                if plot==True: 
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
            
            else: 
                # Incremental learning 
                
                # Build vocabulary from your data for corpus 2 
                model_1.build_vocab(corpus2_, progress_per=10000, update=True)
                # Fine-tune model on your data
                model_1.train(corpus2_, total_examples=model_1.corpus_count, epochs=epochs)    
                # get embeddings 
                embeddings_corpus_2 = matrix(targets, model_1)
                # results without alignment 
                output = cosine_similarity(targets, embeddings_corpus_1.to_numpy(), embeddings_corpus_2.to_numpy())
                s = classify(output)
                accuracy(s, results)
                
                if plot==True: 
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
    else: 
            # pre-trained 
                    
            # Load pre-trained word2vec model
            pretrained_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
            # Initialize new word2vec model with same parameters as pre-trained model
            model = Word2Vec(vector_size=300, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                            sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=3)
            # Build vocabulary from your data
            model.build_vocab(corpus1_, progress_per=10000)
            # Set weights of new model to pre-trained model
            model.wv.vectors = pretrained_model.vectors
            # Fine-tune model on your data
            model.train(corpus1_, total_examples=model.corpus_count, epochs=epochs)    
            # Get the embeddings 
            embeddings_corpus_1 = matrix(targets, model) 
            
            if alignment == "OP": 
                # Initialiaze SGNS model
                model_2 = Word2Vec(vector_size=300, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                                sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=3)
                # build vocabulary
                model_2.build_vocab(corpus2_, progress_per=10)
                # Set weights of new model to pre-trained model
                model_2.wv.vectors = pretrained_model.vectors
                # Train the model
                model_2.train(corpus2_, total_examples=model_2.corpus_count, epochs=epochs)
                # Get the embeddings 
                embeddings_corpus_2 = matrix(targets, model_2)
                
                # Alignment 
                new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)
                output = cosine_similarity(targets, new_A, B)
                #Results 
                s = classify(output)
                accuracy(s, results)
                
                if plot==True: 
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
            
            else: 
                # Incremental learning 
                
                # Build vocabulary from your data for corpus 2 
                model.build_vocab(corpus2_, progress_per=10000, update=True)
                # Fine-tune model on your data
                model.train(corpus2_, total_examples=model_1.corpus_count, epochs=epochs)    
                # get embeddings 
                embeddings_corpus_2 = matrix(targets, model)
                # results without alignment 
                output = cosine_similarity(targets, embeddings_corpus_1.to_numpy(), embeddings_corpus_2.to_numpy())
                s = classify(output)
                accuracy(s, results)
                
                if plot==True: 
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
                                            
    return print('run completed!') 

#######################################################################################
#######################################################################################



#######################################################################################
# Try pipeline 
sgns_pipeline(targets=targets, pretrained=False, alignment="OP", epochs=2, plot=True)
#######################################################################################



########################
# 1) Run only with own train model  with alignment 

# Initialiaze SGNS model
model_1 = Word2Vec(vector_size=300, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                   sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=3)
# build vocabulary
model_1.build_vocab(corpus1_, progress_per=10000)

# save the model for later use
#model_1.save("word2vec_SGNS_1.model")
# Load the model
#model_1 = Word2Vec.load("word2vec_SGNS_1.model")
# Train the model
model_1.train(corpus1_, total_examples=model_1.corpus_count, epochs=100)

# save after train
model_1.save("word2vec_SGNS_1.model")

# Load the model
model_1 = Word2Vec.load("word2vec_SGNS_1.model")


# Extract embeddings 
embeddings_corpus_1 = matrix(targets, model_1)

# embeddings_corpus_all_1 = matrix(common_words,model_1)


model_2 = Word2Vec(vector_size=300, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                   sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=3)
# build vocabulary
model_2.build_vocab(corpus2_, progress_per=10000)

# Train the model
model_2.train(corpus1_, total_examples=model_2.corpus_count, epochs=100)

# save after train
model_2.save("word2vec_SGNS_2.model")

# Load the model
model_2 = Word2Vec.load("word2vec_SGNS_2.model")


# Extract embeddings 
embeddings_corpus_2 = matrix(targets, model_2)

#embeddings_corpus_all_2 = matrix(common_words,model_2)

# Alignment 
new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)
output = cosine_similarity(targets, new_A, B)
output
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


#### Run with scipy alignment 

new_A, B = OP_2(embeddings_corpus_1, embeddings_corpus_2)
output = cosine_similarity(targets, new_A, B)
output



########################
# 2) Run only with own train model  with incremental learning & alignment 

model_1 = Word2Vec(vector_size=300, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                   sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=3)
# build vocabulary
model_1.build_vocab(corpus1_, progress_per=1000)

# Train the model
model_1.train(corpus1_, total_examples=model_1.corpus_count, epochs=100)
# save after train
model_1.save("word2vec_SGNS_1.model")

#Load the model
model_1 = Word2Vec.load("word2vec_SGNS_1.model")


embeddings_corpus_1 = matrix(targets, model_1)

#embeddings_corpus_all_1 = matrix(common_words,model_1)


# Build vocabulary from your data for corpus 2 
model_1.build_vocab(corpus2_, progress_per=10000, update=True)

# Fine-tune model on your data
model_1.train(corpus2_, total_examples=model_1.corpus_count, epochs=100)

# Save fine-tuned model
model_1.save('word2vec_SGNS_2_incremental.model')

#Load the model
model_1 = Word2Vec.load("word2vec_SGNS_2_incremental.model")

embeddings_corpus_2 = matrix(targets, model_1)


# Alignment 
new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)
output = cosine_similarity(targets, new_A, B)
output
#Results 
s = classify(output)
accuracy(s, results)




######################################
######################################
# 3) Use pretrained model with incremental learning & Alignment 

# Load pre-trained word2vec model
pretrained_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Initialize new word2vec model with same parameters as pre-trained model
model = Word2Vec(vector_size=300, window=10,min_count=1, workers=cores-1, sg=1, negative=3)

# Set weights of model to pre-trained vectors
#model.wv.add_vectors(pretrained_model.index_to_key, pretrained_model.vectors)
model.wv.vectors = pretrained_model.vectors

# Build vocabulary from your data
model.build_vocab(corpus1_, progress_per=10000, min_count=1)

# Fine-tune model on your data
model.train(corpus1_, total_examples=model.corpus_count, epochs=100)

# Save fine-tuned model
model.save('pre_trained_fine_tuned_model_1.model')

#Load the model
model = Word2Vec.load("pre_trained_fine_tuned_model_1.model")

embeddings_corpus_1 = matrix(targets, model)


# incremental

# Build vocabulary from your data for corpus 2 
model.build_vocab(corpus2_, progress_per=10000, min_count=1, update=True)

#model.wv.add_vectors(corpus2_, ignore_missing=True)

# Fine-tune model on your data
model.train(corpus2_, total_examples=model.corpus_count, epochs=100)

# Save fine-tuned model
model.save('pre_trained_fine_tuned_model_2_incremental.model')

# Load the model
model = Word2Vec.load("pre_trained_fine_tuned_model_2_incremental.model")

embeddings_corpus_2 = matrix(targets, model)


# Alignment 
new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)
output = cosine_similarity(targets, new_A, B)
output
#Results 
s = classify(output)
accuracy(s, results)


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
model_2 = Word2Vec(vector_size=300, window=10, min_count=1, workers=cores-1, sg=1, negative=3)

# Build vocabulary from your data
model_2.build_vocab(corpus2_, progress_per=1000)

# Set weights of new model to pre-trained model
#model.intersect_word2vec_format('pretrained_model.bin', binary=True, lockf=1.0)
model_2.wv.vectors = pretrained_model.vectors

# Fine-tune model on your data
model_2.train(corpus2_, total_examples=model_2.corpus_count, epochs=100)

embeddings_corpus_2 = matrix(targets, model_2)

# Alignment 
new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)
output = cosine_similarity(targets, new_A, B)

#Results 
s = classify(output)
accuracy(s, results)

output


# Find k-nearest words of selected words based on similarity 




# Preprocess your corpus into a list of list of words
#corpus = [['word', 'list', 'document', ...], ['another', 'document', 'in', ...], ...]

# Train a Word2Vec model on your preprocessed corpus
#model = Word2Vec(corpus, size=100, window=5, min_count=1, workers=4)

# Select target words for analysis
#target_words = ['selected', 'words', 'to', 'analyze']

# Retrieve word vectors for target words for corpus 1 
word_vectors_1 = [model_1.wv[word] for word in targets]

# Create a similarity index for the Word2Vec model
similarity_index = WordEmbeddingSimilarityIndex(model_1.wv)

# Define the number of nearest words to retrieve
k = 5

# Derive the k nearest words for each target word

knn_words = {}

for i, word in enumerate(targets):
    similarities = similarity_index.most_similar(word, topn=k)
    nearest_words = sorted(similarities, key=lambda item: -item[1])[:k]
    knn_words[word] = nearest_words
    print(f"Nearest words to '{word}': {nearest_words}")
    


targets 

knn_words['word']







# Retrieve word vectors for target words in both models
word_vectors1 = [model_1.wv[word] for word in targets]
word_vectors2 = [model_2.wv[word] for word in targets]

# Create a similarity index for the first Word2Vec model
similarity_index1 = WordEmbeddingSimilarityIndex(model_1.wv)

# Create a similarity index for the second Word2Vec model
similarity_index2 = WordEmbeddingSimilarityIndex(model_2.wv)

# Define the number of nearest words to retrieve
k = 5

# Calculate JSD score for each target word
jsd_scores = []
for i, word in enumerate(targets):
    # Retrieve k nearest words for the first corpus
    similarities1 = similarity_index1.most_similar(word, topn=k)
    nearest_words1 = [(w, sim) for w, sim in similarities1 if w in model_1.wv.vocab]

    # Retrieve k nearest words for the second corpus
    similarities2 = similarity_index2.most_similar(word, topn=k)
    nearest_words2 = [(w, sim) for w, sim in similarities2 if w in model_2.wv.vocab]

    # Extract the nearest word vectors
    nearest_word_vectors1 = np.array([model_1.wv[word] for word, _ in nearest_words1])
    nearest_word_vectors2 = np.array([model_2.wv[word] for word, _ in nearest_words2])

    # Calculate the probability distributions for the nearest words
    distribution1 = np.array([sim for _, sim in nearest_words1])
    distribution2 = np.array([sim for _, sim in nearest_words2])

    # Normalize the probability distributions
    distribution1 /= np.sum(distribution1)
    distribution2 /= np.sum(distribution2)

    # Calculate the JSD score between the distributions
    jsd = jensenshannon(distribution1, distribution2)

    jsd_scores.append(jsd)

# Print the JSD scores for each target word
for word, score in zip(targets, jsd_scores):
    print(f"JSD score for '{word}': {score}")
    
    
    
    
    
    
    
    
    
    
    
  
# Retrieve word vectors for target words in both models
word_vectors1 = [model_1.wv[word] for word in targets]
word_vectors2 = [model_2.wv[word] for word in targets]

# Define the number of nearest words to retrieve
k = 5

# Calculate JSD score for each target word
jsd_scores = []
for i, word in enumerate(targets):
    # Retrieve k nearest words for the first corpus
    nearest_words1 = model_1.wv.most_similar(word, topn=k)

    # Retrieve k nearest words for the second corpus
    nearest_words2 = model_2.wv.most_similar(word, topn=k)

    # Extract the nearest word vectors
    nearest_word_vectors1 = np.array([model_1.wv[word] for word, _ in nearest_words1])
    nearest_word_vectors2 = np.array([model_2.wv[word] for word, _ in nearest_words2])

    # Calculate the probability distributions for the nearest words
    distribution1 = np.array([sim for _, sim in nearest_words1])
    distribution2 = np.array([sim for _, sim in nearest_words2])
distribution1
    # Normalize the probability distributions
    distribution1 /= np.sum(distribution1)
    distribution2 /= np.sum(distribution2)

    # Calculate the JSD score between the distributions
    jsd = jensenshannon(distribution1, distribution2)

    jsd_scores.append(jsd)

# Print the JSD scores for each target word
for word, score in zip(targets, jsd_scores):
    print(f"JSD score for '{word}': {score}")