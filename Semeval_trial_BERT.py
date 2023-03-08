import os
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from collections import OrderedDict
import unidecode
import numpy as np
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
# logging.basicConfig(level=logging.INFO)
import os
import matplotlib.pyplot as plt


# set device to cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

torch.cuda.get_device_name(0)

# import pre-trained model

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Use the pre-trained Base BERT model
model = BertModel.from_pretrained('bert-base-uncased')
model.cuda()
# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

root_dir = os.getcwd()

# Convert txt.gz to txt
# uncomment to run
'''
with gzip.open('ccoha1.txt.gz', 'rb') as fin:
    with open('ccoha1.txt', 'wb') as fout:
        data = fin.read()
        fout.write(data)
        
with gzip.open('ccoha2.txt.gz', 'rb') as fin:
    with open('ccoha2.txt', 'wb') as fout:
        data = fin.read()
        fout.write(data)
'''

# class for


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
                #   Avoiding the sentences with multiple occurrences of the target term for the time being###
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
target_toks = []

# print(target_uni)
for k in t1:
    target_toks.append(tokenizer.tokenize(k))

print(" ")
print('converted target toks', target_toks)


# BERT functions

def _pre_bert(doc, index, t):

    s = [doc[ind] for ind in index[t]]
    print('len of sentences', len(s))
    l = len(s)
    marked_text = ["[CLS] " + text + " [SEP]" for text in s]
    tokenized_text = [tokenizer.tokenize(m) for m in marked_text]

    tokenized_text = [x[:512] if len(x) > 512 else x for x in tokenized_text]
    indexed_tokens = [tokenizer.convert_tokens_to_ids(
        x) for x in tokenized_text]
    segments_ids = [[1] * len(x) for x in tokenized_text]
    return s, marked_text, tokenized_text, indexed_tokens, segments_ids, l

# tokenizer.tokenize("contemplation")

def _bert_features(tokens_tensor, segments_tensors, tokenized_text):
    # print(len(tokens_tensor[0]))
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor.to(
            device), segments_tensors.to(device))
    # print ("Number of layers:", len(encoded_layers))
    layer_i = 0

    # print ("Number of batches:", len(encoded_layers[layer_i]))
    batch_i = 0

    # print ("Number of tokens:", len(encoded_layers[layer_i][batch_i]))
    token_i = 0

    # print ("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))
    # Convert the hidden state embeddings into single token vectors

    # Holds the list of 12 layer embeddings for each token
    # Will have the shape: [# tokens, # layers, # features]
    token_embeddings = []

    # For each token in the sentence...
    # tokenized_text=[x for x in tokenized_text if x not in ['_', 'n', '##n','v', '##b']]
    
    for token_i in range(len(tokenized_text)):

        # Holds 12 layers of hidden states for each token
        hidden_layers = []

        # For each of the 12 layers...
        for layer_i in range(len(encoded_layers)):

            # Lookup the vector for `token_i` in `layer_i`
            vec = encoded_layers[layer_i][batch_i][token_i]

            hidden_layers.append(vec)

        token_embeddings.append(hidden_layers)

    # Sanity check the dimensions:
    # print ("Number of tokens in sequence:", len(token_embeddings))
    # print ("Number of layers per token:", len(token_embeddings[0]))
    return token_embeddings
# s,marked_text,tokenized_text,indexed_tokens,segments_ids


def _get_embeddings(pre, tg):
    m_embed_full = []
    # print('len(pre[0])',len(pre[0]))
    # print(tg)
    for _, item in enumerate(pre[0]):
        # Convert inputs to PyTorch tensors
        # print(item)
        token_list = pre[2][_]

        tokens_tensor = torch.tensor([pre[3][_]])
        segments_tensors = torch.tensor([pre[4][_]])

        # Predict hidden states features for each layer
        token_embeddings = _bert_features(
            tokens_tensor, segments_tensors, pre[2][_])
        concatenated_last_4_layers = [torch.cat(
            (layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in token_embeddings]  # [number_of_tokens, 3072]

        summed_last_4_layers = [torch.sum(torch.stack(
            layer)[-4:], 0) for layer in token_embeddings]  # [number_of_tokens, 768]

        # consider the tokenized target

        indxs = []
        # print(token_list)
        for tok in tg:
            '''
            remove -1,-2,-3
            '''
            if tok in token_list:
                if tok not in ['_', 'n', '##n', 'v', '##b']:
                    indxs.append(token_list.index(tok))

        # print('indxs',indxs)
        if len(indxs) == 1:
            bert_embed = concatenated_last_4_layers[indxs[0]]
            m_embed_full.append(bert_embed)
        elif len(indxs) > 1:
            b_emb = []
            for ind in indxs:
                b_emb.append(concatenated_last_4_layers[ind])
            bert_embed = torch.sum(torch.stack(b_emb), 0)
            m_embed_full.append(bert_embed)
        # indx=token_list.index(tg.lower())
        # indx = [i for (i, elem) in enumerate(pre[2][_]) if t in elem]
        # print('indx',indx)
        # print(pre[1][_],indx)

        # if len(indx)>0:
        # bert_embed=concatenated_last_4_layers[indx[0]]

        # cosine_similarity(summed_last_4_layers[10].reshape(1,-1), summed_last_4_layers[19].reshape(1,-1))[0][0]

    return m_embed_full


# [target_words[1]]
def embeddings_extract(target_words, target_toks, doc1, index_t1, doc2, index_t2):
    t = target_words
    X = []
    X_C1 = []
    X_C2 = []
    sents_all = []
    lens1 = []
    lens2 = []
    for k, t in enumerate(target_words):
        berts = []
        sents = []
        print('The target word is', t)

        # get the sentences from corpus c1 and c2 for the specific target word 't'

        pre1 = _pre_bert(doc1, index_t1, t)

        pre2 = _pre_bert(doc2, index_t2, t)
        # lens1.append(pre1[-1])
        # lens2.append(pre2[-1])
        # print(pre1)

        sents.extend(pre1[0])
        sents.extend(pre2[0])
        # aggregate all the embeddings
        # s,marked_text,tokenized_text,indexed_tokens,segments_ids

        '''
    Get the embeddings of the targets from corpus 1 and 2
    '''
        b1 = _get_embeddings(pre1, target_toks[k])
       #print('len of t1', len(b1))
        b2 = _get_embeddings(pre2, target_toks[k])
       #print('len of t2', len(b2))
        '''
    store the lenghts of no. of sentences extracted for each target word for each corpus
    '''
        lens1.append(len(b1))
        lens2.append(len(b2))

        berts.extend(b1)
        berts.extend(b2)
       #print('len of each target word extractions is', len(berts))
        X.append(berts)
        X_C1.append(b1)  # the embeddings for C1
        X_C2.append(b2)  # embeddings for C2
        sents_all.append(sents)
        
      # take the mean value of each sentence for every word embedding
         
        
        
    return X, X_C1, X_C2, lens1, lens2, sents_all


X, X_C1, X_C2, lens1, lens2, sents_all = embeddings_extract(
    target_words, target_toks, doc1, index_t1, doc2, index_t2)


def mean_embeddings(X_C1):

    X_C1_m = []


    for i,_ in enumerate(X_C1):
        x=0
        for j,_ in enumerate(X_C1[i]): 
            x+=_ 
        x = x/len(_)

        X_C1_m.append(x)
        
    return X_C1_m

X_C1_m = mean_embeddings(X_C1)

X_C2_m = mean_embeddings(X_C2)



def cosine_similarity(target_words,X_C1_m,X_C2_m):
    
    t = target_words 
    cos = torch.nn.CosineSimilarity(dim=0,eps=1e-8)
    output = {}
    
    for i,word in enumerate(t):
        output[word] = cos(X_C1_m[i],X_C2_m[i])

    return output

  
output = cosine_similarity(target_words,X_C1_m,X_C2_m)        


def classify(output):
    s = []
    for i,j in output.items(): 
        if j>0.5:
            s.append(0)
        else:
            s.append(1)
    return s 

s = classify(output)


def accuracy(s,results):
    count=0
    for i,word in enumerate(output):
        if s[i]==int(results[i]): 
            count+=1
    acc = count/len(results)
    
    return acc 

accuracy(s,results)

