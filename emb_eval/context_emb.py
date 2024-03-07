'''
Created on 14 Jan 2020
@author: camilo thorne
'''


import numpy as np
from scipy.spatial.distance import cosine
from allennlp.commands.elmo import ElmoEmbedder
from ner_eval.NERpredict import corpus_reader

from gensim import models as gmod
from gensim.models import KeyedVectors
import time
from tqdm.autonotebook import tqdm
from numpy import float32 as REAL
from gensim import utils

import matplotlib
matplotlib.use('macosx')
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def getSentences(word, corpus):
    '''
    get all (tokenized) sentences where a token occurs
    returns a set of sentences
    '''
    sents = []
    for sent in corpus:
        if word in sent:
            sents.append(sent)
    return sents


def getEmbeddings(word, sents, elmo_emb):
    '''
    run ELMo embedding on sentences and for each, recover
    the embeddings for the target word
    returns a pair (word, embedding_set)
    '''
    #print('word: \t', word)
    embs = []
    for tokens in sents:
        # average the 3 layers returned from ELMo
        #print('\t embedding:\t', tokens)
        layers  = np.average(elmo_emb.embed_sentence(tokens), axis=0)
        ind     = tokens.index(word)
        embs.append(layers[ind])
    return (word, np.stack(embs,axis=0))


def combineEmbeddings(word, embeddings):
    '''
    aggregate all embeddings (related to one word)
    returns a pair (word, embedding)
    '''
    msum = np.average(embeddings,axis=0)
    return (word, msum)


def computeSimMatrix(word_embs):
    '''
    compute similarity matrix
    '''
    mat = np.zeros(shape=(len(word_embs),len(word_embs)))
    for i in range(0,len(word_embs)):
        for j in range(i, len(word_embs)):
            cos = cosine(word_embs[i][1], word_embs[j][1])
            mat[i][j] = (1 - cos)
    words = [t for (t,_) in word_embs]
    #print(mat)
    return (words, mat)


def get_most_similar(word, simmatrix, title, K):
    '''
    return top K similarity scores
    '''
    (tokens, mat) = simmatrix
    ind           = tokens.index(word)
    values        = mat[ind][:]
    sim_x           = {}
    for i in range(ind+1, len(values)):
        sim_x[tokens[i]] = values[i]
    sim = {k: v for k, v in sorted(sim_x.items(), key=lambda item: item[1], reverse=True)}   
    print('--------------------')
    print(word, '--', title)
    print('--------------------') 
    for t in list(sim.keys())[0:K]: 
        print(t,':\t',sim[t])
        
        
def dict2w2v(mdict, path):
    '''
    convert dictionary to W2V model, and save
    '''
    m = gmod.keyedvectors.Word2VecKeyedVectors(vector_size=1024)
    m.vocab = mdict
    m.vectors = np.array(list(mdict.values()))
    print('saving', path)
    my_save_word2vec_format(binary=True, fname='../data/' + path, total_vec=len(mdict), vocab=m.vocab, vectors=m.vectors)
    return m       


def my_save_word2vec_format(fname, vocab, vectors, binary=True, total_vec=2):
    '''
    Store the input-hidden weight matrix in the same format used by the original
    C word2vec-tool, for compatibility.
    ----------
    Parameters
    ----------
    fname : str
        The file path used to save the vectors in.
    vocab : dict
        The vocabulary of words.
    vectors : numpy.array
        The vectors to be stored.
    binary : bool, optional
        If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
    total_vec : int, optional
        Explicitly specify total number of vectors
        (in case word vectors are appended with document vectors afterwards).
    '''
    if not (vocab or vectors):
        raise RuntimeError("no input")
    if total_vec is None:
        total_vec = len(vocab)
    vector_size = vectors.shape[1]
    assert (len(vocab), vector_size) == vectors.shape
    with utils.smart_open(fname, 'wb') as fout:
        print(total_vec, vector_size)
        fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        # store in sorted order: most frequent words at the top
        for word, row in vocab.items():
            if binary:
                row = row.astype(REAL)
                fout.write(utils.to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))


def learn_tsne(model, k=None):
    '''fit a 2d t-SNE model'''
    print('learning t-SNE')
    if k == None:
        X = model[model.vocab]
    else:
        X = model[model.vocab][0:k]        
    tsne    = TSNE(n_jobs=8)
    result  = tsne.fit_transform(X)
    return result


def plot_reduc(result, title):
    '''plot the output'''
    print('plotting dim-reduction')
    with PdfPages('../data/plots/' + title + '.pdf') as pdf:
        _, ax = plt.subplots(figsize=(12,12),dpi=100)
        ax.plot(result[:, 0], result[:, 1], 'o')
        ax.set_title('Embedding: 2D, ' + title)
        plt.xlabel('D1')
        plt.ylabel('D2')
        ax.set_yticklabels([]) #Hide ticks
        ax.set_xticklabels([]) #Hide ticks
        pdf.savefig()
        #plt.show()
    
    
def plot_reduc_label(model, result, title, k=None):
    '''add the word to the groups and focus on specific sets'''
    if k==None:
        postfix = '.pdf'
    else:
        postfix = '-top_' + str(k) + '.pdf'
    with PdfPages('../data/plots/' + title + postfix) as pdf:
        print('plotting dim-reduction (2)')
        _, ax = plt.subplots(figsize=(12,12),dpi=100)
        if k == None:
            ax.plot(result[:, 0], result[:, 1], 'o')
        else:
            ax.plot(result[0:k, 0], result[0:k, 1], 'o')
        ax.set_title('Embedding: 2D, ' + title)
        plt.xlabel('D1')
        plt.ylabel('D2')
        ax.set_yticklabels([]) #Hide ticks
        ax.set_xticklabels([]) #Hide ticks
        if k == None:
            words = list(model.vocab)
        else:
            words = list(model.vocab)[0:k]
            print(words)
        for i, word in enumerate(words):
            plt.annotate(word, xy=(result[i, 0], result[i, 1]))
        pdf.savefig()
    print('saving plot (2) to ../data/')  

    

if __name__ == '__main__':
    '''
    main method
    '''
    

#     print('==>\t loading corpus')
#     mcorpus = [s.split() for (s,_) in corpus_reader('/Users/thorne1/BioNLP-frameworks/NER-data/Klinger-data/conll/test_bio.conll')]
#     
#     
#     print('==>\t loading vocabulary') 
#     v_dict   = open('../data/test_vocab_nosymb_inter.txt', 'r', encoding='utf-8')
#     vocab    = [w.strip() for w in v_dict.readlines()]
    
    
#     print('==>\t loading vocabulary') 
#     v_dict   = open('../data/vocab_ibuprofen.txt', 'r', encoding='utf-8')
#     vocab    = [w.strip() for w in v_dict.readlines()]       

 
#     print('==>\t loading ELMo embeddings')
#     elmo_chemu = ElmoEmbedder(
#         options_file='/Users/thorne1/BioNLP-frameworks/CheMU-embeddings/options.json', 
#         weight_file='/Users/thorne1/BioNLP-frameworks/CheMU-embeddings/weights.hdf5'
#     )
#      
#      
#     print('==>\t generating embeddings')
#     chemu_embeddings    = {}
#     t = tqdm(vocab, position=0, ascii=True, desc="progress", dynamic_ncols=True, miniters=1, bar_format = "{desc}: {percentage:.0f}%  [{elapsed}<{remaining}]")
#     for token in t:
#             sents             = getSentences(token, mcorpus)
#             (w, embs_chem)    = getEmbeddings(token, sents, elmo_chemu)
#             (w, emb_chemu)    = combineEmbeddings(w, embs_chem)
#             chemu_embeddings[w] = emb_chemu
#     time.sleep(0.25)
#     t.close()
#              
#      
#     print('==>\t saving embeddings')     
#     chemu_m = dict2w2v(chemu_embeddings, 'chemu_elmo_test.bin') 
    
    
#     elmo_allen = ElmoEmbedder(
#         options_file='/Users/thorne1/BioNLP-frameworks/biomed-embeddings/elmo_2x4096_512_2048cnn_2xhighway_options.json', 
#         weight_file='/Users/thorne1/BioNLP-frameworks/biomed-embeddings/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5'
#     ) 
#     
#     
#     print('==>\t generating embeddings')
#     allen_embeddings    = {}
#     # aggregate vectors
#     t = tqdm(vocab, position=0, ascii=True, desc="progress", dynamic_ncols=True, miniters=1, bar_format = "{desc}: {percentage:.0f}%  [{elapsed}<{remaining}]")
#     for token in t:
#             sents             = getSentences(token, mcorpus)
#             (w, embs_alle)    = getEmbeddings(token, sents, elmo_allen)
#             (w, emb_allen)    = combineEmbeddings(w, embs_alle)
#             allen_embeddings[w] = emb_allen
#     time.sleep(0.25)
#     t.close()
#             
#     
#     print('==>\t saving embeddings')     
#     allen_m = dict2w2v(allen_embeddings, 'pubmed_elmo_test.bin')     


    print('==>\t loading embeddings')  
    chemu_m    = KeyedVectors.load_word2vec_format('../data/chemu_elmo_test.bin', binary=True)
    allen_m    = KeyedVectors.load_word2vec_format('../data/pubmed_elmo_test.bin', binary=True)


    print('==>\t running t-SNE')       
    chemu_t = learn_tsne(chemu_m)   
    allen_t = learn_tsne(allen_m)   
     
 
    print('==>\t visualizing results')   
    plot_reduc(chemu_t, 'ELMo CheMU (test), t-SNE')
    plot_reduc(allen_t, 'ELMo PubMed (test), t-SNE')  
    plot_reduc_label(chemu_m, chemu_t, 'ELMo CheMU (test), t-SNE', 100)
    plot_reduc_label(allen_m, allen_t, 'ELMo PubMed (test), t-SNE', 100)

   
#     print('==>\t generating embeddings')
#     chemu_embeddings    = []
#     allen_embeddings    = []
#     for token in vocab:
#         sents             = getSentences(token, mcorpus)
#         (w, embs_chem)    = getEmbeddings(token, sents, elmo_chemu)
#         (w, embs_alle)    = getEmbeddings(token, sents, elmo_allen)
#         (w, emb_chemu)    = combineEmbeddings(w, embs_chem)
#         (w, emb_allen)    = combineEmbeddings(w, embs_alle)
#         chemu_embeddings.append((w, emb_chemu))
#         allen_embeddings.append((w, emb_allen))
#         
#         
#     print('==>\t building similarity matrixes')
#     sim_chemu = computeSimMatrix(chemu_embeddings)
#     sim_allen = computeSimMatrix(allen_embeddings)
#     
#     
#     print('==>\t running similarity queries') 
#     get_most_similar('ibuprofen', sim_chemu, 'ELMo CheMU', 10)
#     get_most_similar('ibuprofen', sim_allen, 'ELMo PubMed', 10)