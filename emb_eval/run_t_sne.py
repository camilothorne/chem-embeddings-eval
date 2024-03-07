'''
Created on 15 Aug 2019
@author: camilo thorne
'''

from __future__ import print_function


import matplotlib
matplotlib.use('macosx')
#import time
from sklearn.decomposition import PCA, TruncatedSVD
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


def load_w2v_model(path, typ='gensim'):
    '''
    load W2V model, return keyed vectors
    '''
    print('loading embedding', path)
    if typ=='w2v-bin':
        model   = KeyedVectors.load_word2vec_format(path, binary=True)
    if typ=='w2v-txt':
        model   = KeyedVectors.load_word2vec_format(path, binary=False)
    if typ=='gensim':
        g_model = Word2Vec.load(path)
        model   = g_model.wv
    return model


def learn_pca(model, k=None):
    '''fit a 2d PCA model'''
    print('learning PCA')
    if k == None:
        X = model[model.vocab]
    else:
        X = model[model.vocab][0:k]
    pca     = PCA(n_components=2)
    result  = pca.fit_transform(X)
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    return result


def learn_svd(model, k=None):
    '''fit a 2d SVD model'''
    print('learning SVD')
    if k == None:
        X = model[model.vocab]
    else:
        X = model[model.vocab][0:k]
    svd     = TruncatedSVD(n_components=2)
    result  = svd.fit_transform(X)
    return result


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


def learn_pca_v(model, vocab, k=None):
    '''fit a 2d PCA model'''
    print('learning PCA')
    if k == None:
        X = model[vocab]
    else:
        X = model[vocab][0:k]
    pca     = PCA(n_components=2)
    result  = pca.fit_transform(X)
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    return result


def learn_svd_v(model, vocab, k=None):
    '''fit a 2d SVD model'''
    print('learning SVD')
    if k == None:
        X = model[vocab]
    else:
        X = model[vocab][0:k]
    svd     = TruncatedSVD(n_components=2)
    result  = svd.fit_transform(X)
    return result


def learn_tsne_v(model, vocab, k=None):
    '''fit a 2d t-SNE model'''
    print('learning t-SNE')
    if k == None:
        X = model[vocab]
    else:
        X = model[vocab][0:k]        
    tsne    = TSNE(n_jobs=8)
    result  = tsne.fit_transform(X)
    return result


def plot_reduc(result, title):
    '''plot the output'''
    print('plotting dim-reduction')
    with PdfPages('../data/plots/' + title + '.pdf') as pdf:
        _, ax = plt.subplots(figsize=(12,12),dpi=100)
        ax.plot(result[:, 0], result[:, 1], 'o')
        ax.set_title('SkipGram Embedding: 2D, ' + title)
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
        ax.set_title('SkipGram Embedding: 2D, ' + title)
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
    

def plot_reduc_label_v(vocab, result, title, k=None):
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
        ax.set_title('SkipGram Embedding: 2D, ' + title)
        plt.xlabel('D1')
        plt.ylabel('D2')
        ax.set_yticklabels([]) #Hide ticks
        ax.set_xticklabels([]) #Hide ticks
        if k == None:
            words = list(vocab)
        else:
            words = list(vocab)[0:k]
            print(words)
        for i, word in enumerate(words):
            plt.annotate(word, 
                         xy=(result[i, 0], result[i, 1]),
                         horizontalalignment='center', 
                         verticalalignment='top',
                         fontsize=8
                         )
        pdf.savefig()
    print('saving plot (2) to ../data/plots/')      


if __name__ == '__main__':
    
    
#     # paths
#     mat2vec = '/Users/thorne1/BioNLP-frameworks/mat2vec/mat2vec/embeddings/pretrained_embeddings'   #200d
#     pub_W2V = '/Users/thorne1/BioNLP-frameworks/biomed-embeddings/PubMed-w2v.bin'                   #200d
#     che_W2V = '/Users/thorne1/BioNLP-frameworks/CheMU-embeddings/patent_w2v.txt'                    #200d
#     dru_W2V = '/Users/thorne1/BioNLP-frameworks/drug-embeddings/model_dimension_420.bin'            #420d
#
#    
#     # load models
#     model   = load_w2v_model(mat2vec)
#     model   = load_w2v_model(pub_W2V, 'w2v-bin')
#     model   = load_w2v_model(che_W2V, 'w2v-txt')
#     model   = load_w2v_model(dru_W2V, 'w2v-bin')
#
#     
#     # t-SNE
#     result  = learn_tsne(model, 10000)
#     plot_reduc_label(model, result, 'mat2vec - t-SNE', 100)
#     plot_reduc_label(model, result, 'PubMed, t-SNE', 100)
#     plot_reduc_label(model, result, 'CheMU - t-SNE', 100)
#     plot_reduc_label(model, result, 'drug - t-SNE', 100)
#     time.sleep(5)
#
#     
#     # PCA
#     result  = learn_pca(model)
#     plot_reduc_label(model, result, 'mat2vec - PCA', 100)
#     plot_reduc_label(model, result, 'PubMed, PCA', 100)
#     plot_reduc_label(model, result, 'CheMU - PCA', 100)
#     plot_reduc_label(model, result, 'drug - PCA', 100)
#     time.sleep(5)
#
# 
#     # SVD
#     result  = learn_svd(model)    
#     plot_reduc_label(model, result, 'mat2vec - SVD', 100)
#     plot_reduc_label(model, result, 'PubMed, SVD', 100)
#     plot_reduc_label(model, result, 'CheMU - SVD', 100)
#     plot_reduc_label(model, result, 'drug - SVD', 100) 


    # paths
    mat2vec = '../data/mat2vec_inter.bin'   #200d
    pub_W2V = '../data/pubmed_inter.bin'    #200d
    che_W2V = '../data/chemu_inter.bin'     #200d
    dru_W2V = '../data/drug_inter.bin'      #420d
 
    
    # load models
    model_m   = load_w2v_model(mat2vec, 'w2v-bin')
    model_p   = load_w2v_model(pub_W2V, 'w2v-bin')
    model_c   = load_w2v_model(che_W2V, 'w2v-bin')
    model_d   = load_w2v_model(dru_W2V, 'w2v-bin')   
    
    # load test set vocabulary intersection
    v_dict   = open('../data/test_vocab_nosymb_inter.txt', 'r', encoding='utf-8')
    vocab    = [w.strip() for w in v_dict.readlines()]
    print('test set size:', len(vocab))
    
    # learn t-SNE
    result_m  = learn_tsne_v(model_m, vocab)
    result_p  = learn_tsne_v(model_p, vocab)
    result_c  = learn_tsne_v(model_c, vocab)
    result_d  = learn_tsne_v(model_d, vocab)
    
    # plot first K points
    plot_reduc_label_v(vocab, result_m, 'Mat2vec (test), t-SNE', 100)
    plot_reduc_label_v(vocab, result_p, 'PubMed (test), t-SNE', 100)
    plot_reduc_label_v(vocab, result_c, 'CheMU (test), t-SNE', 100)
    plot_reduc_label_v(vocab, result_d, 'Drug (test), t-SNE', 100)
    
    # plot all points
    plot_reduc(result_m, 'Mat2vec (test), t-SNE')
    plot_reduc(result_p, 'PubMed (test), t-SNE')
    plot_reduc(result_c, 'CheMU (test), t-SNE')
    plot_reduc(result_d, 'Drug (test), t-SNE')     