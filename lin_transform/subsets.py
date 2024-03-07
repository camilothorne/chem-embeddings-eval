'''
Created on 20 Aug 2019
@author: camilo thorne
'''


from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
from numpy import corrcoef 
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances
from scipy.stats import ttest_rel as t_test
from scipy.stats import pearsonr as correl
from scipy.stats import beta, spearmanr
from lin_transform import pyvenn as venn
import matplotlib.pyplot as plt 


def load_w2v_model(path, typ='gensim'):
    '''
    load W2V model (returns keyed vectors)
    '''
    print('==>\t loading embedding\t', path)
    if typ=='w2v-bin':
        model   = KeyedVectors.load_word2vec_format(path, binary=True)
    if typ=='w2v-txt':
        model   = KeyedVectors.load_word2vec_format(path, binary=False)
    if typ=='gensim':
        g_model = Word2Vec.load(path)
        model   = g_model.wv
    return model


def save_w2v_model(model, filename):
    '''
    save W2V model
    '''
    print('==>\t saving (bin) to ../data/\t', filename)
    model.save_word2vec_format('../data/' + filename, binary=True)


def save_w2v_model_txt(model, filename):
    '''
    save W2V model
    '''
    print('==>\t saving (txt) to ../data/\t', filename)
    model.save_word2vec_format('../data/' + filename, encoding='utf-8', binary=False)
    

def learn_svd(model, dims):
    '''fit a K-d SVD model (returns reduced space)'''
    print('==>\t learning SVD, reducing to', str(dims), 'dims')
    svd     = TruncatedSVD(n_components=dims)
    result  = svd.fit_transform(model)
    return result


def compute_inter(models):        
    '''
    compute intersection of vocabularies (returns intersection)
    '''    
    inter = set(models[0])
    for i in range(1, len(models)):
        inter = inter.intersection(set(models[i]))
    print('models cover', str(len(inter)), 'common words' )
    print('---------------------------------------------------------------------------------------')     
    return inter


def compute_distance(mod1, mod2, inter, k, title, svd=False):
    '''
    compute Euclidian distance of models over (returns a vector of k distances)
    '''
    if svd:
        dist = pairwise_distances(mod1[list(inter)[0:k]], mod2[0:k])
    else:
        dist = pairwise_distances(mod1[list(inter)[0:k]], mod2[list(inter)[0:k]])
    print('Euclidian distance (avg.): ', title, ' %.2f' % (sum(sum(dist[0:][0:]))/(k*k)))
    print('---------------------------------------------------------------------------------------')
    return dist


def measure_overlap(voc1, voc2, title):        
    '''
    measure overlap of vocabularies (returns percentage)
    '''
    v1      = set(voc1)
    v2      = set(voc2)
    inter   = len(set.intersection(v1,v2))
    union   = len(set.union(v2,v2))
    over    = (inter/union)*100
    print(title, '\tembeddings cover %.0f' % len(set(voc1)), 'and %.0f' % len(set(voc2)), 'words resp.')
    print(title, '\tembeddings share %.0f' % inter, 'tokens')
    print(title, '\tembeddings overlap by %.2f' % over + ' %')
    print(title, '\tembeddings share %.0f' % inter, 'tokens')
    print('---------------------------------------------------------------------------------------')
    return over


def measure_sig(res1, res2, title):        
    '''
    measure significance of distances (prints t-statistic and p-value)
    '''  
    t, p    = t_test( sum(res1[0:][0:]), sum(res2[0:][0:]) )
    print('t-test: ', title, ' p-value: %.2f' % p, 't: %.2f' % t)
    print('---------------------------------------------------------------------------------------')


def matrix_cor(matrix):
    '''
    measure correlation over matrix 
    (returns arrays of r statistics and p values)
    '''
    r = np.corrcoef(matrix)
    rf = r[np.triu_indices(r.shape[0], 1)]
    df = matrix.shape[1] - 2
    ts = rf * rf * (df / (1 - rf * rf))
    pf = beta(0.5 * df, 0.5, df / (df + ts))
    p = np.zeros(shape=r.shape)
    p[np.triu_indices(p.shape[0], 1)] = pf
    p[np.tril_indices(p.shape[0], -1)] = pf
    p[np.diag_indices(p.shape[0])] = np.ones(p.shape[0])
    return r, p


def measure_mcor(v1, v2, title):        
    '''
    measure correlation of measures (prints corr. coef. and p-value)
    '''
    c, p = correl( sum(v1[0:][0:]), sum(v2[0:][0:]) )
    print('Pearson: ', title, ' corr.: %.2f' % c, 'p-value: %.2f' % p)
    print('---------------------------------------------------------------------------------------')


def measure_cor(mod1, mod2, inter, k, title, svd=False):
    '''
    measure correlation of embeddings (prints corr. coefs;
    correlation is here normalized co-variance)
    '''
    if svd:
        dist = corrcoef(mod1[list(inter)[0:k]], mod2[0:k])
    else:
        dist = corrcoef(mod1[list(inter)[0:k]], mod2[list(inter)[0:k]])
    print('norm. co-variance (avg.): ', title, ' %.2f' % (sum(sum(dist[0:][0:]))/(k*k)))
    print('---------------------------------------------------------------------------------------')
    return dist


def measure_spear(mod1, mod2, inter, k, title, svd=False):
    '''
    measure avg. Spearman correlation of embeddings 
    (prints avg. Spearman corr. coefs and p-values)
    '''
    if svd:
        rh, p = spearmanr(mod1[list(inter)[0:k]], mod2[0:k])
    else:
        rh, p = spearmanr(mod1[list(inter)[0:k]], mod2[list(inter)[0:k]])
    print('Spearman rho (avg.): ', title, ' %.2f' % (sum(sum(rh[0:][0:]))/(k*k)))
    print('p-value (avg.)     : ', title, ' %.2f' % (sum(sum( p[0:][0:]))/(k*k)))
    print('---------------------------------------------------------------------------------------')
    return rh, p



def restrict_w2v(w2v, restricted_word_set, ret=True):
    '''
    restrict embedding to a given vocabulary (mutates embedding)
    '''
    print('==>\t restricting embedding to word set')
    new_vectors      = []
    new_vocab        = {}
    new_index2entity = []
    for i in range(len(w2v.vocab)):
        word  = w2v.index2entity[i]
        vec   = w2v.vectors[i]
        vocab = w2v.vocab[word]
        if word in restricted_word_set:
            vocab.index     = len(new_index2entity)
            new_index2entity.append(word)
            new_vocab[word] = vocab
            new_vectors.append(vec)
    w2v.vocab        = new_vocab
    w2v.vectors      = np.array(new_vectors)
    w2v.index2entity = np.array(new_index2entity)
    w2v.index2word   = np.array(new_index2entity)
    print('\t dims (restriction):', w2v.vectors.shape)
    if ret: 
        return w2v


def plot_venn(my_list, title, names=['CheMU W2V', 'PubMed W2V', 'Mat2Vec W2V', 'Drug W2V', 'Test corpus']):
    '''
    plot overlaps
    '''
    print('plotting overlaps')
    labels = venn.get_labels([set(m) for m in my_list], fill=['number'])
    if len(names) == 5:
        fig, _ = venn.venn5(labels, names)
    if len(names) == 4:
        fig, _ = venn.venn4(labels, names)        
    fig.savefig('../data/plots/'+title+'.pdf', bbox_inches='tight')
    plt.close()
    print('---------------------------------------------------------------------------------------')



if __name__ == '__main__':


    # paths
    mat2vec = '/Users/thorne1/BioNLP-frameworks/mat2vec/mat2vec/embeddings/pretrained_embeddings'   #200d
    pub_W2V = '/Users/thorne1/BioNLP-frameworks/biomed-embeddings/PubMed-w2v.bin'                   #200d
    che_W2V = '/Users/thorne1/BioNLP-frameworks/CheMU-embeddings/patent_w2v.txt'                    #200d
    dru_W2V = '/Users/thorne1/BioNLP-frameworks/drug-embeddings/model_dimension_420.bin'            #420d


    # load models    
    mod_m2v   = load_w2v_model(mat2vec)
    mod_pub   = load_w2v_model(pub_W2V, 'w2v-bin')  
    mod_che   = load_w2v_model(che_W2V, 'w2v-txt')
    mod_dru   = load_w2v_model(dru_W2V, 'w2v-bin')


    # compute intersection
    print('---------------------------------------------------------------------------------------')
    inter     = compute_inter([mod_pub.vocab, mod_m2v.vocab, mod_che.vocab, mod_dru.vocab])
 
    
    # measure overlap
    measure_overlap(mod_m2v.vocab, mod_pub.vocab, 'mat2vec -- PubMed')
    measure_overlap(mod_dru.vocab, mod_pub.vocab, 'Drug -- PubMed')   
    measure_overlap(mod_che.vocab, mod_pub.vocab, 'CheMU -- PubMed')
    measure_overlap(mod_che.vocab, mod_m2v.vocab, 'CheMU -- mat2vec')
    measure_overlap(mod_che.vocab, mod_dru.vocab, 'CheMU -- Drug')
    measure_overlap(mod_dru.vocab, mod_m2v.vocab, 'Drug -- mat2vec')


    # reduce drug w2v to 200d
    X       = learn_svd(mod_dru[inter], 200)
    print('---------------------------------------------------------------------------------------')
  
    
    # measure average distance (over first 500 common words)
    d_m2p   = compute_distance(mod_pub, mod_m2v, inter, 500, 'mat2vec - PubMed ')
    d_d2p   = compute_distance(mod_pub, X, inter, 500, 'Drug    - PubMed ', svd=True)
    d_c2p   = compute_distance(mod_pub, mod_che, inter, 500, 'CheMU   - PubMed ')      
    d_m2c   = compute_distance(mod_che, mod_m2v, inter, 500, 'CheMU   - mat2vec')     
    d_c2d   = compute_distance(mod_che, X, inter, 500, 'CheMU   - Drug   ', svd=True)     
    d_m2d   = compute_distance(mod_m2v, X, inter, 500, 'Drug    - mat2vec', svd=True) 


    # statistical significance
    measure_sig(d_m2p, d_d2p, 'mat2vec - PubMed vs. Drug    - PubMed')
    measure_sig(d_d2p, d_c2p, 'Drug    - PubMed vs. CheMU   - PubMed')
    measure_sig(d_c2p, d_m2c, 'CheMU   - PubMed vs. mat2vec -  CheMU')      
    measure_sig(d_m2c, d_c2d, 'mat2vec - CheMU  vs. CheMU   -   Drug')     
    measure_sig(d_c2d, d_m2d, 'CheMU   - Drug   vs. mat2vec -   Drug')     


    # normalized covariance (over first 500 common words)
    measure_cor(mod_pub, mod_m2v, inter, 500, 'mat2vec - PubMed ')
    measure_cor(mod_pub, X, inter, 500, 'Drug    - PubMed ', svd=True)
    measure_cor(mod_pub, mod_che, inter, 500, 'CheMU   - PubMed ')      
    measure_cor(mod_che, mod_m2v, inter, 500, 'CheMU   - mat2vec')     
    measure_cor(mod_che, X, inter, 500, 'CheMU   - Drug   ', svd=True)     
    measure_cor(mod_m2v, X, inter, 500, 'Drug    - mat2vec', svd=True) 

        
#     # restrict W2V embeddings to intersection
#     mod_m2v_r   = restrict_w2v(mod_m2v, inter)
#     mod_pub_r   = restrict_w2v(mod_pub, inter)
#     mod_che_r   = restrict_w2v(mod_che, inter) 
#
#     
#     # restrict and reduce drug w2v
#     mod_dru_r               = restrict_w2v(mod_dru, inter)
#     mod_dru_r.vectors       = X
#
#      
#     # serialize restricted embeddings
#     save_w2v_model(mod_m2v_r, 'mat2vec_inter.bin')
#     save_w2v_model(mod_pub_r, 'pubmed_inter.bin')
#     save_w2v_model(mod_che_r, 'chemu_inter.bin' )
#     save_w2v_model(mod_dru_r, 'drug_inter.bin' )  
# 
#  
#     print('loading test corpus vocabulary (inter)') 
#     v_dict   = open('../data/test_vocab_nosymb_inter.txt', 'r', encoding='utf-8')
#     vocab    = [w.strip() for w in v_dict.readlines()]
#     print('---------------------------------------------------------------------------------------')
#
#     
#     # restrict embeddings to intersection
#     mod_m2v_r   = restrict_w2v(mod_m2v, vocab)
#     mod_pub_r   = restrict_w2v(mod_pub, vocab)
#     mod_che_r   = restrict_w2v(mod_che, vocab)
#     mod_dru_r   = restrict_w2v(mod_dru, vocab)
#
#       
#     # serialize restricted embeddings
#     save_w2v_model(mod_m2v_r, 'mat2vec_w2v_test.bin')
#     save_w2v_model(mod_pub_r, 'pubmed_w2v_test.bin')
#     save_w2v_model(mod_che_r, 'chemu_w2v_test.bin' )
#     save_w2v_model(mod_dru_r, 'drug_w2v_test.bin' )  
#     print('---------------------------------------------------------------------------------------')
#
#
#     # restrict to Fraunhofer test set vocabulary
#     new_inter = set(vocab).intersection(set(inter.vocab))
#     # serialize intersection
#     save_vocabulary(new_inter, 'test_vocab_nosymb_inter.txt') 


    print('loading test corpus vocabulary (all)') 
    vt_dict   = open('../data/test_vocab_nosymb.txt', 'r', encoding='utf-8')
    vocab_t   = [w.strip() for w in vt_dict.readlines()]
    print('---------------------------------------------------------------------------------------')


    # plot overlaps
    my_list   = [list(mod_che.vocab.keys()), list(mod_pub.vocab.keys()), 
               list(mod_m2v.vocab.keys()), list(mod_dru.vocab.keys()), vocab_t]
    my_list_2 = [list(mod_che.vocab.keys()), list(mod_pub.vocab.keys()), 
               list(mod_m2v.vocab.keys()), list(mod_dru.vocab.keys())]
    plot_venn(my_list,   'w2v_test_overlaps')
    plot_venn(my_list_2, 'w2v_overlaps',   names=['CheMU W2V', 'PubMed W2V', 'Mat2Vec W2V', 'Drug W2V'])