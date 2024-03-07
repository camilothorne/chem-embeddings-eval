'''
Created on 29 Aug 2019
@author: camilo thorne
'''


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lars
from gensim.models import KeyedVectors, Word2Vec


def disable_warnings():
    '''
    disable warnings
    '''
    import warnings
    warnings.filterwarnings('ignore')  # ignore all warnings


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


def learn_regressor(Model, s_emb, t_emb, title):
    '''
    learn generic regressor over embeddings with default
    hyper-parameters using K processors (returns model, prints R^2)
    '''
    model = Model()
    model.fit(s_emb[s_emb.vocab], t_emb[t_emb.vocab])
    print('R^2:',  title, '%.3f' % model.score(s_emb[s_emb.vocab], t_emb[t_emb.vocab]))
    print('coeffs:', model.coef_.shape)
    print('---------------------------------------------------------------------------------------')    
    return model


if __name__ == '__main__':
    
    # disable warnings
    disable_warnings()
    
    # paths
    mat2vec = '../data/mat2vec_inter.bin'   #200d, Mat2VeC
    pub_W2V = '../data/pubmed_inter.bin'    #200d, PubMed
    che_W2V = '../data/chemu_inter.bin'     #200d, CheMU
    dru_W2V = '../data/drug_inter.bin'      #200d, Drug    
    
    # load models    
    mod_che   = load_w2v_model(che_W2V, 'w2v-bin')
    mod_m2v   = load_w2v_model(mat2vec, 'w2v-bin')
    mod_pub   = load_w2v_model(pub_W2V, 'w2v-bin')
    mod_dru   = load_w2v_model(dru_W2V, 'w2v-bin')    
    print('---------------------------------------------------------------------------------------')     
    
    # CheMU --> Mat2Vec
    print('==>\t learning\t CheMU --> Mat2Vec mapping')
    print('---------------------------------------------------------------------------------------') 
    learn_regressor(LinearRegression, mod_che, mod_m2v,   '\t(CheMU --> Mat2Vec, LinearReg)  \t')
    learn_regressor(ElasticNet, mod_che, mod_m2v,         '\t(CheMU --> Mat2Vec, ElasticReg) \t')
    learn_regressor(Lasso, mod_che, mod_m2v,              '\t(CheMU --> Mat2Vec, LassoReg)   \t')
    learn_regressor(Ridge, mod_che, mod_m2v,              '\t(CheMU --> Mat2Vec, RidgeReg)   \t')
    learn_regressor(Lars, mod_che, mod_m2v,               '\t(CheMU --> Mat2Vec, LarsReg)    \t')
 
    # CheMU --> PubMed   
    print('==>\t learning\t CheMU --> PubMed mapping')
    print('---------------------------------------------------------------------------------------') 
    learn_regressor(LinearRegression, mod_che, mod_pub,   '\t(CheMU --> PubMed, LinearReg)   \t')
    learn_regressor(ElasticNet, mod_che, mod_pub,         '\t(CheMU --> PubMed, ElasticReg)  \t')
    learn_regressor(Lasso, mod_che, mod_pub,              '\t(CheMU --> PubMed, LassoReg)    \t')
    learn_regressor(Ridge, mod_che, mod_pub,              '\t(CheMU --> PubMed, RidgeReg)    \t')
    learn_regressor(Lars, mod_che, mod_pub,               '\t(CheMU --> PubMed, LarsReg)     \t')
    
    # CheMU --> Drug    
    print('==>\t learning\t CheMU --> Drug mapping')
    print('---------------------------------------------------------------------------------------') 
    learn_regressor(LinearRegression, mod_che, mod_dru,   '\t(CheMU --> Drug, LinearReg)     \t')
    learn_regressor(ElasticNet, mod_che, mod_dru,         '\t(CheMU --> Drug, ElasticReg)    \t')
    learn_regressor(Lasso, mod_che, mod_dru,              '\t(CheMU --> Drug, LassoReg)      \t')
    learn_regressor(Ridge, mod_che, mod_dru,              '\t(CheMU --> Drug, RidgeReg)      \t')
    learn_regressor(Lars, mod_che, mod_dru,               '\t(CheMU --> Drug, LarsReg)       \t')
    
    # PubMed --> Mat2Vec
    print('==>\t learning\t PubMed --> Mat2Vec mapping')
    print('---------------------------------------------------------------------------------------') 
    learn_regressor(LinearRegression, mod_pub, mod_m2v,   '\t(PubMed --> Mat2Vec, LinearReg) \t')
    learn_regressor(ElasticNet, mod_pub, mod_m2v,         '\t(PubMed --> Mat2Vec, ElasticReg)\t')
    learn_regressor(Lasso, mod_pub, mod_m2v,              '\t(PubMed --> Mat2Vec, LassoReg)  \t')
    learn_regressor(Ridge, mod_pub, mod_m2v,              '\t(PubMed --> Mat2Vec, RidgeReg)  \t')
    learn_regressor(Lars, mod_pub, mod_m2v,               '\t(PubMed --> Mat2Vec, LarsReg)   \t')
 
    # PubMed --> CheMU   
    print('==>\t learning\t PubMed --> CheMU mapping')
    print('---------------------------------------------------------------------------------------') 
    learn_regressor(LinearRegression, mod_pub, mod_che,   '\t(PubMed --> CheMU, LinearReg)   \t')
    learn_regressor(ElasticNet, mod_pub, mod_che,         '\t(PubMed --> CheMU, ElasticReg)  \t')
    learn_regressor(Lasso, mod_pub, mod_che,              '\t(PubMed --> CheMU, LassoReg)    \t')
    learn_regressor(Ridge, mod_pub, mod_che,              '\t(PubMed --> CheMU, RidgeReg)    \t')
    learn_regressor(Lars, mod_pub, mod_che,               '\t(PubMed --> CheMU, LarsReg)     \t')
    
    # PubMed --> Drug    
    print('==>\t learning\t PubMed --> Drug mapping')
    print('---------------------------------------------------------------------------------------') 
    learn_regressor(LinearRegression, mod_pub, mod_dru,   '\t(PubMed --> Drug, LinearReg)    \t')
    learn_regressor(ElasticNet, mod_pub, mod_dru,         '\t(PubMed --> Drug, ElasticReg)   \t')
    learn_regressor(Lasso, mod_pub, mod_dru,              '\t(PubMed --> Drug, LassoReg)     \t')
    learn_regressor(Ridge, mod_pub, mod_dru,              '\t(PubMed --> Drug, RidgeReg)     \t')
    learn_regressor(Lars, mod_pub, mod_dru,               '\t(PubMed --> Drug, LarsReg)      \t')    