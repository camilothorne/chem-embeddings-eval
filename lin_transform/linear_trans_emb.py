'''
Created on 16 Dec 2019
@author: camilo thorne
'''


# redirect Keras/TF warnings
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
stderr = sys.stderr
sys.stderr = open('/dev/null', 'w')


# Python
from keras.models import load_model
from lin_transform.subsets import load_w2v_model, save_w2v_model, restrict_w2v
from sklearn.decomposition import TruncatedSVD


def disable_warnings():
    '''
    disable warnings
    '''
    import warnings
    warnings.filterwarnings('ignore')  # ignore all Python warnings
    
    
def return_trans_w2v(w2v, model):
    '''
    convert W2V embedding into its linear lin_transform
    '''
    print('==>\t computing linear lin_transform')
    X           = model.predict(w2v.vectors)
    print('\t dims (transformation):', X.shape)
    return X
    
    
def load_keras_model(path, model_name):
    '''
    load keras model (json + h5)
    '''
    print('==>\t loading regressor\t', model_name)
    model = load_model(path + '/' + model_name + '.h5')
    model.summary()
    return model 


def reduce_dims(model, dims):
    '''fit a K-d SVD model (returns reduced space) on embedding'''
    print('==>\t learning SVD, reducing to', str(dims), 'dims')
    svd     = TruncatedSVD(n_components=dims)
    result  = svd.fit_transform(model[model.wv.vocab])
    return result


def compute_diff(mod1, mod2):        
    '''
    compute difference of vocabularies (returns difference)
    ''' 
    root = set(mod1)
    rest = set(mod2)
    diff = rest.difference(root) # mod2 - mod1 (gets rid of intersection!)
    print('==>\t embeddings disagree on', str(len(diff)), 'words' ) 
    return diff


if __name__ == '__main__':
 
    
#     # drug -- path dim reduction
#     dru_W2V  = '/Users/thorne1/BioNLP-frameworks/drug-embeddings/model_dimension_420.bin'           #420d 
#     # drug -- reduce dims
#     w2v_d           = load_w2v_model(dru_W2V, 'w2v-bin')    
#     w2v_X           = reduce_dims(w2v_d, 200)    
#     w2v_d.vectors   = w2v_X
#     save_w2v_model(w2v_d, 'drug_dim_200.bin') # save to ../data/drug_dim_200.bin, 200d
#     # drug -- clean up
#     del w2v_d
#     del w2v_X       
    
    
    # input paths
    mat2vec  = '/Users/thorne1/BioNLP-frameworks/mat2vec/mat2vec/embeddings/pretrained_embeddings'   #200d
    pub_W2V  = '/Users/thorne1/BioNLP-frameworks/biomed-embeddings/PubMed-w2v.bin'                   #200d
    che_W2V  = '/Users/thorne1/BioNLP-frameworks/CheMU-embeddings/patent_w2v.txt'                    #200d
    drux_W2V = '../data/drug_dim_200.bin'                                                             #200d
 
 
    print('---------------------------------------------------------------------------------------')
    
    
    # load embeddings
    w2v_0   = load_w2v_model(che_W2V,  'w2v-txt')
    w2v_1   = load_w2v_model(pub_W2V,  'w2v-bin')
    w2v_2   = load_w2v_model(drux_W2V, 'w2v-bin')  
    w2v_3   = load_w2v_model(mat2vec,  'gensim')      
    
    v_0     = w2v_0.vocab
    v_1     = w2v_1.vocab
    v_2     = w2v_2.vocab
    v_3     = w2v_3.vocab    

    
    print('---------------------------------------------------------------------------------------')    
         
         
    # pubchem
    reg_1           = load_keras_model('../data', 'reg-best-pub-chemu')
    diff            = compute_diff(v_0, v_1)    
    restrict_w2v(w2v_1, diff, ret=False)
    X_1             = return_trans_w2v(w2v_1, reg_1)    
    w2v_1.vectors   = X_1
    save_w2v_model(w2v_1, 'pubmed_trans-res-200.bin')  
    
    
    print('---------------------------------------------------------------------------------------')
    
    
    # drug
    reg_2           = load_keras_model('../data', 'reg-best-dru-chemu')    
    diffx           = compute_diff(diff, v_2)  
    restrict_w2v(w2v_2, diffx, ret=False)      
    X_2             = return_trans_w2v(w2v_2, reg_2)
    w2v_2.vectors   = X_2
    save_w2v_model(w2v_2, 'drug_trans-res-200.bin') 


    print('---------------------------------------------------------------------------------------')
    
    
    # mat2vec
    reg_3           = load_keras_model('../data', 'reg-best-m2v-chemu')    
    diffy           = compute_diff(diffx, v_3)
    restrict_w2v(w2v_3, diffy, ret=False)              
    X_3             = return_trans_w2v(w2v_3, reg_3)
    w2v_3.vectors   = X_3
    save_w2v_model(w2v_3, 'mat2vec_trans-res-200.bin') 