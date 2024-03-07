'''
Created on 3 Sep 2019
@author: camilo thorne
'''


# redirect Keras/TF warnings
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
stderr = sys.stderr
sys.stderr = open('/dev/null', 'w')


import numpy
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from gensim.models import KeyedVectors, Word2Vec
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf


# redirect warnings to null 
sys.stderr = stderr


def disable_warnings():
    '''
    disable warnings
    '''
    import warnings
    warnings.filterwarnings('ignore')  # ignore all Python warnings  


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


def custom_loss(Y_true, Y_pred):
    '''
    custom loss function, Y_pred and Y_true are tensors
    based on Mikolov's paper (sum of Euclidean norms):
    
        L(W) = \Sum_i ||W*x_i - y_i||
    
    '''
    return K.sum(tf.norm(Y_pred - Y_true, axis=1),axis=-1)


def baseline_model():
    '''
    create baseline model
    '''
    #print('\t using MSE linear baseline model')
    model   = Sequential()
    model.add(Dense(200, input_dim=200, kernel_initializer='normal', activation='linear'))
    model.add(Dropout(0.5))
    # compile model
    opt     = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])
    with open('../data/baseline_model.json', 'w') as f:
        f.write(model.to_json())
    model.summary()
    return model


def custom_layer_model():
    '''
    create custom layer model
    '''
    #print('\t using custom loss linear baseline model')    
    inputs  =  Input(shape=(200,))
    layer   =  Dense(200, kernel_initializer='normal', activation='linear')(inputs)
    res     =  Dropout(0.5)(layer)
    # compile model
    opt     = Adam(lr=1e-3, decay=1e-3 / 200)
    model   = Model(inputs=inputs, outputs=res)
    model.compile(loss=custom_loss, optimizer=opt, metrics=[custom_loss])
    with open('../data/custom_layer_model.json', 'w') as f:
        f.write(model.to_json())
    model.summary()
    return model


def baseline_relu_model():
    '''
    create relu baseline model
    '''
    #print('\t using MSE ReLu model')
    model   = Sequential()
    model.add(Dense(200, input_dim=200, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    # compile model
    opt     = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])
    with open('../data/baseline_relu_model.json', 'w') as f:
        f.write(model.to_json())
    model.summary()
    return model


def large_model():
    '''
    create improved model
    '''
    #print('\t using MSE linear deep model')
    model = Sequential()
    model.add(Dense(200, input_dim=200, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, kernel_initializer='normal', activation='linear'))
    model.add(Dropout(0.5))
    # compile model
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])
    with open('../data/improved_model.json', 'w') as f:
        f.write(model.to_json())
    model.summary()
    return model


def callback(name):
    '''
    set callback functions to early stop training and save the best model so far
    '''
    callbacks    = [EarlyStopping(monitor='mean_squared_error', mode='min', verbose=1), 
                    ModelCheckpoint(filepath='../data/reg-best-'+name+'.h5', 
                                    monitor='mean_squared_error', mode='min',
                                    save_best_only=True)]
    return callbacks


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
    print('==>\t compile regressor, save best')    
    
    # fix random seed for reproducibility    
    seed            = 7
    numpy.random.seed(seed)
    estimator       = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=32, verbose=0)      
    kfold           = KFold(n_splits=5, random_state=seed)
    
    print('---------------------------------------------------------------------------------------')  
    
    
    print('train and evaluate mat2vec --> CheMU')    
    results         = cross_val_score(estimator, mod_m2v[mod_m2v.vocab], mod_che[mod_che.vocab], cv=kfold, 
                                      fit_params={'callbacks':callback('m2v-chemu')})
    print('---------------------------------------------------------------------------------------')  
    print("mat2vec --> CheMU results: %.2f MSE (%.2f STD)" % (results.mean(), results.std()))       
    print('---------------------------------------------------------------------------------------')  


    print('train and evaluate PubMed --> CheMU')    
    results         = cross_val_score(estimator, mod_pub[mod_pub.vocab], mod_che[mod_che.vocab], cv=kfold, 
                                      fit_params={'callbacks':callback('pub-chemu')})
    print('---------------------------------------------------------------------------------------') 
    print("PubMed --> CheMU results: %.2f MSE (%.2f STD)" % (results.mean(), results.std()))  
    print('---------------------------------------------------------------------------------------')  
    
    
    print('train and evaluate Drug --> CheMU')    
    results         = cross_val_score(estimator, mod_dru[mod_dru.vocab], mod_che[mod_che.vocab], cv=kfold,
                                      fit_params={'callbacks':callback('dru-chemu')})
    print('---------------------------------------------------------------------------------------') 
    print("Drug --> CheMU results: %.2f MSE (%.2f STD)" % (results.mean(), results.std()))
    print('---------------------------------------------------------------------------------------')  