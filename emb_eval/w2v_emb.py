'''
Created on 14 Jan 2020
@author: camilo thorne
'''


from gensim.models import KeyedVectors
import pandas as pd
from emb_eval.intrinsic_eval import getVocab, getInchis, saveInChIs


def get_most_sim(emb, title, query, df, resu=None):
    '''
    print top 10 most similar words to query in word embedding
    '''
    my_list = emb.most_similar(query)
    print('--------------------')
    print(query, '--', title)
    print('--------------------')
    for tok in my_list:
        print(tok[0],':\t', tok[1])
        if resu is not None:
            resu.add(tok[0])
    df[title] = [t for t,_ in my_list]
        
        
def save_vocabulary(union, name):
    '''
    serialize union
    '''
    mdict = open('../data/results_embeddings/'+name, 'w', encoding='utf-8')
    print('==>\t serializing', name)  
    for tok in union:
        mdict.write(str(tok)+'\n')
    mdict.close()


def save_results(df, name):
    '''
    serialize union
    '''
    print('==>\t serializing', name)  
    df.to_csv('../data/results_embeddings/'+name, encoding='utf-8', index=False, sep='\t')
    with open('../data/results_embeddings/latex_'+name, 'w', encoding='utf-8') as f:
        f.write(df.to_latex(index=False))
        f.close()


if __name__ == '__main__':
    '''
    main method
    '''
    
    # load intersecting embeddings
    chemu    = KeyedVectors.load_word2vec_format('../data/chemu_w2v_test.bin',   binary=True)
    pubme    = KeyedVectors.load_word2vec_format('../data/pubmed_w2v_test.bin',  binary=True)
    ddrug    = KeyedVectors.load_word2vec_format('../data/drug_w2v_test.bin',    binary=True)
    m2vec    = KeyedVectors.load_word2vec_format('../data/mat2vec_w2v_test.bin', binary=True)
    celmo    = KeyedVectors.load_word2vec_format('../data/chemu_elmo_test.bin',  binary=True)
    pelmo    = KeyedVectors.load_word2vec_format('../data/pubmed_elmo_test.bin', binary=True)
    
    
    # get most similar to ibuprofen (FP)
    resu     = set([])
    df       = pd.DataFrame()
    get_most_sim(chemu, 'CheMU W2V', 'ibuprofen',   df, resu)
    get_most_sim(pubme, 'PubMed W2V', 'ibuprofen',  df, resu)
    get_most_sim(ddrug, 'Drug W2V', 'ibuprofen',    df, resu)
    get_most_sim(m2vec, 'Mat2Vec W2V', 'ibuprofen', df, resu)
    get_most_sim(celmo, 'CheMU ELMo', 'ibuprofen',  df, resu)
    get_most_sim(pelmo, 'PubMed ELMo', 'ibuprofen', df, resu)
    resu = ['ibuprofen'] + list(resu)
    print('--------------------')
    
    
    # serialize all shared terms + results
    save_vocabulary(resu,  'vocab_ibuprofen.txt')
    save_results(df,  'similarity_ibuprofen.txt')    


    # get InChIs
    vocabi  = getVocab('../data/results_embeddings/vocab_ibuprofen.txt')
    inchisi = getInchis(vocabi)
    saveInChIs(inchisi, '../data/results_embeddings/norm_ibuprofen.txt')


    # get most similar to hydroxymethyl (TP)
    resu     = set([])
    df       = pd.DataFrame()
    get_most_sim(chemu, 'CheMU W2V', 'hydroxymethyl',   df, resu)
    get_most_sim(pubme, 'PubMed W2V', 'hydroxymethyl',  df, resu)
    get_most_sim(ddrug, 'Drug W2V', 'hydroxymethyl',    df, resu)
    get_most_sim(m2vec, 'Mat2Vec W2V', 'hydroxymethyl', df, resu)
    get_most_sim(celmo, 'CheMU ELMo', 'hydroxymethyl',  df, resu)
    get_most_sim(pelmo, 'PubMed ELMo', 'hydroxymethyl', df, resu)
    resu = ['hydroxymethyl'] + list(resu)
    print('--------------------')
    
    
    # serialize all shared terms + results
    save_vocabulary(resu,  'vocab_hydroxymethyl.txt')
    save_results(df,  'similarity_hydroxymethyl.txt')   
    
    
    # get InChIs
    vocabh  = getVocab('../data/results_embeddings/vocab_hydroxymethyl.txt')
    inchish = getInchis(vocabh)
    saveInChIs(inchish, '../data/results_embeddings/norm_hydromethyl.txt')
     

 
    print('--------------------')
    print('size of shared vocabulary:', len(chemu.vocab), 'tokens')    
    