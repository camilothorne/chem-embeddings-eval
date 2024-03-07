'''
Created on 14 Feb 2020
@author: camilo thorne
'''

import matplotlib
matplotlib.use('macosx')

import pubchempy as pcp
import pandas as pd
import time, numpy as np
from rdkit import Chem
from rdkit import DataStructs
from tqdm.autonotebook import tqdm
from gensim.models import KeyedVectors
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats


def getVocab(path):
    '''
    load vocabulary
    '''
    v_dict   = open(path, 'r', encoding='utf-8')
    vocab    = [w.strip() for w in v_dict.readlines()]
    v_dict.close()
    return vocab


def loadDrugs(path):
    '''
    load drug pairs
    '''
    df = pd.read_csv(path, encoding='utf-8', sep='\t')
    return df


def getInchis(vocab):
    '''
    get InChIs from DrugBank
    '''
    print('==>\t getting InChIs')
    inchis = {}
    voc    = set(vocab)
    time.sleep(2) 
    tq = tqdm(voc, position=0, ascii=True, desc="progress", dynamic_ncols=True, 
              miniters=1, bar_format = "{desc}: {percentage:.0f}%  [{elapsed}<{remaining}]")  
    for word in tq:
        resu = pcp.get_compounds(word, 'name')
        if len(resu) > 0:
            inchi = resu[0].inchi # pick InChI of the first hit
            inchis[word] = inchi
    time.sleep(2) 
    return inchis
            
            
def saveInChIs(mdict, path):
    '''
    generate dataframe
    '''
    print('==>\t saving InChIs to', path)    
    df          = pd.DataFrame.from_dict(mdict, orient='index', columns=['InChI'])
    df['name']  = df.index
    df.to_csv(path, sep='\t', encoding='utf-8', index=False)
    return df


def computeFingerSim(df):
    '''
    compute molecular similarities
    '''
    inchi1  = list(df['InChI'].values)
    inchi2  = list(df['InChI_2'].values)
    mols1   = [Chem.RDKFingerprint(Chem.MolFromInchi(i)) for i in inchi1]
    mols2   = [Chem.RDKFingerprint(Chem.MolFromInchi(i)) for i in inchi2]
    res     = np.zeros(len(inchi1))
    for i in range(0,len(inchi1)):
        sim      = DataStructs.FingerprintSimilarity(mols1[i], mols2[i], metric=DataStructs.DiceSimilarity)
        res[i]   = sim
    print('==>\t molecular similarity computation\n')
    df['RDKit'] = res
    print(df.head(10))
    return df


def computeEmbSim(df, model, colname):
    '''
    compute model similarities
    '''
    name1  = list(df['name'].values)
    name2  = list(df['name_2'].values)
    res     = np.zeros(len(name1))
    for i in range(0,len(name1)):
        sim      = model.similarity(name1[i],name2[i])
        res[i]   = sim
    print('\n==>\t', colname, 'similarity computation\n')
    df[colname] = res
    print(df.head(10))
    return df


def plot_correlation(corr, title):
    '''plot the output'''
    print('\n==>\t plotting correlation')
    with PdfPages('../data/plots/' + title + '.pdf') as pdf:
        plt.figure(figsize=(9, 9))
        pl = sns.heatmap(corr, annot=True, square=True, cbar=False)
        pl.set_xticklabels(pl.get_xticklabels(), rotation=45)  
        pl.set_title('Embedding correlation (Pearson)')    
        pdf.savefig()
        

def getCorr(df):
    '''get p-values for correlation'''
    df_corr = pd.DataFrame()    # Correlation matrix
    df_p = pd.DataFrame()       # Matrix of p-values
    for x in df.columns:
        for y in df.columns:
            corr = stats.pearsonr(df[x], df[y])
            df_corr.loc[x,y]    = '%.2f' % corr[0]
            df_p.loc[x,y]       = '%.2f' % corr[1]
    return df_corr, df_p


def getTTest(df):
    '''run t-test'''
    df_corr = pd.DataFrame()    # Correlation matrix
    df_p = pd.DataFrame()       # Matrix of p-values
    for x in df.columns:
        for y in df.columns:
            corr = stats.ttest_ind(df[x], df[y])
            df_corr.loc[x,y]    = '%.2f' % corr[0]
            df_p.loc[x,y]       = '%.2f' % corr[1]
    return df_corr, df_p



if __name__ == '__main__':
    
    
    # set Pandas display options
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)    
    
    
#     # get InChIs
#     vocab  = getVocab('../data/test_vocab_nosymb_inter.txt')
#     inchis = getInchis(vocab)
#     dfi    = saveInChIs(inchis, '../data/test_vocab_nosymb_inter_structs.csv')


    # load intersecting embeddings
    chemu    = KeyedVectors.load_word2vec_format('../data/chemu_w2v_test.bin',   binary=True)
    pubme    = KeyedVectors.load_word2vec_format('../data/pubmed_w2v_test.bin',  binary=True)
    ddrug    = KeyedVectors.load_word2vec_format('../data/drug_w2v_test.bin',    binary=True)
    m2vec    = KeyedVectors.load_word2vec_format('../data/mat2vec_w2v_test.bin', binary=True)
    celmo    = KeyedVectors.load_word2vec_format('../data/chemu_elmo_test.bin',  binary=True)
    pelmo    = KeyedVectors.load_word2vec_format('../data/pubmed_elmo_test.bin', binary=True)

    
    # load drug pairs
    df      = loadDrugs('../data/results_embeddings/test_vocab_drug_pairs.tsv') 

    
    # compute similarities
    df      = computeFingerSim(df)
    df      = computeEmbSim(df, chemu, 'CheMU_W2V')
    df      = computeEmbSim(df, pubme, 'PubMed_W2V')
    df      = computeEmbSim(df, ddrug, 'Drug_W2V')
    df      = computeEmbSim(df, m2vec, 'Mat2Vec_W2V')
    df      = computeEmbSim(df, celmo, 'CheMU_ELMo')
    df      = computeEmbSim(df, pelmo, 'PubMed_ELMo')

    
    # save dataframe
    df.to_csv('../data/results_embeddings/test_vocab_drug_pairs_simil.tsv', sep='\t', encoding='utf-8', index=False)
    
    
    # plot correlation
    dfcr = df[['RDKit','CheMU_W2V', 'PubMed_W2V', 'Drug_W2V', 'Mat2Vec_W2V', 'CheMU_ELMo', 'PubMed_ELMo']]
    corr = dfcr.corr(method='pearson')
    plot_correlation(corr, 'embedding_sim_correlation')
    

    # save correlation tables    
    corr_p, p_vals = getCorr(dfcr)    
    with open('../data/results_embeddings/embedding_correlation.txt', 'w', encoding='utf-8') as f:
        cols = pd.Series(['RDKit','CheMU_W2V', 'PubMed_W2V', 'Drug_W2V', 'Mat2Vec_W2V', 'CheMU_ELMo', 'PubMed_ELMo'])
        corr_p.index = cols
        f.write(corr_p.to_latex())
        f.write('\n')
        p_vals.index = cols
        f.write(p_vals.to_latex())
        f.close()
    print()
    print(corr_p)    
    print()    
    print(p_vals)


    # run t-test 
    corr_p, p_vals = getTTest(dfcr)
    with open('../data/results_embeddings/embedding_t_test.txt', 'w', encoding='utf-8') as f:
        cols = pd.Series(['RDKit','CheMU_W2V', 'PubMed_W2V', 'Drug_W2V', 'Mat2Vec_W2V', 'CheMU_ELMo', 'PubMed_ELMo'])
        corr_p.index = cols
        f.write(corr_p.to_latex())
        f.write('\n')
        p_vals.index = cols
        f.write(p_vals.to_latex())
        f.close()
    print()
    print(corr_p)    
    print()    
    print(p_vals)
    
    
#     plt.figure(figsize=(9, 9))
#     pl =sns.heatmap(corr, annot=True, square=True, cbar=False)
#     pl.set_xticklabels(pl.get_xticklabels(),rotation=45)   
#     pl.set_title('Embedding correlation (Pearson, p-value < 0.005)') 
#     plt.show()    