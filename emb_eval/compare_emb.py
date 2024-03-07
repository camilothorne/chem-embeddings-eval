'''
Created on 20 Jan 2020
@author: camilo thorne
'''


import pandas as pd
import numpy as np
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import accuracy
from nltk.metrics import jaccard_distance
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
#from PIL import Image


def readDF(dfin):
    '''
    loads pandas dataframe
    '''
    df = pd.read_csv(dfin, sep='\t', encoding='utf-8', error_bad_lines=False)
    return df


def getFeatures(df):
    '''
    gets features of dataframe
    '''
    return list(df)


def printMatrix(features, mat):
    '''
    print matrix
    '''
    features = [f.split(' ')[1] + ' ' + f.split(' ')[0] for f in features]
    head     = '\t'.join(features)
    body     = ''
    for i in range(0,len(features)):
        body = body + features[i]
        for j in range(0, len(features)):
            body = body + '\t' + str(mat[i][j])
        body = body + '\n'
    print('\t' + head + '\n' +  body)
    print(' & ' + 
            head.replace('\t',' & ') + '\\\ \n' +  
            body.replace('\t',' & ').replace('nan','---').replace('\n','\\\ \n'))


def genAgJud(df, colu):
    '''
    generate agreement judgements
    '''
    data        = df[colu]
    judgements  = []
    for index, value in data.items():
        judgements.append((colu, index, value))
    return judgements


def genAccJud(df, colu):
    '''
    generate accuracy judgements
    '''
    data        = df[colu]
    judgements  = []
    for _, value in data.items():
        judgements.append(value)
    return judgements    


def computeAgrMatrix(df, features):
    '''
    compute agreement matrix
    '''
    mat     = np.zeros(shape=(len(features),len(features)))
    mat[:]  = np.NaN
    for i in range(0,len(features)):
        mat[i][i] = "%.2f" % 1.00
        for j in range(i+1, len(features)):         
            ju1 = genAgJud(df, features[i])
            ju2 = genAgJud(df, features[j])
            task = AnnotationTask(ju1 + ju2)
            mat[i][j] = "%.2f" % task.avg_Ao()
    print('==>\t agreement matrix\n')
    printMatrix(features, mat)
    return (features, mat)


def computeAccMatrix(df, features):
    '''
    compute accuracy matrix
    '''
    mat     = np.zeros(shape=(len(features),len(features)))
    mat[:]  = np.NaN
    for i in range(0,len(features)):
        for j in range(i, len(features)):
            ju1 = genAccJud(df, features[i])
            ju2 = genAccJud(df, features[j])
            mat[i][j] = "%.2f" % accuracy(ju1, ju2)
    print('==>\t accuracy matrix\n')
    printMatrix(features, mat)
    return (features, mat)


def computeJaccMatrix(df, features):
    '''
    compute similarity matrix
    '''
    mat     = np.zeros(shape=(len(features),len(features)))
    mat[:]  = np.NaN
    for i in range(0,len(features)):
        for j in range(i, len(features)):
            ju1 = genAccJud(df, features[j])
            ju2 = genAccJud(df, features[i])            
            mat[i][j] = "%.2f" % (1 - jaccard_distance(set(ju1), set(ju2)))
    print('==>\t overlap matrix\n')
    printMatrix(features, mat)
    return (features, mat)


def computeSimMatrix(df):
    '''
    compute molecular similarity matrix
    '''
    words   = list(df['name'].values)
    inchis  = list(df['InChI'].values)
    mols    = [Chem.MolFromInchi(i) for i in inchis]
    fps     = [Chem.RDKFingerprint(m) for m in mols]
    mat     = np.zeros(shape=(len(df),len(df)))
    for i in range(0,len(df)):
        for j in range(i, len(df)):
            cos = DataStructs.FingerprintSimilarity(fps[i], fps[j], metric=DataStructs.DiceSimilarity)
            mat[i][j] = cos
    print('==>\t molecular matrix computation\n')
    return (words, mat)


def generateMolImages(df, path):
    '''
    generate molecule visualizations
    '''
    print('\n==>\t saving molecular visualization to', path)
    words   = list(df['name'].values)
    inchis  = list(df['InChI'].values)
    mols    = [Chem.MolFromInchi(i) for i in inchis]
    fps     = [Chem.RDKFingerprint(m) for m in mols]
    mat     = np.zeros(shape=(len(df),len(df)))
    for i in range(0,len(df)):
        for j in range(0, len(df)):
            cos = DataStructs.FingerprintSimilarity(fps[i], fps[j], metric=DataStructs.DiceSimilarity)
            mat[i][j] = cos
    mdict = {}
    wdict = {}
    for i in range(0, len(df)):
        mdict[mols[i]]  = mat[0,i]
        wdict[words[i]] = mat[0,i] 
    mol   = sorted(mdict, key=mdict.__getitem__, reverse=True)
    wdict = {k: v for k, v in sorted(wdict.items(), key=lambda item: item[1], reverse=True)}       
    for m in mol: tmp=AllChem.Compute2DCoords(m)
    img=Draw.MolsToGridImage(mol, molsPerRow=4, subImgSize=(200,200), legends=[x + ':%.2f' % wdict[x] for x in wdict.keys()], useSVG=True)
    with open(path, 'w') as f:
        f.write(img)
        f.close()


def get_most_similar(word, simmatrix, title):
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
    for t in list(sim.keys()): 
        print(t,':\t',sim[t])


if __name__ == '__main__':   
    
    
    df          = readDF('../data/results_embeddings/similarity_hydroxymethyl.txt')
    features    = getFeatures(df)
    computeJaccMatrix(df, features)
    computeAccMatrix(df,  features)
    computeAgrMatrix(df,  features)


    df          = readDF('../data/results_embeddings/similarity_ibuprofen.txt')
    features    = getFeatures(df)
    computeJaccMatrix(df, features)
    computeAccMatrix(df,  features)
    computeAgrMatrix(df,  features)

    
    df_ibu      = readDF('../data/results_embeddings/norm_ibuprofen.txt')    
    simmx       = computeSimMatrix(df_ibu)
    get_most_similar('ibuprofen', simmx, 'molecular similarity')
    generateMolImages(df_ibu, '../data/plots/ibuprofen_molgrid_sorted.svg')

    
    print()

    
    df_h      = readDF('../data/results_embeddings/norm_hydroxymethyl.txt')    
    simmh       = computeSimMatrix(df_h)
    get_most_similar('hydroxymethyl', simmh, 'molecular similarity')
    generateMolImages(df_h, '../data/plots/hydroxymethyl_molgrid_sorted.svg')    