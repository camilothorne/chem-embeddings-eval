'''
Created on 18 Sep 2019
@author: camilo thorne
'''


from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from ner_eval.NERDatasetReader import NERDatasetReader
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter


def disable_warnings():
    '''
    disable warnings
    '''
    import warnings
    warnings.filterwarnings('ignore')  # ignore all Python warnings  


def corpus_reader(corpus_path):
    '''
    load corpus for prediction
    '''
    
    corpus      = open(corpus_path, 'r', encoding='utf-8')
    sents       = corpus.read().split('\n\n')
    formatted   = []
    for sent in sents[0:-1]: # skip last line, get both tokens and tags
        toks  = sent.split('\n')
        ws    = [w.split('\t')[0] + ' ' for w in toks]
        ts    = [w.split('\t')[1] + ' ' for w in toks]
        wres  = ''.join(ws).strip()
        tres  = ''.join(ts).strip()
        formatted.append((wres, tres))
    corpus.close()
    return formatted


def predict_tags(sent, predictor, output=None, cnt=None):
    '''
    write predictions, generate CoNLL 2000 output
    '''
    
    if cnt is not None:
        print('processing sentence #', cnt) 
    print(sent[0], '\n') # print only the sentence
    if output == None:
        results = predictor.predict(sentence=sent)
        for word, tag in zip(results["words"], results["tags"]):
            print(f"{word}\t{tag}")
        print()  
    else:
        tokens, tag = sent  # recover both tokens and tags
        tags        = tag.split(' ')
        results     = predictor.predict(sentence=tokens)
        # 'results' is a dictionary object containing:
        #     - logits: raw score for each label, per word (see model vocabulary)
        #     - mask:   value mask
        #     - words:  words
        #     - tags:   predicted labels
        assert len(tokens.split(' ')) == len(results["words"]) # check that input was correctly tokenized
        for i in range(0, len(results["words"])):
            output.write(results["words"][i] + '\t' + tags[i] + '\t' + results["tags"][i]+'\n')
        output.write('\n')    


if __name__ == '__main__':
    '''
    main method
    '''
    
    # load model + corpus
    #sents                   = corpus_reader('/Users/thorne1/BioNLP-frameworks/NER-data/Klinger-data/conll/test_bio.conll') # test
    #sents                   = corpus_reader('/Users/thorne1/BioNLP-frameworks/NER-data/Klinger-data/conll/dev_bio_biosem.conll') # dev
    sents                   = corpus_reader('/Users/thorne1/BioNLP-frameworks/NER-data/relevancy_gs_free/conll/relevancy_test_IUPAC.conll') # reaxys
    
    # register custom dataset reader
    reader                  = NERDatasetReader()
    
    # load + configure predictor
    #predictor               = Predictor.from_archive(load_archive("../data/allennlp_runs/crf-model", cuda_device=-1), None) 
    #predictor               = Predictor.from_archive(load_archive("../data/allennlp_runs/crf-model_2", cuda_device=-1), None) 
    #predictor               = Predictor.from_archive(load_archive("../data/allennlp_runs/crf-model_21", cuda_device=-1), None)     
    #predictor               = Predictor.from_archive(load_archive("../data/allennlp_runs/allen_0", cuda_device=-1), None) 
    #predictor               = Predictor.from_archive(load_archive("../data/allennlp_runs/allen_1", cuda_device=-1), None) 
    #predictor               = Predictor.from_archive(load_archive("../data/allennlp_runs/allen_2", cuda_device=-1), None)
    #predictor               = Predictor.from_archive(load_archive("../data/allennlp_runs/allen_3", cuda_device=-1), None) 
    #predictor               = Predictor.from_archive(load_archive("../data/allennlp_runs/allen_4", cuda_device=-1), None) 
    #predictor               = Predictor.from_archive(load_archive("../data/allennlp_runs/allen_1_elmo", cuda_device=-1), None) 
    predictor               = Predictor.from_archive(load_archive("../data/allennlp_runs/allen_elmo", cuda_device=-1), None) 
    #predictor               = Predictor.from_archive(load_archive("../data/allennlp_runs/allen_char", cuda_device=-1), None) 
    #predictor               = Predictor.from_archive(load_archive("../data/allennlp_runs/allen_w2v", cuda_device=-1), None) 
    #predictor               = Predictor.from_archive(load_archive("../data/allennlp_runs/allen_bioelmo", cuda_device=-1), None) 
    #predictor               = Predictor.from_archive(load_archive("../data/allennlp_runs/allen_elmo_only", cuda_device=-1), None) 
    #predictor               = Predictor.from_archive(load_archive("../data/allennlp_runs/allen_w2v_trans", cuda_device=-1), None) 
    
    # the corpus is whitespace tokenized, hence tokenize by whitespace
    predictor._tokenizer    = JustSpacesWordSplitter()
    
    # test model on examples
    sentence0   ='''The ( propargyloxy ) methyl acyclonucleoside analogues of 6 - chloropurine , adenine , 
    6 - methoxypurine , hypoxanthine , 6 - mercaptopurine , and azathioprine have been prepared .'''   
    sentence1   ='''While there are several secondary effects of the loss of Rb on retinal development including , 
    limited cell death in the ONL , M .'''  
    predict_tags(sentence0, predictor)
    predict_tags(sentence1, predictor)
    
    # run model on sentences and print results in CoNLL 2000 format
    #output = open('/Users/thorne1/BioNLP-frameworks/NER-data/Klinger-data/conll/dev_bio_biosem_preds.conll', 'w', encoding='utf-8')
    
    # test set, runs from 18.11.2019
    #output = open('/Users/thorne1/BioNLP-frameworks/NER-data/Klinger-data/conll/test_bio_preds.conll', 'w', encoding='utf-8')
    #output = open('/Users/thorne1/BioNLP-frameworks/NER-data/Klinger-data/conll/test_bio_preds_2.conll', 'w', encoding='utf-8')
    #output = open('/Users/thorne1/BioNLP-frameworks/NER-data/Klinger-data/conll/test_bio_preds_21.conll', 'w', encoding='utf-8')
    
    # test set, runs from 24.11.2019
    #output = open('/Users/thorne1/BioNLP-frameworks/NER-data/Klinger-data/conll/test_bio_preds_allen_0.conll', 'w', encoding='utf-8')
    #output = open('/Users/thorne1/BioNLP-frameworks/NER-data/Klinger-data/conll/test_bio_preds_allen_1.conll', 'w', encoding='utf-8')
    #output = open('/Users/thorne1/BioNLP-frameworks/NER-data/Klinger-data/conll/test_bio_preds_allen_3.conll', 'w', encoding='utf-8')
    #output = open('/Users/thorne1/BioNLP-frameworks/NER-data/Klinger-data/conll/test_bio_preds_allen_2.conll', 'w', encoding='utf-8')
    #output = open('/Users/thorne1/BioNLP-frameworks/NER-data/Klinger-data/conll/test_bio_preds_allen_4.conll', 'w', encoding='utf-8')
    #output = open('/Users/thorne1/BioNLP-frameworks/NER-data/Klinger-data/conll/test_bio_preds_allen_1_elmo.conll', 'w', encoding='utf-8')
    #output = open('/Users/thorne1/BioNLP-frameworks/NER-data/Klinger-data/conll/test_bio_preds_allen_elmo.conll', 'w', encoding='utf-8')    
    #output = open('/Users/thorne1/BioNLP-frameworks/NER-data/Klinger-data/conll/test_bio_preds_allen_char.conll', 'w', encoding='utf-8')  
    #output = open('/Users/thorne1/BioNLP-frameworks/NER-data/Klinger-data/conll/test_bio_preds_allen_w2v.conll', 'w', encoding='utf-8')  
    #output = open('/Users/thorne1/BioNLP-frameworks/NER-data/Klinger-data/conll/test_bio_preds_allen_bioelmo.conll', 'w', encoding='utf-8')  
    #output = open('/Users/thorne1/BioNLP-frameworks/NER-data/Klinger-data/conll/test_bio_preds_allen_elmo_only.conll', 'w', encoding='utf-8')  
    #output = open('/Users/thorne1/BioNLP-frameworks/NER-data/Klinger-data/conll/test_bio_preds_allen_w2v_trans.conll', 'w', encoding='utf-8')  

    # reaxys data, 27.2.2020 
    output = open('/Users/thorne1/BioNLP-frameworks/NER-data/relevancy_gs_free/conll/relevancy_test_IUPAC_preds_allen_elmo.conll', 'w', encoding='utf-8')

    cnt    = 1
    for sent in sents:
        predict_tags(sent, predictor, output, cnt)
        cnt = cnt  + 1 
    output.close()
    
    