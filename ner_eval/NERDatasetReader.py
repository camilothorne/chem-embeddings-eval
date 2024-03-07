'''
Created on 17 Sep 2019
@author: camilo thorne
'''


# python3 stuff
from typing import Dict, List, Sequence, Iterable
import itertools
import logging
from overrides import overrides


# allennlp
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
#from allennlp.data.dataset_readers.conll2003 import _is_divider


# logger
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _is_divider(line: str) -> bool:
    '''
    file parser
    '''
    empty_line = line.strip() == ''
    if empty_line:
        return True
    else:
        return False


@DatasetReader.register("cam_conll2003")
class NERDatasetReader(DatasetReader):
    '''
    dataset reader class
    '''
    
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tag_label: str = "ner",
                 feature_labels: Sequence[str] = (),
                 lazy: bool = False,
                 coding_scheme: str = "IOB",
                 label_namespace: str = "labels") -> None:
        '''
        constructor that extends AllenNLP reader class 
        '''
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.tag_label = tag_label
        self.feature_labels = set(feature_labels)
        self.coding_scheme = coding_scheme
        self.label_namespace = label_namespace
        self._original_coding_scheme = "IOB"
    
    
    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        '''
        file reader
        '''
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            # group into alternative divider / sentence chunks.
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                # ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    fields = [line.strip().split() for line in lines]
                    # unzipping trick returns tuples, but our Fields need lists
                    fields = [list(field) for field in zip(*fields)]
                    #print(len(fields))
                    if len(fields) == 2:
                        tokens_, ner_tags = fields
                        #print(len(tokens_), len(ner_tags))
                        # TextField requires ``Token`` objects
                        tokens = [Token(token) for token in tokens_]
                        yield self.text_to_instance(tokens, ner_tags)
    
    
    def text_to_instance(self, # type: ignore
                         tokens: List[Token],
                         ner_tags: List[str] = None) -> Instance:
        '''
        we take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        '''
        # pylint: disable=arguments-differ
        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {'tokens': sequence}
        instance_fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})
        coded_ner = ner_tags
        if 'ner' in self.feature_labels:
            if coded_ner is None:
                raise ConfigurationError("Dataset reader was specified to use NER tags as "
                                         " features. Pass them to text_to_instance.")
            instance_fields['ner_tags'] = SequenceLabelField(coded_ner, sequence, "ner_tags")
        if self.tag_label == 'ner' and coded_ner is not None:
            instance_fields['tags'] = SequenceLabelField(coded_ner, sequence,self.label_namespace)
        return Instance(instance_fields)


#if __name__ == '__main__':
#    pass