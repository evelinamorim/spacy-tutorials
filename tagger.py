import nltk
import pickle
import os
import re
import random
from pathlib import Path

import sys

import spacy
from spacy.vocab import Vocab
from spacy.tokens import Doc
from spacy.gold import GoldParse
from spacy.symbols import ORTH, LEMMA, POS, TAG
from spacy.tokenizer import Tokenizer

nlp = spacy.blank('en')

# customizing my tokenizer
special_case = [{ORTH: u'i', LEMMA: u'i', POS: u'PRN'}, {ORTH: u'have'}]
nlp.tokenizer.add_special_case(u'i\'ve', special_case)

suffix_re = re.compile(r'[!\.?:\"+\'+]*')

prefix_re = re.compile(r'''^[\[\("']''')
infix_re = re.compile(r'''[-~]''')

simple_url_re = r'''^https?://'''
emoticon_re = r'(^|\s)(:(\')*(D|\)|\(|p|P|3)|:(\/)+)(?=\s|[^[0-9A-Za-z]+-]|$)'
emoticon_nose_re = r'(^|\s)(:-(D|\)|\(|p|P|3)|:-\/)(?=\s|[^[0-9A-Za-z]+-]|$)'
unknown_emoticon_re = r'\+\_O'

token_match_re = re.compile("(%s|%s|%s|%s)" % (simple_url_re, emoticon_re, emoticon_nose_re, unknown_emoticon_re))


# initializing the Tokenizer
nlp.tokenizer = Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                     suffix_search=suffix_re.search, 
                                     infix_finditer=infix_re.finditer,
                                     token_match=token_match_re.match)


tagger_data_path = ''



class Train:

    def __init__(self):

        dt1 = self.read_tagger_data(tagger_data_path + 'daily547.conll')
        dt2 = self.read_tagger_data(tagger_data_path + 'oct27.conll')

        self.__tagger_data = dt1 + dt2

        self.tag_map = {'N': {'pos': 'NOUN'}, 'V': {'pos': 'VERB'}, 'S':{'pos':'ADJ','Other':{'PronType':'prs','Poss':'yes'}},\
                      'O':{'pos':'ADJ'}, '^':{'pos':'PROPN'}, 'Z':{'pos':'ADJ'},\
                      'A':{'pos':'ADJ'}, 'R':{'pos':'ADV'}, '!':{'pos':'PUNCT','PunctType':'Excl'},\
                      'D':{'pos':'DET'},'P':{'pos':'ADP'},'&':{'pos':'CCONJ'},\
                      'T':{'pos':'VERB'},'X':{'pos':'ADV','pos':'ADJ'},\
                      '#':{'pos':'SYM','Other':{'SymType':'numbersign'}},'@':{'pos':'PROPN'},\
                      '~':{'pos':'CCONJ'},'E':{'pos':'SYM','Other':{'Style':'Expr'}},'U':{'pos':'X'},\
                      '$':{'pos':'NUM'},',':{'pos':'PUNCT'},'G':{'pos':'X'},\
                      'L':{'pos':'VERB','Other':{'Style':'colloquial','typo':'yes'}},\
                      'M':{'pos':'VERB'},'Y':{'pos':'ADV','AdvType':'ex'}}
        
        self.vocab = Vocab(tag_map=self.tag_map)
 
    def read_tagger_data(self, data_file):
        """
        @data_file: it is a string that represents the full path of the data file 
        that will be used for the training. The training file will be in the Conll 
        format.

        @return list: Returns a list of sentences from an CONLL file
        """

        fd = open(data_file, 'r')
        sent_lst = []
        sent = []

        for line in fd:
            lst_tok = line.split()
            if len(lst_tok) >= 2:
                sent.append(tuple(lst_tok))
            else:
                # it is empty, then starting a new sentence
                sent_lst.append(sent)
                sent = []
        fd.close()

        return sent_lst

    def build_tagger_traindata(self):

        train_data = []
        for s in self.__tagger_data:
            sent = ""
            tag_lst = []

            for word, tag in s:
                sent = sent + word + " "
                tag_lst.append(tag)
            train_data.append((sent.rstrip(),{'tags':tag_lst}))

        return train_data

    def update_taggermodel(self, train_data, niter=50):

        
        tagger = nlp.create_pipe('tagger')

        for tag, values in self.tag_map.items():
            tagger.add_label(tag, values)

        nlp.add_pipe(tagger)
        optimizer = nlp.begin_training()

        for idx in range(niter):
            random.shuffle(train_data)

            losses = {}
            for text, ann in train_data:
                nlp.update([text], [ann], sgd=optimizer, losses=losses)

            print(losses)

    def save_modeltagger(self, output_dir):

        path_output_dir = Path(output_dir)
        nlp.to_disk(path_output_dir)

if __name__ == '__main__':

    data_file_tagger = ''
    data_file = ''
    output_dir = ''


    t = Train()

    # pipeline for the twitter tagger model
    train_data = t.build_tagger_traindata()
    t.update_taggermodel(train_data)
    t.save_modeltagger(output_dir)

