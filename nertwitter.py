# -*- coding: utf-8 -*-
# author: Evelin Amorim
# 23/03/2018

import re
import random
from pathlib import Path


import spacy
from spacy.vocab import Vocab
from spacy.tokens import Doc
from spacy.gold import GoldParse
from spacy.symbols import ORTH, LEMMA, POS, TAG
from spacy.tokenizer import Tokenizer

nlp = spacy.blank('en')

# customizing my tokenizer, since tweets is tokenized in a differente way

# I've will me mapped to I have
special_case = [{ORTH: u'i', LEMMA: u'i', POS: u'PRN'}, {ORTH: u'have'}]
nlp.tokenizer.add_special_case(u'i\'ve', special_case)

# punctuations that repeate itself, like: !!! or ???, which is very common 
# on tweets
suffix_re = re.compile(r'[!\.?:\"+\'+]*')

# quotes and characters that usually are prefix of words, then 
# we tokenize them
prefix_re = re.compile(r'''^[\[\("']''')
# tokenize words that contains - between them
infix_re = re.compile(r'''[-~]''')

# tokenize urls
simple_url_re = r'''^https?://'''
# tokenize emoticons without 'nose', for instance :) and :(
emoticon_re = r'(^|\s)(:(\')*(D|\)|\(|p|P|3)|:(\/)+)(?=\s|[^[0-9A-Za-z]+-]|$)'
# tokenize emoticons with 'nose', for instance :-) and :-(
emoticon_nose_re = r'(^|\s)(:-(D|\)|\(|p|P|3)|:-\/)(?=\s|[^[0-9A-Za-z]+-]|$)'
# in the dataset this symbol came up, then I included as an unknown symbol
unknown_emoticon_re = r'\+\_O'

# combine the above tokens (urls, emoticons and unknown symbols) in the token match expression
# to find the exact tokens and tokenize them
token_match_re = re.compile("(%s|%s|%s|%s)" % (simple_url_re, emoticon_re, emoticon_nose_re, unknown_emoticon_re))


# initializing the Tokenizer
nlp.tokenizer = Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                     suffix_search=suffix_re.search, 
                                     infix_finditer=infix_re.finditer,
                                     token_match=token_match_re.match)



class Train:
    """
    This class reads a conll dataset, train a ner model using spaCy library, 
    and save the model
   
    """

 
    def read_ner_data(self, data_file):
        """
        @data_file string: it is a file name, that contains ner annotation in conll 
        format

        @returns: [(string,dictionary of entities in the string)] 
        """ 

        data_lst = []

        fd = open(data_file + '/train', 'r')

        offset = 0
        txt = ""
        tag_lst = []
        for line in fd:

            dt = line.replace('\n','').split()

            if len(dt) > 0:
                offset = offset + len(txt)
                txt = txt + " " + dt[0]
                tag = dt[1]
                if tag != 'O':
                    tag_lst.append((offset, offset+len(dt[0]), tag))
            else:
                data_lst.append((txt, {'entities':tag_lst}))
                tag_lst = []
                offset = 0
                txt = ""

        fd.close()

        return data_lst

    def update_nermodel(self, train_data, n_iter=30):
        """
        Since spaCy model is trained in general datasets, this 
        method aims to augument the model with new data.

        
        @train_data [(string,dictionary of entities in the string)]: list of data 
         to train a new model of ner in the spaCy library
        @n_iter: Number of epochs that the update will perform
        """


        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)

        # add labels
        for _, annotations in train_data:
            for ent in annotations.get('entities'):
                ner.add_label(ent[2])

        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

        with nlp.disable_pipes(*other_pipes):  # only train NER
            optimizer = nlp.begin_training()
            for itn in range(n_iter):
                random.shuffle(train_data)
                losses = {}
                for text, annotations in train_data:
                    nlp.update(
                        [text],  # batch of texts
                        [annotations],  # batch of annotations
                        drop=0.5,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
                        losses=losses)
                print(losses)

        nlp.to_disk(Path('.'))

if __name__ == '__main__':

    data_file_tagger = ''
    data_file = ''
    output_dir = ''


    t = Train()


    ner_data = t.read_ner_data(data_file)
    t.update_nermodel(ner_data)
