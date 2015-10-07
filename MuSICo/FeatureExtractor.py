#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Sentence import find_locations

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"


CONTEXT_WINDOW = 3
N_GRAMS_SIZE = 4
MAX_TOKENS = 6
MIN_TOKENS = 1
regex_clean_simple = re.compile('</?[A-Z]+>', re.U)

import re
import StringIO

from MuSICo.ReVerb import Reverb
from nltk.data import load
from nltk import word_tokenize, ngrams


class FeatureExtractor:

    def __init__(self):
        print "Loading PoS tagger"
        self.tagger = load('taggers/maxent_treebank_pos_tagger/english.pickle')
        self.reverb = Reverb()

    def extract_index(self, sentence, e1, e2):

        sentence_no_tags = re.sub(regex_clean_simple, "", sentence)
        text_tokens = word_tokenize(sentence_no_tags)
        text_tagged = self.tagger.tag(text_tokens)
        assert len(text_tagged) == len(text_tokens)

        e1_info = find_locations(e1, text_tokens)
        e2_info = find_locations(e2, text_tokens)
        e1_b = e1_info[1][0]
        e2_b = e2_info[1][0]
        if e1_b >= e2_b:
            e1_b = e1_info[1][0]
            e2_b = e2_info[1][1]
        assert e1_b < e2_b

        distance = e2_b - e1_b
        #TODO: garantir isto
        #assert MAX_TOKENS >= distance >= MIN_TOKENS
        if distance > MAX_TOKENS or distance < MIN_TOKENS:
            return None

        before = text_tokens[:e1_b]
        before = before[-CONTEXT_WINDOW:]
        between_pos = text_tagged[e1_b+len(e1_info[0]):e2_b]
        between = text_tokens[e1_b+len(e1_info[0]):e2_b]
        after = text_tokens[e2_b+len(e2_info[0]):]
        after = after[:CONTEXT_WINDOW]
        shingles = StringIO.StringIO()

        reverb_pattern = self.reverb.extract_reverb_patterns_tagged_ptb(between_pos)
        if len(reverb_pattern) > 0:
            reverb_pattern = '_'.join([t[0] for t in reverb_pattern])
            passive_voice = self.reverb.detect_passive_voice(reverb_pattern)
            reverb_pattern = '_'.join([t[0] for t in reverb_pattern])
            if passive_voice is True:
                reverb_pattern += '_RVB_PASSIVE'
            else:
                reverb_pattern += '_RVB'
            shingles.write(reverb_pattern.encode("utf8") + ' ')

            #TODO: passar para aqui parte do código da ReVerb class
            #TODO: negation detection before verb
            #"not", "neither", "nobody", "no", "none", "nor", "nothing", "nowhere", "never"
            #"_" + pattern + "_RVB_" + prefix);

            #TODO: normalized propositions
            """
            #"pp") || auxPOS[i].equals("p-acp") || auxPOS[i].startsWith("pf") ) {
            #  normalized[i] + "_PREP_" + prefix);
            """

        bef_grams = self.extract_ngrams(' '.join(before), "BEF")
        bet_grams = self.extract_ngrams(' '.join(between), "BET")
        aft_grams = self.extract_ngrams(' '.join(after), "AFT")

        for shingle in bef_grams, bet_grams, aft_grams:
            shingles.write(shingle.encode("utf8") + ' ')

        return shingles

    def extract_classify(self, before, between, between_pos, after):
        shingles = StringIO.StringIO()

        # extract ReVerb pattern from BET context
        reverb_pattern = self.reverb.extract_reverb_patterns_tagged_ptb(between_pos)
        if len(reverb_pattern) > 0:
            reverb_pattern = '_'.join([t[0] for t in reverb_pattern])
            passive_voice = self.reverb.detect_passive_voice(reverb_pattern)
            reverb_pattern = '_'.join([t[0] for t in reverb_pattern])
            if passive_voice is True:
                reverb_pattern += '_RVB_PASSIVE'
            else:
                reverb_pattern += '_RVB'
            shingles.write(reverb_pattern.encode("utf8") + ' ')

            """
            #TODO:
            #"not", "neither", "nobody", "no", "none", "nor", "nothing", "nowhere", "never"
            #"_" + pattern + "_RVB_" + prefix);
            """
            #TODO: normalized propositions
            """
            "pp") || auxPOS[i].equals("p-acp") || auxPOS[i].startsWith("pf") ) {
              normalized[i] + "_PREP_" + prefix);
            """

        # # extract n-grams pattern from BET context
        bef_grams = self.extract_ngrams(' '.join(before), "BEF")
        bet_grams = self.extract_ngrams(between, "BET")
        aft_grams = self.extract_ngrams(' '.join(after), "AFT")

        for shingle in bef_grams, bet_grams, aft_grams:
            shingles.write(shingle.encode("utf8") + ' ')

        return shingles

    def extract_ngrams(self, text, context):
        tmp = StringIO.StringIO()
        chrs = ['_' if c == ' ' else c for c in text]
        for g in ngrams(chrs, self.n_grams_size):
            tmp.write(''.join(g) + '_' + context + ' ')
        return tmp.getvalue()