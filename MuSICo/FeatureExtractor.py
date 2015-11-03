#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import re
import StringIO

from ReVerb import Reverb
from Sentence import find_locations, Sentence

from nltk import word_tokenize, ngrams, bigrams
from nltk.data import load
from nltk.stem.wordnet import WordNetLemmatizer

CONTEXT_WINDOW = 3
N_GRAMS_SIZE = 4
MAX_TOKENS = 6
MIN_TOKENS = 1
regex_clean_simple = re.compile('</?[A-Z]+>', re.U)


class FeatureExtractor:

    def __init__(self):
        self.tagger = load('taggers/maxent_treebank_pos_tagger/english.pickle')
        self.reverb = Reverb()
        self.lmtzr = WordNetLemmatizer()

    def extract_features(self, after, before, between, between_pos):
        shingles = StringIO.StringIO()

        # relational patterns corresponding to: a verb, followed by nouns, adjectives, or adverbs,
        # and ending with a preposition;
        reverb_pattern = self.reverb.extract_reverb_patterns_tagged_ptb(between_pos)
        if len(reverb_pattern) > 0:
            passive_voice = self.reverb.detect_passive_voice(reverb_pattern)
            pattern = '_'.join([t[0] for t in reverb_pattern])
            if passive_voice is True:
                pattern += '_RVB_PASSIVE'
            else:
                pattern += '_RVB'

            shingles.write(pattern.encode("utf8").strip() + ' ')

            # normalized version of reverb_patterns
            pattern_normalized = ''
            for t in reverb_pattern:
                if t[1].startswith("V"):
                    pattern_normalized += self.lmtzr.lemmatize(t[0], 'v') + '_'
                else:
                    pattern_normalized += self.lmtzr.lemmatize(t[0]) + '_'

            pattern_normalized += 'RVB_NORM'
            shingles.write(pattern_normalized.encode("utf8").strip() + ' ')

        # print between_pos

        # all verbs (except aux) in between

        # verb forms in the past participle tense;
        """
        VB	Verb, base form
        28.	VBD	Verb, past tense
        29.	VBG	Verb, gerund or present participle
        30.	VBN	Verb, past participle
        31.	VBP	Verb, non-3rd person singular present
        32.	VBZ	Verb, 3rd person singular present
        """
        # prepositions;

        # infinitive forms of verbs, except auxiliary verbs;

        #nouns from the BET context
        for t in between_pos:
            if t[1] == 'NN':
                shingles.write(t[0].encode("utf8").strip()+'_NN_BET' + ' ')

        # n-grams of characters
        bef_grams = self.extract_ngrams_chars(' '.join(before), "BEF")
        bet_grams = self.extract_ngrams_chars(' '.join(between), "BET")
        aft_grams = self.extract_ngrams_chars(' '.join(after), "AFT")

        for shingle in bef_grams, bet_grams, aft_grams:
            shingles.write(shingle.encode("utf8").strip() + ' ')

        return shingles

    def process_index(self, sentence, e1, e2):
        sentence_no_tags = re.sub(regex_clean_simple, "", sentence)
        text_tokens = word_tokenize(sentence_no_tags)
        text_tagged = self.tagger.tag(text_tokens)
        assert len(text_tagged) == len(text_tokens)

        e1_info = find_locations(e1, text_tokens)
        e2_info = find_locations(e2, text_tokens)

        for e1_b in e1_info[1]:
            for e2_b in e2_info[1]:
                distance = e2_b - e1_b
                if distance > MAX_TOKENS or distance < MIN_TOKENS:
                    continue
                else:
                    before = text_tokens[:e1_b]
                    before = before[-CONTEXT_WINDOW:]
                    between_pos = text_tagged[e1_b+len(e1_info[0]):e2_b]
                    between = text_tokens[e1_b+len(e1_info[0]):e2_b]
                    after = text_tokens[e2_b+len(e2_info[0]):]
                    after = after[:CONTEXT_WINDOW]

                    return self.extract_features(after, before, between, between_pos)

    def process_classify(self, line):
        sentence = Sentence(line.strip(), MAX_TOKENS,  MIN_TOKENS, CONTEXT_WINDOW, self.tagger)
        for rel in sentence.relationships:
            bet_text = [t[0] for t in rel.between]
            shingles = self.extract_features(rel.after, rel.before, bet_text, rel.between)
            return rel, shingles

    @staticmethod
    def extract_ngrams_chars(text, context):
        tmp = StringIO.StringIO()
        chrs = ['_' if c == ' ' else c for c in text]
        for g in ngrams(chrs, N_GRAMS_SIZE):
            tmp.write(''.join(g) + '_' + context + ' ')
        return tmp.getvalue()

    @staticmethod
    def extract_bigrams(text):
        tokens = word_tokenize(text)
        return [gram[0]+' '+gram[1] for gram in bigrams(tokens)]