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
        self.aux_verbs = ['be', 'have']

    @staticmethod
    def extract_prepositions(context, shingles):
        for token in context:
            if token[1].startswith("IN") or token[1].startswith("TO"):
                shingles.write(token[0].encode("utf8").strip() + '_PREP ')

    def extract_verbs(self, context, shingles, context_tag):
        for token in context:
            if token[1].startswith("V") and self.lmtzr.lemmatize(token[0], 'v') not in self.aux_verbs:
                #   VB	- Verb, base form
                if token[1] == "VB":
                    shingles.write(token[0].encode("utf8").strip() + '_VB_' + context_tag + ' ')

                # VBD	Verb, past tense
                # VBN	Verb, past participle
                if token[1] == "VBD":
                    shingles.write(token[0].encode("utf8").strip() + '_VBD_' + context_tag + ' ')
                if token[1] == "VBN":
                    shingles.write(token[0].encode("utf8").strip() + '_VBN_' + context_tag + ' ')

                # VBP	Verb, non-3rd person singular present
                # VBZ	Verb, 3rd person singular present
                if token[1] == "VBP":
                    shingles.write(token[0].encode("utf8").strip() + '_VBP_' + context_tag + ' ')
                if token[1] == "VBZ":
                    shingles.write(token[0].encode("utf8").strip() + '_VBZ_' + context_tag + ' ')

                # VBG	Verb, gerund or present participle
                if token[1] == "VBG":
                    shingles.write(token[0].encode("utf8").strip() + '_VBG_' + context_tag + ' ')

    def extract_features(self, after, before, between, e1_type, e2_type):
        shingles = StringIO.StringIO()

        # add entities type
        shingles.write(e1_type.encode("utf8").strip() + '_ENTITY1 ')
        shingles.write(e2_type.encode("utf8").strip() + '_ENTITY2 ')

        # relational patterns corresponding to: a verb, followed by nouns, adjectives, or adverbs,
        # and ending with a preposition;
        reverb_pattern = self.reverb.extract_reverb_patterns_tagged_ptb(between)
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

        # verbs from all contexts, except aux verbs
        self.extract_verbs(before, shingles, 'BEF')
        self.extract_verbs(between, shingles, 'BET')
        self.extract_verbs(after, shingles, 'AFT')

        # prepositions from BET context
        self.extract_prepositions(between, shingles)

        # nouns from the BET context
        for t in between:
            if t[1] == 'NN':
                shingles.write(t[0].encode("utf8").strip()+'_NN_BET' + ' ')

        # n-grams of characters from all contexts
        bef_grams = self.extract_ngrams_chars(' '.join([token[0] for token in before]), "BEF")
        bet_grams = self.extract_ngrams_chars(' '.join([token[0] for token in between]), "BET")
        aft_grams = self.extract_ngrams_chars(' '.join([token[0] for token in after]), "AFT")

        for shingle in bef_grams, bet_grams, aft_grams:
            shingles.write(shingle.encode("utf8").strip() + ' ')

        return shingles

    def process_index(self, sentence, e1, e2):
        sentence_no_tags = re.sub(regex_clean_simple, "", sentence)
        text_tokens = word_tokenize(sentence_no_tags)
        text_tagged = self.tagger.tag(text_tokens)
        assert len(text_tagged) == len(text_tokens)

        # extract entities types
        e1_type = re.search(r'<[A-Z]+>'+e1+'</[A-Z]+>', sentence).group(0)
        e2_type = re.search(r'<[A-Z]+>'+e2+'</[A-Z]+>', sentence).group(0)
        e1_type = e1_type[1:4]
        e2_type = e2_type[1:4]

        e1_info = find_locations(e1, text_tokens)
        e2_info = find_locations(e2, text_tokens)

        for e1_b in e1_info[1]:
            for e2_b in e2_info[1]:
                distance = abs(e2_b - e1_b)
                if distance > MAX_TOKENS or distance < MIN_TOKENS:
                    continue
                else:
                    before = text_tagged[:e1_b]
                    before = before[-CONTEXT_WINDOW:]
                    between = text_tagged[e1_b+len(e1_info[0]):e2_b]
                    after = text_tagged[e2_b+len(e2_info[0]):]
                    after = after[:CONTEXT_WINDOW]

                    return self.extract_features(after, before, between, e1_type, e2_type)

    def process_classify(self, line):
        sentence_no_tags = re.sub(regex_clean_simple, "", line)
        text_tokens = word_tokenize(sentence_no_tags)
        text_tagged = self.tagger.tag(text_tokens)
        assert len(text_tagged) == len(text_tokens)

        sentence = Sentence(line.strip(), MAX_TOKENS,  MIN_TOKENS, CONTEXT_WINDOW, self.tagger)
        relationships = []

        for rel in sentence.relationships:
            shingles = self.extract_features(rel.after, rel.before, rel.between, rel.e1_type, rel.e2_type)
            relationships.append((rel, shingles))

        return relationships

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