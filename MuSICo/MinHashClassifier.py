#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import StringIO
import re
import os
import sys
import codecs
import pickle
import MinHash

from nltk.data import load
from nltk import word_tokenize, ngrams

from ReVerb import Reverb
from Sentence import Sentence, Relationship
from LocalitySensitiveHashing import LocalitySensitiveHashing


N_BANDS = 100
N_SIGS = 800
KNN = 7
CONTEXT_WINDOW = 3
N_GRAMS_SIZE = 4
MAX_TOKENS = 6
MIN_TOKENS = 1


def extract_ngrams(text, context):
    tmp = StringIO.StringIO()
    chrs = ['_' if c == ' ' else c for c in text]
    for g in ngrams(chrs, N_GRAMS_SIZE):
        tmp.write(''.join(g) + '_' + context + ' ')
    return tmp.getvalue()


def classify_sentences(data_file, extractor, lsh):
    """
    receives a file with sentences where entities are identified
    and for each classifies the existing relationships between the entities

    :param data_file:
    :param extractor:
    :param lsh:
    :return:
    """
    f_sentences = codecs.open(data_file, encoding='utf-8')
    for line in f_sentences:
        if len(line) > 1:
            sentence = line.strip()
            sentence = Sentence(sentence)
            for rel in sentence.relationships:
                if rel.arg1type is None and rel.arg2type is None:
                    continue
                else:
                    # extract features/shingles
                    features = extractor.extract_features(rel)
                    shingles = features.getvalue().strip().split(' ')

                    # calculate min-hash sigs
                    sigs = MinHash.signature(shingles, N_SIGS)
                    rel.sigs = sigs

                    # find closest neighbours
                    types = lsh.classify(rel)
                    print rel.sentence.encode("utf8") + '\targ1:'+rel.ent1.encode("utf8") + '\targ2:ent2'+rel.ent2.encode("utf8") + '\t' + types.encode("utf8")

    f_sentences.close()


def tokenize_entity(entity):
    parts = word_tokenize(entity)
    if parts[-1] == '.':
        replace = parts[-2] + parts[-1]
        del parts[-1]
        del parts[-1]
        parts.append(replace)
    return parts


def find_locations(entity_string, text_tokens):
    locations = []
    e_parts = tokenize_entity(entity_string)
    for i in range(len(text_tokens)):
        if text_tokens[i:i + len(e_parts)] == e_parts:
            locations.append(i)
    return e_parts, locations


def load_training_relationships(data_file, tagger, reverb):
    relationships = []
    rel_id = 0
    regex_clean_simple = re.compile('</?[A-Z]+>', re.U)
    f_sentences = codecs.open(data_file, encoding='utf-8')
    f_features = open('features.txt', 'w')

    #TODO: this can all be parallized
    for line in f_sentences:
        if line.startswith("instance"):
            e1, e2, score = line.split("instance : ")[1].split('\t')

        if line.startswith("sentence"):
            sentence = line.split("sentence : ")[1]
            sentence_no_tags = re.sub(regex_clean_simple, "", sentence)
            text_tokens = word_tokenize(sentence_no_tags)
            text_tagged = tagger.tag(text_tokens)
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
            if distance > MAX_TOKENS or distance < MIN_TOKENS:
                continue

            #TODO: garantir isto
            #assert MAX_TOKENS >= distance >= MIN_TOKENS

            before = text_tokens[:e1_b]
            before = before[-CONTEXT_WINDOW:]
            between_pos = text_tagged[e1_b+len(e1_info[0]):e2_b]
            between = text_tokens[e1_b+len(e1_info[0]):e2_b]
            after = text_tokens[e2_b+len(e2_info[0]):]
            after = after[:CONTEXT_WINDOW]
            patterns = reverb.extract_reverb_patterns_tagged_ptb(between_pos)

            #print e1, e2
            #print sentence

            shingles = StringIO.StringIO()
            #TODO: escrever o tipo de relação
            #f_features.write(rel_type + '\t')

            reverb_pattern = '_'.join([t[0] for t in patterns])
            if len(reverb_pattern) > 0:
                reverb_pattern += '_RVB'
                shingles.write(reverb_pattern.encode("utf8") + ' ')

            bef_grams = extract_ngrams(' '.join(before), "BEF")
            bet_grams = extract_ngrams(' '.join(between), "BET")
            aft_grams = extract_ngrams(' '.join(after), "AFT")

            for shingle in bef_grams, bet_grams, aft_grams:
                shingles.write(shingle.encode("utf8") + ' ')

            # calculate min-hash sigs
            sigs = MinHash.signature(shingles.getvalue().split(), N_SIGS)
            """
            rel.sigs = sigs
            rel.identifier = rel_id
            relationships.append(rel)
            """
            rel_id += 1

    f_sentences.close()
    f_features.close()

    return relationships


def load_shingles(shingles_file):
    """
    Parses already extracted shingles from a file.
    File format is: relaltionship_type \t shingle1 shingle2 shingle3 ... shingle_n
    """
    relationships = []
    rel_identifier = 0
    f_shingles = codecs.open(shingles_file, encoding='utf-8')

    for line in f_shingles:
        sys.stdout.write('.')
        rel_type, shingles_string = line.split('\t')
        shingles = shingles_string.strip().split(' ')

        # calculate min-hash sigs
        sigs = MinHash.signature(shingles, N_SIGS)

        rel = Relationship(None, None, None, None, None, None, None, None, rel_type, rel_identifier)
        rel.sigs = sigs
        rel.identifier = rel_identifier
        relationships.append(rel)
        rel_identifier += 1
    
    f_shingles.close()

    return relationships


def main():

    ###########################
    # CLASSIFY A NEW SENTENCE
    ###########################
    # argv[1] - true
    # argv[2] - bands file
    # argv[3] - dict (sigs->sentence_id) file
    # argv[4] - sentence to classify

    if sys.argv[1] == 'true':

        print "Loading PoS tagger"
        model = open('postagger/datasets/cintil-reduced-tagset.pkl', "rb")
        pos_tagger = pickle.load(model)
        model.close()

        print "Loading verbs conjugations"
        f_verbs = open('verbs/verbs_conj.pkl', "rb")
        verbs = pickle.load(f_verbs)
        f_verbs.close()

        extractor = FeatureExtractor(pos_tagger, verbs)
        lsh = LocalitySensitiveHashing(N_BANDS, N_SIGS, KNN)
        
        # read sentences from file
        # create a relationship object
        # extract features/shingles and calculate min-hash sigs
        classify_sentences(sys.argv[2], extractor, lsh)

    ############################
    # INDEX TRAINING INSTANCES #
    ############################
    # argv[1] - false
    # argv[2] - training data file (example: train_data.txt)

    elif sys.argv[1] == 'false':

        # calculate min-hash sigs (from already extracted shingles) index in bands
        if os.path.isfile("features.txt"):
            print "Calculating min-hash sigs from features.txt file"
            relationships = load_shingles('features.txt')
            print "\n"
            print "Indexing ", len(relationships), "relationships"
            print "MinHash Signatures: ", N_SIGS
            print "Bands             : ", N_BANDS
            print "Sigs per Band     : ", N_SIGS / N_BANDS
            lsh = LocalitySensitiveHashing(N_BANDS, N_SIGS, KNN)
            lsh.create()
            for r in relationships:
                lsh.index(r)

        # load sentences, extract features, calculate min-hash sigs, index in bands
        else:
            print "Loading PoS tagger"
            tagger = load('taggers/maxent_treebank_pos_tagger/english.pickle')
            reverb = Reverb()

            print "Extracting features from training data and calculating min-hash sigs"
            relationships = load_training_relationships(sys.argv[2], tagger, reverb)
            print "\n"
            print "Indexing ", len(relationships), "relationships"
            print "MinHash Signatures: ", N_SIGS
            print "Bands             : ", N_BANDS
            lsh = LocalitySensitiveHashing(N_BANDS, N_SIGS, KNN)
            lsh.create()
            for r in relationships:
                lsh.index(r)

if __name__ == "__main__":
    main()
