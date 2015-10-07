#!/usr/bin/env python
# -*- coding: utf-8 -*-
from MuSICo.FeatureExtractor import FeatureExtractor

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import StringIO
import re
import os
import sys
import codecs
import MinHash

from nltk.data import load
from nltk import word_tokenize, ngrams

from ReVerb import Reverb
from Sentence import Sentence, find_locations
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


def classify_sentences(data_file, lsh):
    #TODO: isto também pode ser paralelizado
    print "Loading PoS tagger"
    tagger = load('taggers/maxent_treebank_pos_tagger/english.pickle')
    reverb = Reverb()
    fe = FeatureExtractor(tagger, reverb, N_GRAMS_SIZE)
    count = 0
    f_sentences = codecs.open(data_file, encoding='utf-8')
    for line in f_sentences:
        if line.startswith("#"):
            continue
        else:
            # generate relationship
            sentence = Sentence(line.strip(), MAX_TOKENS,  MIN_TOKENS, CONTEXT_WINDOW, tagger)
            for rel in sentence.relationships:
                shingles = StringIO.StringIO()
                """
                print rel.sentence
                print rel.e1, '\t', rel.e2
                print rel.before
                print rel.between
                print rel.after
                """
                reverb_pattern = reverb.extract_reverb_patterns_tagged_ptb(rel.between)
                bet_text = ' '.join([t[0] for t in reverb_pattern])

                # extract features
                if len(reverb_pattern) > 0:
                    passive_voice = reverb.detect_passive_voice(reverb_pattern)
                    #print reverb_pattern
                    #print passive_voice
                    #TODO: usar passive voice como feature nos shingles
                    reverb_pattern = '_'.join([t[0] for t in reverb_pattern])
                    reverb_pattern += '_RVB'
                    shingles.write(reverb_pattern.encode("utf8") + ' ')

                bef_grams = extract_ngrams(' '.join(rel.before), "BEF")
                bet_grams = extract_ngrams(bet_text, "BET")
                aft_grams = extract_ngrams(' '.join(rel.after), "AFT")

                for shingle in bef_grams, bet_grams, aft_grams:
                    shingles.write(shingle.encode("utf8") + ' ')

                # calculate min-hash sigs
                sigs = MinHash.signature(shingles.getvalue().split(), N_SIGS)

                # find closest neighbours
                types = lsh.classify(sigs)
                if types is not None:
                    print rel.e1, '\t', rel.e2
                    print rel.sentence.encode("utf8")
                    print types.encode("utf8")
                    print

        count += 1
        if count % 100 == 0:
            sys.stdout.write(".")

    f_sentences.close()


def load_training_relationships(data_file, rel_type, tagger, reverb, rel_id):
    relationships = []
    f_sentences = codecs.open(data_file, encoding='utf-8')
    f_features = open('features.txt', 'w')

    fe = FeatureExtractor()
    print "Loading PoS tagger"
    tagger = load('taggers/maxent_treebank_pos_tagger/english.pickle')
    reverb = Reverb()

    for line in f_sentences:
        if line.startswith("instance"):
            try:
                e1, e2, score = line.split("instance : ")[1].split('\t')
            except Exception, e:
                print e
                print line
                sys.exit(-1)

        if line.startswith("sentence") and e1 is not None and e2 is not None:
            sentence = line.split("sentence : ")[1]

            # extract features
            shingles = fe.extract_index(sentence, e1, e2)

            # calculate min-hash sigs
            sigs = MinHash.signature(shingles.getvalue().split(), N_SIGS)
            relationships.append((rel_type, rel_id, sigs))
            if rel_id % 100 == 0:
                sys.stdout.write(".")
            rel_id += 1


            rel_id += 1

    f_sentences.close()
    f_features.close()

    return rel_id, relationships


def load_shingles(shingles_file):
    """
    Parses already extracted shingles from a file.
    File format is: relaltionship_type \t shingle1 shingle2 shingle3 ... shingle_n
    """
    relationships = []
    rel_id = 0
    f_shingles = codecs.open(shingles_file, encoding='utf-8')

    for line in f_shingles:
        sys.stdout.write('.')
        rel_type, shingles_string = line.split('\t')
        shingles = shingles_string.strip().split(' ')

        # calculate min-hash sigs
        sigs = MinHash.signature(shingles, N_SIGS)
        rel_id = None
        rel_type = None
        rel_id += 1
    
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

    if sys.argv[1] == 'classify':
        lsh = LocalitySensitiveHashing(N_BANDS, N_SIGS, KNN)

        # read sentences from file, extract features/shingles and calculate min-hash sigs
        classify_sentences(sys.argv[2], lsh)

    ############################
    # INDEX TRAINING INSTANCES #
    ############################
    # argv[1] - index
    # argv[2] - directory with training data

    elif sys.argv[1] == 'index':
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
            #TODO: parallelize
            print "Loading PoS tagger"
            tagger = load('taggers/maxent_treebank_pos_tagger/english.pickle')
            reverb = Reverb()
            rel_id = 0
            print "Extracting features from training data and calculating min-hash sigs\n"
            for f in os.listdir(sys.argv[2]):
                current_file = os.path.join(sys.argv[2], f)
                rel_type = f.split("_")[0]
                print current_file, rel_type
                rel_id, relationships = load_training_relationships(current_file, rel_type, tagger, reverb, rel_id)
                print "\n"
                print "Indexing ", len(relationships), "relationships"
                print "MinHash Signatures : ", N_SIGS
                print "Bands              : ", N_BANDS
                print
                lsh = LocalitySensitiveHashing(N_BANDS, N_SIGS, KNN)
                lsh.create()
                for r in relationships:
                    lsh.index(r)

if __name__ == "__main__":
    main()
