#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import os
import sys
import codecs
import MinHash

from FeatureExtractor import FeatureExtractor
from LocalitySensitiveHashing import LocalitySensitiveHashing

N_BANDS = 50
N_SIGS = 600
KNN = 7
CONTEXT_WINDOW = 3
N_GRAMS_SIZE = 4
MAX_TOKENS = 6
MIN_TOKENS = 1


def classify_sentences(data_file, lsh):
    #TODO: isto também pode ser paralelizado
    count = 0
    fe = FeatureExtractor()
    f_sentences = codecs.open(data_file, encoding='utf-8')
    for line in f_sentences:
        if line.startswith("#"):
            continue
        else:
            # extract features
            rel, shingles = fe.process_classify(line)

            # calculate min-hash sigs
            sigs = MinHash.signature(shingles.getvalue().split(), N_SIGS)

            # find closest neighbours
            types = lsh.classify(sigs)
            if types is not None:
                print rel.e1, rel.e2
                print rel.sentence.encode("utf8")
                print types.encode("utf8")
                print

        count += 1
        if count % 100 == 0:
            sys.stdout.write(".")

    f_sentences.close()


def load_training_relationships(data_file, rel_type, rel_id):
    relationships = []
    f_sentences = codecs.open(data_file, encoding='utf-8')
    f_features = open('features.txt', 'w')
    fe = FeatureExtractor()
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
            # extract features and calculate min-hash sigs
            shingles = fe.process_index(sentence, e1, e2)
            sigs = MinHash.signature(shingles.getvalue().split(), N_SIGS)
            relationships.append((rel_type, rel_id, sigs))
            if rel_id % 100 == 0:
                sys.stdout.write(".")
            f_features.write(str(rel_id)+'\t'+shingles.getvalue()+'\n')
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
            rel_id = 0
            print "Extracting features from training data and calculating min-hash sigs\n"
            print "MinHash Signatures : ", N_SIGS
            print "Bands              : ", N_BANDS
            for f in os.listdir(sys.argv[2]):
                current_file = os.path.join(sys.argv[2], f)
                rel_type = f.split("_")[0]
                print "\n"+current_file, rel_type
                rel_id, relationships = load_training_relationships(current_file, rel_type, rel_id)
                lsh = LocalitySensitiveHashing(N_BANDS, N_SIGS, KNN)
                lsh.create()
                print "\nIndexing ", len(relationships), "relationships"
                for r in relationships:
                    lsh.index(r)

if __name__ == "__main__":
    main()
