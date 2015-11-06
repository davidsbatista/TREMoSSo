#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import os
import sys
import time
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
    fe = FeatureExtractor()
    f_sentences = codecs.open(data_file, encoding='utf-8')
    f_output = open("results.txt", "wb")
    count = 0
    start = time.time()
    for line in f_sentences:
        if line.startswith("#"):
            continue
        else:
            rel, shingles = fe.process_classify(line)
            sigs = MinHash.signature(shingles.getvalue().split(), N_SIGS)

            # find closest neighbours
            types = lsh.classify(sigs)
            if types is not None:
                f_output.write("instance: " + rel.e1+"\t"+rel.e2+'\n')
                f_output.write("sentence: " + rel.sentence.encode("utf8")+"\n")
                f_output.write("rel_type: " + types.encode("utf8")+'\n\n')

        count += 1
        if count % 100 == 0:
            sys.stdout.write("Processed " + str(count) + " in %.2f seconds" % (time.time() - start)+"\n")
            f_output.flush()

    f_output.close()
    f_sentences.close()


def load_training_relationships(data_file):
    #TODO: parallelize
    relationships = []
    f_sentences = codecs.open(data_file, encoding='utf-8')
    fe = FeatureExtractor()
    start = time.time()

    for line in f_sentences:
        if line == '\n':
            continue
        rel_id, rel_type, e1, e2, sentence = line.split('\t')
        rel_id = int(rel_id.split(":")[1])
        shingles = fe.process_index(sentence, e1, e2)
        sigs = MinHash.signature(shingles.getvalue().split(), N_SIGS)
        relationships.append((rel_type, rel_id, sigs, shingles))
        if rel_id % 100 == 0:
            sys.stdout.write("Processed " + str(rel_id) + " in %.2f seconds" % (time.time() - start)+"\n")

    sys.stdout.write("Total processing time" + " in %.2f seconds" % (time.time() - start)+"\n")
    f_sentences.close()
    f_features = open('features.txt', 'w')

    # write shingles to file
    for rel in relationships:
        f_features.write(str(rel[1])+'\t'+rel[0].decode("utf8")+'\t'+' '.join(rel[3].getvalue().split())+'\n')
    f_features.close()

    return relationships


def load_shingles(shingles_file):
    """
    Parses already extracted shingles from a file.
    File format is: relaltionship_type \t shingle1 shingle2 shingle3 ... shingle_n
    """
    start = time.time()
    relationships = []
    f_shingles = codecs.open(shingles_file, encoding='utf-8')
    for line in f_shingles:
        rel_id, rel_type, shingles = line.split('\t')
        rel_id = int(rel_id)
        shingles = shingles.strip().split(' ')
        # calculate min-hash sigs
        sigs = MinHash.signature(shingles, N_SIGS)
        relationships.append((rel_type, rel_id, sigs, shingles))
        if rel_id % 100 == 0:
            sys.stdout.write("Processed " + str(rel_id) + " in %.2f seconds" % (time.time() - start)+"\n")
    f_shingles.close()

    return relationships


def index_sigs(relationships):
    print "\nIndexing ", len(relationships), "relationships"
    print "MinHash Signatures : ", N_SIGS
    print "Bands              : ", N_BANDS
    lsh = LocalitySensitiveHashing(N_BANDS, N_SIGS, KNN)
    lsh.create()
    start = time.time()
    for r in relationships:
        lsh.index(r)
    sys.stdout.write("Indexing time: %.2f seconds" % (time.time() - start) + "\n")


def main():

    ###########################
    # CLASSIFY A NEW SENTENCE
    ###########################
    # argv[1] - classify
    # argv[2] - sentence to classify

    if sys.argv[1] == 'classify':
        lsh = LocalitySensitiveHashing(N_BANDS, N_SIGS, KNN)
        classify_sentences(sys.argv[2], lsh)

    ############################
    # INDEX TRAINING INSTANCES #
    ############################
    # argv[1] - index
    # argv[2] - file with training sentences

    elif sys.argv[1] == 'index':
        # calculate min-hash sigs (from already extracted shingles) index in bands
        if os.path.isfile("features.txt"):
            print "Calculating min-hash sigs from features.txt file"
            relationships = load_shingles('features.txt')
            index_sigs(relationships)

        # load sentences, extract features, calculate min-hash sigs, index in bands
        else:
            print "Extracting features from training data and calculating min-hash sigs\n"
            relationships = load_training_relationships(sys.argv[2])
            index_sigs(relationships)

if __name__ == "__main__":
    main()
