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
    elapsed_time = 0
    for line in f_sentences:
        if line.startswith("#"):
            continue
        else:
            start_time = time.time()
            # extract features
            rel, shingles = fe.process_classify(line)
            # compute signatures
            sigs = MinHash.signature(shingles.getvalue().split(), N_SIGS)
            # find closest neighbours
            types = lsh.classify(sigs)
            elapsed_time += time.time() - start_time
            if types is not None:
                f_output.write("instance: " + rel.e1+"\t"+rel.e2+'\n')
                f_output.write("sentence: " + rel.sentence.encode("utf8")+"\n")
                f_output.write("rel_type: " + types.encode("utf8")+'\n\n')

        count += 1
        if count % 100 == 0:
            sys.stdout.write("Processed " + str(count) + " in %.2f seconds" % elapsed_time+"\n")
            f_output.flush()

    f_output.close()
    f_sentences.close()


def process_training_data(data_file):
    #TODO: parallelize
    print "Extracting features from training data and indexing in LSH\n"
    print "MinHash Signatures : ", N_SIGS
    print "Bands              : ", N_BANDS
    print
    relationships = []
    f_sentences = codecs.open(data_file, encoding='utf-8')
    lsh = LocalitySensitiveHashing(N_BANDS, N_SIGS, KNN)
    lsh.create()
    fe = FeatureExtractor()
    count = 0
    elapsed_time = 0
    for line in f_sentences:
        if line == '\n':
            continue
        rel_id, rel_type, e1, e2, sentence = line.split('\t')
        rel_id = int(rel_id.split(":")[1])
        start_time = time.time()
        shingles = fe.process_index(sentence, e1, e2)
        shingles = shingles.getvalue().strip().split(' ')
        sigs = MinHash.signature(shingles, N_SIGS)
        lsh.index(rel_type, rel_id, sigs)
        elapsed_time += time.time() - start_time
        relationships.append((rel_type, rel_id, sigs, shingles))
        count += 1
        if count % 100 == 0:
            sys.stdout.write("Processed " + str(count) + " in %.2f seconds" % elapsed_time+"\n")

    sys.stdout.write("Total processing time" + " in %.2f seconds" % elapsed_time+"\n")
    f_sentences.close()
    f_features = open('features.txt', 'w')

    # write shingles to file
    for rel in relationships:
        f_features.write(str(rel[1])+'\t'+rel[0].decode("utf8")+'\t'+' '.join(rel[3])+'\n')
    f_features.close()

    return relationships


def index_shingles(shingles_file):
    """
    Parses already extracted shingles from a file.
    File format is: relaltionship_type \t shingle1 shingle2 shingle3 ... shingle_n
    """
    print "Indexing relationships"
    print "MinHash Signatures : ", N_SIGS
    print "Bands              : ", N_BANDS
    print
    lsh = LocalitySensitiveHashing(N_BANDS, N_SIGS, KNN)
    lsh.create()
    f_shingles = codecs.open(shingles_file, encoding='utf-8')
    count = 0
    elapsed_time = 0
    for line in f_shingles:
        rel_id, rel_type, shingles = line.split('\t')
        shingles = shingles.strip().split(' ')
        start_time = time.time()
        sigs = MinHash.signature(shingles, N_SIGS)
        lsh.index(rel_type, rel_id, sigs)
        elapsed_time += time.time() - start_time
        count += 1
        if count % 100 == 0:
            sys.stdout.write("Processed " + str(count) + " in %.2f seconds" % elapsed_time + "\n")

    sys.stdout.write("Total Indexing time: %.2f seconds" % elapsed_time + "\n")
    f_shingles.close()


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
            index_shingles('features.txt')

        # load sentences, extract features, calculate min-hash sigs, index in bands
        else:
            process_training_data(sys.argv[2])

if __name__ == "__main__":
    main()
