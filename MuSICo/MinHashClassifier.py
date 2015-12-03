#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import multiprocessing
import os
import sys
import time
import codecs
import MinHash
import Queue

from FeatureExtractor import FeatureExtractor
from LocalitySensitiveHashing import LocalitySensitiveHashing

N_BANDS = 200
N_SIGS = 600
KNN = 7
CONTEXT_WINDOW = 3
N_GRAMS_SIZE = 4
MAX_TOKENS = 6
MIN_TOKENS = 1


def classify_sentences(data_file, lsh):
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    print "\nLoading sentences from file"
    f_sentences = codecs.open(data_file, encoding='utf-8')
    count = 0
    for line in f_sentences:
        if line.startswith("#") or line.startswith("\n"):
            continue
        count += 1
        if count % 5000 == 0:
            sys.stdout.write(".")
        queue.put(line.strip())
    f_sentences.close()

    print queue.qsize(), "sentences loaded"

    relationships = []
    num_cpus = 12
    pipes = [multiprocessing.Pipe(False) for _ in range(num_cpus)]
    processes = [multiprocessing.Process(target=classify, args=(queue, lsh, pipes[i][1])) for i in range(num_cpus)]

    print "\nClassifying relationship from sentences"
    print "Running", len(processes), "processes"
    start_time = time.time()
    for proc in processes:
        proc.start()

    for i in range(len(pipes)):
        data = pipes[i][0].recv()
        child_instances = data
        for x in child_instances:
            relationships.append(x)
        pipes[i][0].close()

    for proc in processes:
        proc.join()

    elapsed_time = time.time() - start_time
    sys.stdout.write("Processed " + str(count) + " in %.2f seconds" % elapsed_time+"\n")

    f_output = open("classified_sentences.txt", "w")
    for rel in relationships:
        f_output.write("instance: " + rel[0]+"\t"+rel[1]+'\n')
        f_output.write("sentence: " + rel[2].encode("utf8")+"\n")
        f_output.write("rel_type: " + rel[3].encode("utf8")+'\n\n')
    f_output.close()


def classify(queue, lsh, child_conn):
    fe = FeatureExtractor()
    count = 0
    classified_relationships = []
    while True:
        try:
            line = queue.get_nowait()
            count += 1
            if count % 1000 == 0:
                print count, " processed, remaining ", queue.qsize()

            relationships = fe.process_classify(line)
            for r in relationships:
                rel = r[0]
                shingles = r[1]

                # compute signatures
                sigs = MinHash.signature(shingles.getvalue().split(), N_SIGS)

                # find closest neighbours
                types = lsh.classify(sigs)
                if types is not None:
                    classified_r = (rel.e1, rel.e2, rel.sentence, types.encode("utf8"))
                else:
                    classified_r = (rel.e1, rel.e2, rel.sentence, "None")
                classified_relationships.append(classified_r)

        except Queue.Empty:
            print multiprocessing.current_process(), "Queue is Empty"
            child_conn.send(classified_relationships)
            break


def process_training_data(data_file):
    print "Extracting features from training data and indexing in LSH\n"
    print "MinHash Signatures : ", N_SIGS
    print "Bands              : ", N_BANDS
    print
    lsh = LocalitySensitiveHashing(N_BANDS, N_SIGS, KNN)
    lsh.create()

    # parallelization
    # read file into a queue
    # each process runs a FeatureExtractior
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    print "\nLoading sentences from file"
    f_sentences = codecs.open(data_file, encoding='utf-8')
    count = 0
    for line in f_sentences:
        if line.startswith("#") or line.startswith("\n"):
            continue
        count += 1
        if count % 10000 == 0:
            sys.stdout.write(".")
        queue.put(line.strip())
    f_sentences.close()
    print queue.qsize(), "sentences loaded"

    num_cpus = 12
    relationships = []
    pipes = [multiprocessing.Pipe(False) for _ in range(num_cpus)]
    processes = [multiprocessing.Process(target=extract_features, args=(queue, lsh, pipes[i][1])) for i in range(num_cpus)]

    print "\nIndexing relationship instances from sentences"
    print "Running", len(processes), " processes"
    start_time = time.time()
    for proc in processes:
        proc.start()

    for i in range(len(pipes)):
        data = pipes[i][0].recv()
        child_instances = data
        for x in child_instances:
            relationships.append(x)
        pipes[i][0].close()

    for proc in processes:
        proc.join()

    elapsed_time = time.time() - start_time
    sys.stdout.write("Processed " + str(count) + " in %.2f seconds" % elapsed_time+"\n")

    # write shingles to file
    f_features = open("features.txt", "w")
    for rel in relationships:
        f_features.write(str(rel[1])+'\t'+rel[0].decode("utf8")+'\t'+' '.join(rel[3])+'\n')
    f_features.close()

    return relationships


def extract_features(queue, lsh, child_conn):
    fe = FeatureExtractor()
    relationships = []
    count = 0
    while True:
        try:
            line = queue.get_nowait()
            count += 1
            if count % 100 == 0:
                print count, " processed, remaining ", queue.qsize()

            rel_id, rel_type, e1, e2, sentence = line.split('\t')
            rel_id = int(rel_id.split(":")[1])
            shingles = fe.process_index(sentence, e1, e2)

            try:
                shingles = shingles.getvalue().strip().split(' ')
            except AttributeError, e:
                print line
                print shingles
                sys.exit(-1)

            sigs = MinHash.signature(shingles, N_SIGS)
            lsh.index(rel_type, rel_id, sigs)
            relationships.append((rel_type, rel_id, sigs, shingles))

        except Queue.Empty:
            print multiprocessing.current_process(), "Queue is Empty"
            child_conn.send(relationships)
            break


def index_shingles(shingles_file):
    """
    Parses already extracted shingles from a file.
    File format is: relaltionship_type \t shingle1 shingle2 shingle3 ... shingle_n
    """
    f_shingles = codecs.open(shingles_file, encoding='utf-8')
    relationships = []
    print "Reading features file"
    for line in f_shingles:
        rel_id, rel_type, shingles = line.split('\t')
        shingles = shingles.strip().split(' ')
        relationships.append((rel_type, rel_id, shingles))
    f_shingles.close()

    lsh = LocalitySensitiveHashing(N_BANDS, N_SIGS, KNN)
    lsh.create()
    count = 0
    elapsed_time = 0

    for r in relationships:
        start_time = time.time()
        sigs = MinHash.signature(r[2], N_SIGS)
        lsh.index(r[0], r[1], sigs)
        elapsed_time += time.time() - start_time
        count += 1
        if count % 100 == 0:
            sys.stdout.write("Processed " + str(count) + " in %.2f seconds" % elapsed_time + "\n")

    sys.stdout.write("Total Indexing time: %.2f seconds" % elapsed_time + "\n")


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
