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

CONTEXT_WINDOW = 3
N_GRAMS_SIZE = 4
MAX_TOKENS = 6
MIN_TOKENS = 1


def classify_sentences2(data_file, lsh, n_sigs):
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    print "\nLoading sentences/shingles from file"
    f_sentences = codecs.open(data_file, encoding='utf-8')
    count = 0
    for line in f_sentences:

        if line.startswith("e1"):
            e1 = line.split("e1: ")[1].strip()

        if line.startswith("e2"):
            e2 = line.split("e2: ")[1].strip()

        if line.startswith("sentence:"):
            sentence = line.split("sentence: ")[1].strip()

        if line.startswith("shingles:"):
            shingles = line.split("shingles: ")[1].strip()

        if line.startswith("\n"):
            rel = (e1, e2, sentence, shingles)
            queue.put(rel)
            count += 1

            if count % 50000 == 0:
                sys.stdout.write(".")

    f_sentences.close()

    print "\n", queue.qsize(), "sentences loaded"

    relationships = []
    num_cpus = 12
    pipes = [multiprocessing.Pipe(False) for _ in range(num_cpus)]
    processes = [multiprocessing.Process(target=classify2, args=(queue, n_sigs, lsh, pipes[i][1])) for i in range(num_cpus)]

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


def classify2(queue, n_sigs, lsh, child_conn):
    count = 0
    classified_relationships = []
    print multiprocessing.current_process(), "started"
    while True:
        try:
            r = queue.get_nowait()
            count += 1
            if count % 1000 == 0:
                print multiprocessing.current_process(), count, " processed, remaining ", queue.qsize()

            e1 = r[0]
            e2 = r[1]
            sentence = r[2]
            shingles = r[3]

            # compute signatures
            sigs = MinHash.signature(shingles.split(), n_sigs)

            # find closest neighbours
            types = lsh.classify(sigs)
            if types is not None:
                classified_r = (e1, e2, sentence, types.encode("utf8"))
            else:
                classified_r = (e1, e2, sentence, "None")

            classified_relationships.append(classified_r)

        except Queue.Empty:
            print multiprocessing.current_process(), "Queue is Empty"
            child_conn.send(classified_relationships)
            break


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

    """
    f_output = open("set_b_shingles.txt", "w")
    for r in relationships:
        f_output.write("e1: "+r[0].e1.encode("utf8")+'\n')
        f_output.write("e2: "+r[0].e2.encode("utf8")+'\n')
        f_output.write("sentence: "+r[0].sentence.encode("utf8")+'\n')
        f_output.write("shingles: "+r[1].getvalue()+'\n\n')
    f_output.close()
    """

    f_output = open("classified_sentences.txt", "w")
    for rel in relationships:
        f_output.write("instance: " + rel[0]+"\t"+rel[1]+'\n')
        f_output.write("sentence: " + rel[2].encode("utf8")+"\n")
        f_output.write("rel_type: " + rel[3].encode("utf8")+'\n\n')
    f_output.close()


def classify(queue, lsh, child_conn, n_sigs):
    fe = FeatureExtractor()
    count = 0
    classified_relationships = []
    print multiprocessing.current_process(), "started"
    while True:
        try:
            line = queue.get_nowait()
            count += 1
            if count % 1000 == 0:
                print multiprocessing.current_process(), count, " processed, remaining ", queue.qsize()

            relationships = fe.process_classify(line)

            for r in relationships:
                rel = r[0]
                shingles = r[1]

                # compute signatures
                sigs = MinHash.signature(shingles.getvalue().split(), n_sigs)

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


def process_training_data(data_file, n_sigs, n_bands):
    print "Extracting features from training data and indexing in LSH\n"
    print "MinHash Signatures : ", n_sigs
    print "Bands              : ", n_bands
    print
    lsh = LocalitySensitiveHashing(n_bands, n_sigs)
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


def extract_features(queue, lsh, child_conn, n_sigs):
    fe = FeatureExtractor()
    relationships = []
    count = 0
    while True:
        try:
            line = queue.get_nowait()
            count += 1
            if count % 1000 == 0:
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

            sigs = MinHash.signature(shingles, n_sigs)
            lsh.index(rel_type, rel_id, sigs)
            relationships.append((rel_type, rel_id, sigs, shingles))

        except Queue.Empty:
            print multiprocessing.current_process(), "Queue is Empty"
            child_conn.send(relationships)
            break


def index_shingles(shingles_file, n_bands, n_sigs, knn):
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

    print "SIGS  :", n_sigs
    print "BANDS :", n_bands

    lsh = LocalitySensitiveHashing(n_bands, n_sigs, knn)
    lsh.create()
    count = 0
    elapsed_time = 0

    for r in relationships:
        start_time = time.time()
        sigs = MinHash.signature(r[2], n_sigs)
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

    n_bands = int(sys.argv[3])
    n_sigs = int(sys.argv[4])
    knn = int(sys.argv[5])

    if sys.argv[1] == 'classify':
        knn = int(sys.argv[5])
        lsh = LocalitySensitiveHashing(n_bands, n_sigs, knn)
        classify_sentences(sys.argv[2], lsh)

    ####################################################################
    # CLASSIFY A NEW SENTENCE WHERE THE SHINLGES WHERE ALREADY EXTRACTED
    ####################################################################
    # argv[1] - classify2
    # argv[2] - sentence to classify

    elif sys.argv[1] == 'classify2':
        lsh = LocalitySensitiveHashing(n_bands, n_sigs, knn)
        classify_sentences2(sys.argv[2], lsh, n_sigs)

    ############################
    # INDEX TRAINING INSTANCES #
    ############################
    # argv[1] - index
    # argv[2] - file with training sentences

    elif sys.argv[1] == 'index':
        # calculate min-hash sigs (from already extracted shingles) index in bands
        if os.path.isfile("features.txt"):
            index_shingles('features.txt', n_bands, n_sigs, knn)

        # load sentences, extract features, calculate min-hash sigs, index in bands
        else:
            process_training_data(sys.argv[2], n_sigs, n_bands)

if __name__ == "__main__":
    main()
