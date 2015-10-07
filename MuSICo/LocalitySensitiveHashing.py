#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import numpy
import sys
import redis
import pickle

from collections import defaultdict
from operator import itemgetter


class LocalitySensitiveHashing:

    n_sigs = None
    n_bands = None
    sigs_per_band = None
    kNN = None
    bands = []
    minhash_instances = None

    def __init__(self, n_bands, n_sigs, knn):
        if (n_sigs % n_bands) != 0:
            print "Error! Number of hash signatures must be a multiple of number of bands"
            sys.exit(0)
        else:
            self.n_sigs = n_sigs
            self.n_bands = n_bands
            self.sigs_per_band = n_sigs / n_bands
            self.kNN = knn

            for i in range(0, n_bands):
                conn = redis.StrictRedis(host='localhost', port=6379, db=i)
                self.bands.append(conn)
            self.minhash_instances = redis.StrictRedis(host='localhost', port=6379, db=n_bands+1)

    def index(self, rel):
        # generates a list of equaly sized chunks (arrays) from the min-hash array
        # (rel_type, rel_id, sigs)
        chunked = [s for s in self.chunks(rel[2], self.sigs_per_band)]
        for i in range(0, len(chunked)):
            sorted_chunk = numpy.sort(chunked[i])
            self.bands[i].set(tuple(sorted_chunk), (rel[0], rel[1]))

        # save the minh-hash signatures of the relationship
        # allows for real Jaccardi calculation in classication
        sigs_pickled = pickle.dumps(rel[2])
        self.minhash_instances.set(rel[1], sigs_pickled)

    def classify(self, sigs):
        # generates a list of equaly sized chunks (arrays) from the min-hash array
        chunked = [s for s in self.chunks(sigs, self.sigs_per_band)]
        candidates = defaultdict(list)

        for i in range(0, len(chunked)):
            sorted_chunk = numpy.sort(chunked[i])
            # TODO, redis_i, ..., redis_n
            # .set(tuple(sorted_chunk)
            if self.bands[i].get(tuple(sorted_chunk)):
                value = eval(self.bands[i].get(tuple(sorted_chunk)))
                rel_type, rel_id = value[0], value[1]
                candidates[rel_type].append(rel_id)

        # get each candidate full signatures
        # stores candidate and approximate Jaccardi score
        scores = []
        for rel_type in candidates:
            for rel_id in candidates[rel_type]:
                # get candidate min-hash sigs
                candidate_sigs = self.minhash_instances.get(rel_id)

                # calculate aproximate Jaccardi Similarity using the full min-hash sigs
                # logical XOR between two array of min-hash sigs:
                #   - gives False if two elements are equal, True if two elements are different
                #   - return array of boolean values (False,True)
                # sum() of the array of boolean gives the number of True, i.e., ones
                #   - Jaccardi is number of equal signatures (False) over the total number of signatures
                candidate_sigs = pickle.loads(candidate_sigs)
                score = 1 - logical_xor(sigs, candidate_sigs).sum() / float(self.n_sigs)
                scores.append((rel_type, rel_id, score))

        if len(scores) == 0:
            return None

        else:
            rel_sorted = sorted(scores, key=itemgetter(2), reverse=True)
            # consider only the top kNN candidates
            top_k = rel_sorted[:self.kNN]
            output = dict()
            # choose the most common
            for x in top_k:
                if x[0] in output:
                    output[x[0]] += 1
                else:
                    output[x[0]] = 1

            output_sorted = sorted(output, key=itemgetter(1), reverse=True)
            return output_sorted[0]

    def create(self):
        for i in range(0, self.n_bands):
            self.bands.append(dict())

    @staticmethod
    def chunks(l, n):
        for i in xrange(0, len(l), n):
            yield l[i:i + n]


