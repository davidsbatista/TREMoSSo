#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import cPickle
import sys


def main():
    rel_file = sys.argv[1]
    f = open(rel_file)
    print "\nLoading high PMI facts not in the database", rel_file
    relationships = cPickle.load(f)
    for r in relationships:
        print "sentence    :", r.sentence
        print "relationship:", r.e1, r.e2
        print "\n"
    print len(relationships)
    f.close()


if __name__ == "__main__":
    main()