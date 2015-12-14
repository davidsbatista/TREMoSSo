#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"


def main():
    tmp = sys.argv[1]
    sigs, bands, knn = tmp.split("/")[0].split(".")
    acc_precision = 0
    acc_recall = 0
    acc_f1 = 0
    with open(sys.argv[1], "r") as f:
        for line in f:
            if line.endswith(".txt\n"):
                rel_type = line.split("_")[0]
            elif line.startswith("Precision"):
                precision = line.split(":  ")[1]
            elif line.startswith("Recall"):
                recall = line.split(":  ")[1]
            elif line.startswith("F1"):
                f1 = line.split(":  ")[1]
                """
                print rel_type
                print "Precision:", precision,
                print "Recall   :", recall,
                print "F1       :", f1,
                print
                """
                acc_precision += float(precision)
                acc_recall += float(recall)
                acc_f1 += float(f1)

    print "SIGS", sigs
    print "BADNS", bands
    print "KNN", knn
    print "F1", acc_f1 / float(11)
    print

if __name__ == "__main__":
    main()
