#!/bin/bash

N_SIGS=200
N_BANDS=25
KNN=7

# clear REDIS tables
redis-cli flushall

# link to file with generated features
ln -s ../features.txt

# index training data
../MinHashClassifier.py index ../training_data.txt ${N_BANDS} ${N_SIGS}

# classifiy training data
../MinHashClassifier.py classify2 ../set_b_shingles.txt ${N_BANDS} ${N_SIGS} ${KNN}

# create a separate file for each relationship type
../musico2evaluation.sh

# link to files needed by evaluation framework
for i in ../superset_*; do ln -vs ${i}; done
for i in ../*high_pmi*; do ln -sv ${i}; done

# evaluate each relationship type
../evaluate-musico.sh

# output results to single file
for i in *_results.txt; do echo ${i}; tail -n 5 ${i}; done >> results.txt
