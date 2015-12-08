#!/bin/bash

# Make sure REDIS is running
# sudo /etc/init.d/redis-server start

N_SIGS=(100 200 300 400 500 600)
N_BANDS=(25 50 75 100 150)
KNN=(1 3 5 7)

for s in 100 200 300 400 500 600; do
    for b in 25 50 75 100 150; do
        for k in 1 3 5 7; do
            echo $s $b $k
        done
    done
done

# make dir
# change dir

# Index training data
# TODO: accept BANDS and SIGS from command line
#./MinHashClassifier.py index training_data.txt

# Classifiy training data:
#./MinHashClassifier.py classifiy2 <set_b_matched.txt>

