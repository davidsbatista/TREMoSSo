#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import codecs
import sys
import os


def main():
    rel_id = 0
    for f in os.listdir(sys.argv[1]):
        current_file = os.path.join(sys.argv[1], f)
        rel_type = f.split("_")[0]
        f_sentences = codecs.open(current_file, encoding='utf-8')
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

            if line.startswith("passive voice:") and e1 is not None and e2 is not None:
                passive_voice = eval(line.split("passive voice:")[1].strip())

            if line == '\n':

                # extract features and calculate min-hash sigs
                if passive_voice is True and rel_type not in ['owns2', 'founder']:
                    print "id:"+str(rel_id), '\t', rel_type, '\t', e2, '\t', e1, '\t', sentence
                elif passive_voice is False:
                    print "id:"+str(rel_id), '\t', rel_type, '\t', e1, '\t', e2, '\t', sentence

                rel_id += 1

if __name__ == "__main__":
    main()
