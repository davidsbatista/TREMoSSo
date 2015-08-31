#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import re
import sys
import json

from os import listdir
from os.path import isfile, join


categories = ['YAGO_wikicategory_Democratic_Party_United_States_Senators',
              'YAGO_wikicategory_Democratic_Party_Presidents_of_the_United_States',
              'YAGO_wikicategory_Democratic_Party_state_governors_of_the_United_States',
              'YAGO_wikicategory_People_associated_with_the_Tea_Party_movement',
              'YAGO_wikicategory_Republican_Party_United_States_Senators']


def load_sentences(directory):
    #sentences = multiprocessing.Manager.Queue()
    count = 0
    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    for f in files:
        if count % 100 == 0:
            sys.stdout.write(".")
        if f.endswith('.json'):
            #print join(directory, f)
            with open(join(directory, f)) as data_file:
                data = json.load(data_file)
                # first load the entities identified into a strucutre which
                # can be searched as we process each sentence
                entities_data = data['entityMetadata']

                # go through all the sentences and select only sentences that have at least two entities
                # grounded to dbpedia/wikipedia and whose type is organisation, person or location
                sentences = data['annotatedText'].split('\n')
                for s in sentences:
                    number_valid_entities = 0
                    valid_entities = set()

                    # extract all entities in a sentence with a regex
                    wikilink_rx = re.compile(r'\[\[[^\]]+\]\]')
                    entities = re.findall(wikilink_rx, s)

                    # select only entities that are grounded to an URL
                    for e in entities:
                        entity_id = e.split('[[')[1].split('|')[0]
                        try:
                            if 'url' in entities_data[entity_id]:
                                valid_entities.add(e)
                                number_valid_entities += 1
                        except KeyError:
                            pass

                    # from the grounded entities select only those
                    # from a given wiki_category
                    for e in valid_entities:
                        entity_name = re.search(r'(YAGO:[^\|]+)', e).group()
                        for categ in  entities_data[entity_name]['type']:
                            if categ in categories:
                                print e, entity_name, categ

        count += 1


def main():
    load_sentences(sys.argv[1])

if __name__ == "__main__":
    main()
