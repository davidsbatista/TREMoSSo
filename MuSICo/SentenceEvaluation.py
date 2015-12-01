#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import re
from nltk import word_tokenize
from nltk.corpus import stopwords

# tokens between entities which do not represent relationships
bad_tokens = [",", "(", ")", ";", "''",  "``", "'s", "-", "vs.", "v", "'", ":", ".", "--"]
stopwords = stopwords.words('english')
not_valid = bad_tokens + stopwords


def tokenize_entity(entity):
    parts = word_tokenize(entity)
    if parts[-1] == '.':
        replace = parts[-2] + parts[-1]
        del parts[-1]
        del parts[-1]
        parts.append(replace)
    return parts


def find_locations(entity_string, text_tokens):
    locations = []
    e_parts = tokenize_entity(entity_string)
    for i in range(len(text_tokens)):
        if text_tokens[i:i + len(e_parts)] == e_parts:
            locations.append(i)
    return e_parts, locations


class EntitySimple:
    def __init__(self, _e_string, _e_parts, _e_type, _locations):
        self.string = _e_string
        self.parts = _e_parts
        self.type = _e_type
        self.locations = _locations

    def __hash__(self):
        return hash(self.string) ^ hash(self.type)

    def __eq__(self, other):
        return self.string == other.string and self.type == other.type


class EntityLinked:
    def __init__(self, _e_string, _e_parts, _e_type, _locations, _url=None):
        self.string = _e_string
        self.parts = _e_parts
        self.type = _e_type
        self.locations = _locations
        self.url = _url

    def __hash__(self):
        return hash(self.url)

    def __eq__(self, other):
        return self.url == other.url


class Relationship:
    def __init__(self, _sentence, _before, _between, _after, _ent1, _ent2, e1_type, e2_type):
        self.sentence = _sentence
        self.before = _before
        self.between = _between
        self.after = _after
        self.e1 = _ent1
        self.e2 = _ent2
        self.e1_type = e1_type
        self.e2_type = e2_type

    def __eq__(self, other):
        if self.e1 == other.e1 and self.before == other.before and self.between == other.between \
                and self.after == other.after:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.e1) ^ hash(self.e2) ^ hash(''.join(self.before)) ^ hash(''.join(self.between))\
               ^ hash(''.join(self.after))


class SentenceEvaluation:

    def __init__(self, sentence, e1_type, e2_type, max_tokens, min_tokens, window_size, pos_tagger=None, config=None):
        self.relationships = list()

        #determine which type of regex to use according to how named-entties are tagged
        entities_regex = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)
        regex_clean_simple = re.compile('</?[A-Z]+>', re.U)

        # find named-entities
        entities = []
        for m in re.finditer(entities_regex, sentence):
            entities.append(m)

        if len(entities) >= 2:
            # clean tags from text
            sentence_no_tags = re.sub(regex_clean_simple, "", sentence)
            text_tokens = word_tokenize(sentence_no_tags)

            # extract information about the entity, create an Entity instance and store in a
            # structure to hold information collected about all the entities in the sentence
            entities_info = set()
            for x in range(0, len(entities)):
                entity = entities[x].group()
                e_string = re.findall('<[A-Z]+>([^<]+)</[A-Z]+>', entity)[0]
                e_type = re.findall('<([A-Z]+)', entity)[0]
                e_parts, locations = find_locations(e_string, text_tokens)
                e = EntitySimple(e_string, e_parts, e_type, locations)
                entities_info.add(e)

            # create an hashtable:
            # - the key is the starting index in the tokenized sentence of an entity
            # - the value the corresponding Entity instance
            locations = dict()
            for e in entities_info:
                for start in e.locations:
                    locations[start] = e

            # look for pair of entities such that:
            # the distance between the two entities is less than 'max_tokens' and greater than 'min_tokens'
            # the arguments match the seeds semantic types
            sorted_keys = list(sorted(locations))
            for i in range(len(sorted_keys)-1):
                distance = sorted_keys[i+1] - sorted_keys[i]
                e1 = locations[sorted_keys[i]]
                e2 = locations[sorted_keys[i+1]]
                if max_tokens >= distance >= min_tokens and e1.type == e1_type and e2.type == e2_type:

                    # ignore relationships between the same entity
                    if e1.string == e2.string:
                        continue

                    before = text_tokens[:sorted_keys[i]]
                    before = before[-window_size:]
                    between = text_tokens[sorted_keys[i]+len(e1.parts):sorted_keys[i+1]]
                    after = text_tokens[sorted_keys[i+1]+len(e2.parts):]
                    after = after[:window_size]

                    # ignore relationships where BET context is only stopwords or other invalid words
                    if all(x in not_valid for x in text_tokens[sorted_keys[i]+len(e1.parts):sorted_keys[i+1]]):
                        continue

                    r = Relationship(sentence, before, between, after, e1.string, e2.string, e1.type, e2.type)
                    self.relationships.append(r)