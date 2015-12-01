#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import codecs
import fileinput
import functools
import multiprocessing
import time
import sys
import os
import cPickle
import Queue
import re

from whoosh.index import open_dir, os
from whoosh.query import spans
from whoosh import query
from nltk import word_tokenize, bigrams
from nltk.corpus import stopwords
from collections import defaultdict
from SentenceEvaluation import SentenceEvaluation

regex_clean_simple = re.compile('</?[A-Z]+>', re.U)
num_cpus = 0
q_limit = 50000

# relational words to be used in calculating the set C and D aplpying the  with the proximity PMI

founder_unigrams = ['founder', 'co-founder', 'cofounder', 'co-founded', 'cofounded', 'founded', 'founders']
founder_bigrams = ['started by']

founder2_unigrams = ['founder', 'co-founder', 'cofounder', 'co-founded', 'cofounded', 'founded', 'founders']
founder2_bigrams = ['started by']

owns_unigrams = ['owns', 'acquired', 'bought', 'acquisition']
owns_bigrams = ['has acquired', 'which acquired', 'was acquired', 'which owns', 'owned by']

owns2_unigrams = ['parent']
owns2_bigrams = ['unit of', 'subsidiary of', 'owned by']


######## INSTALLATIONS ########

installations_in_unigrams = ['headquarters', 'headquartered', 'offices', 'office']

installations_in_bigrams = [', based in', 'based in', 'located in', 'main office', 'main offices', 'offices in',
                            'building in', 'office in', 'head office', 'head offices']

# 'factory in', 'plant in',


######## AFFILIATION ########

affiliation_unigrams = ['chief', 'chairman', 'executive', 'ceo', 'CEO', 'employer',
                        'president', 'head', 'administrator', 'governor']
affiliation_bigrams = ['chief executive', 'technical chief', 'top executive']


######## STUDIED ########

studied_unigrams = ['graduated', 'graduate']
studied_bigrams = ['graduated from', 'earned phd', 'studied at', 'student at', 'graduated the']
studied_trigrams = ['studied music at', 'studied briefly at', 'graduated from the']
studied_quadrams = [', who graduated from', ', who enrolled at', 'is a graduate of']

studied2_unigrams = ['graduate', 'phd', 'doctorate', 'master', 'mba', 'studied', 'student']
studied2_bigrams = ['graduate', 'phd', 'doctorate', 'master', 'mba', 'studied', 'student']

######## LOCATED-IN ########

located_in_unigrams = ['capital', 'suburb', 'city', 'island', 'region']
located_in_bigrams = ['capital of', 'suburb of', 'city of', 'island', 'region of', 'in southern', 'in northern',
                      'northwest of', 'town in']


######## SPOUSE ########

spouse_unigrams = ['married', 'wife', 'husband']
spouse_bigrams = ['married to', 'his wife', 'her husband']


# tokens between entities which do not represent relationships
bad_tokens = ["(", ")", ";", "''",  "``", "'s", "-", "vs.", "v", "'", ":", ".", "--"]
stopwords = stopwords.words('english')
not_valid = bad_tokens + stopwords

# PMI value for proximity
PMI = 0.65

# Parameters for relationship extraction from Sentence
MAX_TOKENS_AWAY = 6
MIN_TOKENS_AWAY = 1
CONTEXT_WINDOW = 2

# DEBUG stuff
PRINT_NOT_FOUND = False

# stores all variations matched with database
manager = multiprocessing.Manager()
all_in_database = manager.dict()


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


class Triple(object):
    def __init__(self, _e1, _e2, _bet):
        self.e1 = _e1
        self.e2 = _e2
        self.bet_words = _bet

    def __hash__(self):
        sig = hash(self.e1) ^ hash(self.e2) ^ hash(self.bet_words)
        return sig

    def __eq__(self, other):
        if self.e1 == other.e1 and self.e2 == other.e2 and self.bet_words == other.bet_words:
            return True
        else:
            return False


class ExtractedFact(object):
    def __init__(self, e1, e2, bet_words, sentence):
        self.e1 = e1
        self.e2 = e2
        self.bet_words = bet_words
        self.sentence = sentence


####################################
# Misc., Utils, parsing text files #
####################################
def timecall(f):
    @functools.wraps(f)
    def wrapper(*args, **kw):
        start = time.time()
        result = f(*args, **kw)
        end = time.time()
        print "Time taken: %.2f seconds" % (end - start)
        return result

    return wrapper


def process_output(data):
    """
    parses the file with the relationships extracted by the system
    each relationship is transformed into a ExtracteFact class
    """
    system_output = list()
    for line in fileinput.input(data):

        if line.startswith('instance'):
            e1, e2 = line.split("instance:")[1].strip().split('\t')

        if line.startswith('sentence'):
            sentence = line.split("sentence: ")[1]

            sentence_no_tags = re.sub(regex_clean_simple, "", sentence)
            text_tokens = word_tokenize(sentence_no_tags)
            e1_info = find_locations(e1, text_tokens)
            e2_info = find_locations(e2, text_tokens)
            for e1_b in e1_info[1]:
                for e2_b in e2_info[1]:
                    distance = abs(e2_b - e1_b)
                    if distance > MAX_TOKENS_AWAY or distance < MIN_TOKENS_AWAY:
                        continue
                    else:
                        bet = text_tokens[e1_b+len(word_tokenize(e1)):e2_b]
                        bet_words = ' '.join(bet)

                    r = ExtractedFact(e1, e2, bet_words, sentence)
                    system_output.append(r)

    fileinput.close()
    return system_output


def process_dbpedia(data, database, file_rel_type, rel_type):
    for line in fileinput.input(data):
        try:
            e1, rel, e2, p = line.strip().split('>')
            e1 = e1.replace("<http://dbpedia.org/resource/", '')
            e2 = e2.replace("<http://dbpedia.org/resource/", '')
            e2_parts = e2.split(",_")
            pairs = list()
            for part in e2_parts:
                e1 = e1.replace("_", " ")
                e1 = re.sub(r'\([^\)]+\)', '', e1)
                e2 = part.replace("_", " ").strip()
                pairs.append((e1, e2))

        except ValueError:
            print "Error parsing", line
            sys.exit(0)

        #follow the order of the relationships as in the output
        for pair in pairs:
            e1 = pair[0]
            e2 = pair[1]
            if file_rel_type in ['dbpedia_capital.txt', 'dbpedia_largestCity.txt'] or rel_type in ['studied2',
                                                                                                   'founder2',
                                                                                                   'owns2',
                                                                                                   'has-installations2']:
                database[e2.strip().decode("utf8")].add(e1.strip().decode("utf8"))
            else:
                database[e1.strip().decode("utf8")].add(e2.strip().decode("utf8"))
    fileinput.close()


def process_yago(data, database, rel_type):
    for line in fileinput.input(data):
        try:
            e1, rel, e2 = line.strip().split('\t')
            if "/" in e1:
                e1 = e1.split("/")[1]
            if "/" in e2:
                e2 = e2.split("/")[1]
            e1 = e1.replace("<", "").replace(">", "")
            e2 = e2.replace("<", "").replace(">", "").strip().replace(".", "")
            e2_parts = e2.split(",_")
            pairs = list()
            for part in e2_parts:
                e1 = e1.replace("_", " ")
                e1 = re.sub(r'\([^\)]+\)', '', e1)
                e2 = part.replace("_", " ").strip()
                pairs.append((e1, e2))

        except ValueError:
            print "Error parsing", line
            sys.exit(0)

        # follow the order of the relationships as in the output
        for pair in pairs:
            e1 = pair[0]
            e2 = pair[1]
            if rel_type in ['founder', 'affiliation', 'studied2', 'owns2', 'has-installations2']:
                database[e2.strip().decode("utf8")].add(e1.strip().decode("utf8"))
            else:
                database[e1.strip().decode("utf8")].add(e2.strip().decode("utf8"))
    fileinput.close()


def process_freebase(data, database, rel_type):
    for line in fileinput.input(data):
        if line.startswith("#"):
            continue
        try:
            e1, rel, e2 = line.strip().split('\t')
        except ValueError:
            print "Error parsing", line
            sys.exit(0)

        # follow the order of the relationships as in the output
        if rel_type == 'spouse':
            database[e1.strip().decode("utf8")].add(e2.strip().decode("utf8"))
            database[e2.strip().decode("utf8")].add(e1.strip().decode("utf8"))

        elif rel_type in ['founder', 'affiliation', 'owns2', 'has-installations2']:
            database[e2.strip().decode("utf8")].add(e1.strip().decode("utf8"))
        else:
            database[e1.strip().decode("utf8")].add(e2.strip().decode("utf8"))
    fileinput.close()


def process_manually(data, database):
    for line in fileinput.input(data):
        if line.startswith("#"):
            continue
        try:
            e1, e2 = line.strip().split('\t')
        except ValueError:
            print "Error parsing", line
            sys.exit(0)

        database[e1.strip().decode("utf8")].add(e2.strip().decode("utf8"))

    fileinput.close()


def extract_bigrams(text):
    tokens = word_tokenize(text)
    return [gram[0]+' '+gram[1] for gram in bigrams(tokens)]


# ########################################
# Estimations of sets and intersections #
# ########################################
def calculate_a(not_in_database, e1_type, e2_type, index, rel_words_unigrams, rel_words_bigrams):
    m = multiprocessing.Manager()
    queue = m.Queue()
    results = [m.list() for _ in range(num_cpus)]
    not_found = [m.list() for _ in range(num_cpus)]

    for r in not_in_database:
        queue.put(r)

    processes = [multiprocessing.Process(target=proximity_pmi_a, args=(e1_type, e2_type, queue, index, results[i],
                                                                       not_found[i], rel_words_unigrams,
                                                                       rel_words_bigrams)) for i in range(num_cpus)]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()

    a = list()
    for l in results:
        a.extend(l)

    wrong = list()
    for l in not_found:
        wrong.extend(l)

    return a, wrong


def calculate_b_parallel(output, database):
    # intersection between the system output and the database
    # it is assumed that every fact in this region is correct
    m = multiprocessing.Manager()
    queue = m.Queue()
    results = [m.list() for _ in range(num_cpus)]
    no_matches = [m.list() for _ in range(num_cpus)]
    b = list()
    not_found = list()

    for r in output:
        queue.put(r)

    processes = [multiprocessing.Process(target=find_intersection, args=(results[i], no_matches[i], database, queue))
                 for i in range(num_cpus)]

    for proc in processes:
        proc.start()

    for proc in processes:
        proc.join()

    for result in results:
        for e in result:
            b.append(e)

    for no_match in no_matches:
        for e in no_match:
            not_found.append(e)

    return b, not_found


def calculate_c(corpus, database, b, e1_type, e2_type, rel_type, rel_words_unigrams, rel_words_bigrams):

    print rel_words_unigrams
    print rel_words_bigrams
    print

    # contains the database facts described in the corpus but not extracted by the system
    #
    # G' = superset of G, cartesian product of all possible entities and relations (i.e., G' = E x R x E)
    # for now, all relationships from a sentence
    print "Building G', a superset of G"
    m = multiprocessing.Manager()
    queue = m.Queue()
    g_dash = m.list()

    # check if superset G' for e1_type, e2_type already exists and
    # if G' minus KB for rel_type exists

    # if it exists load into g_dash_set
    if os.path.isfile("superset_" + e1_type + "_" + e2_type + ".pkl"):
        f = open("superset_" + e1_type + "_" + e2_type + ".pkl")
        print "\nLoading superset G'", "superset_" + e1_type + "_" + e2_type + ".pkl"
        g_dash_set = cPickle.load(f)
        f.close()

    # else generate G' and G minus D
    else:
        with codecs.open(corpus, 'r', 'utf8') as f:
            data = f.readlines()
            count = 0
            print "Storing in shared Queue"
            for l in data:
                if count % 50000 == 0:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                queue.put(l)
                count += 1
        print "\nQueue size:", queue.qsize()

        processes = [multiprocessing.Process(target=process_corpus, args=(queue, g_dash, e1_type, e2_type))
                     for _ in range(num_cpus)]

        print "Extracting all possible " + e1_type + "," + e2_type + " relationships from the corpus"
        print "Running", len(processes), "threads"

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        print len(g_dash), "relationships built"
        g_dash_set = set(g_dash)
        print len(g_dash_set), "unique relationships"
        print "Dumping into file", "superset_" + e1_type + "_" + e2_type + ".pkl"
        f = open("superset_" + e1_type + "_" + e2_type + ".pkl", "wb")
        cPickle.dump(g_dash_set, f)
        f.close()

    # Estimate G intersection D: look for facts in G' that a match a fact in the database
    if os.path.isfile(rel_type + "_g_intersection_d.pkl") and os.path.isfile(rel_type + "_g_minus_d.pkl"):
        f = open(rel_type + "_g_intersection_d.pkl", "r")
        print "\nLoading G intersected with D", rel_type + "_g_intersection_d.pkl"
        g_intersect_d = cPickle.load(f)
        f.close()

        f = open(rel_type + "_g_minus_d.pkl")
        print "\nLoading superset G' minus D", rel_type + "_g_minus_d.pkl"
        g_minus_d = cPickle.load(f)
        f.close()

    else:
        print "Estimating G intersection with D"
        g_intersect_d = set()
        print "G':", len(g_dash_set)
        print "Database:", len(database.keys())

        # Facts not in the database, to use in estimating set d
        g_minus_d = set()

        queue = manager.Queue()
        results = [manager.list() for _ in range(num_cpus)]
        no_matches = [manager.list() for _ in range(num_cpus)]

        # Load everything into a shared queue
        print "Loading data into a queue"
        count = 0
        for r in g_dash_set:
            queue.put(r)
            count += 1
            if count % 5000 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()

        print

        processes = [multiprocessing.Process(target=find_intersection, args=(results[i], no_matches[i], database,
                                                                             queue))
                     for i in range(num_cpus)]

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        for result in results:
            for e in result:
                g_intersect_d.add(e)

        for no_match in no_matches:
            for e in no_match:
                g_minus_d.add(e)

        print "Extra filtering: from the intersection of G' with D, select only those based on keywords"
        print "G intersection with D", len(g_intersect_d)
        filtered = set()

        for r in g_intersect_d:
            unigrams_bet = r.between
            unigrams_bef = r.before
            unigrams_aft = r.after
            bigrams_bet = extract_bigrams(' '.join(r.between))
            try:
                if any(x in rel_words_unigrams for x in unigrams_bet):
                    filtered.add(r)
                    continue
                if any(x in rel_words_unigrams for x in unigrams_bef):
                    filtered.add(r)
                    continue
                if any(x in rel_words_unigrams for x in unigrams_aft):
                    filtered.add(r)
                    continue
                elif any(x in rel_words_bigrams for x in bigrams_bet):
                    filtered.add(r)
                    continue
            except Exception, e:
                print r.between
                print r.before
                print r.after
                print r.sentence
                print "rel_words_bigrams", rel_words_bigrams
                print "rel_words_unigrams", rel_words_unigrams
                sys.exit(-1)

        g_intersect_d = filtered

        print len(g_intersect_d), "relationships in the corpus which are in the KB"
        if len(g_intersect_d) > 0:
            # dump G intersected with D to file
            f = open(rel_type + "_g_intersection_d.pkl", "wb")
            cPickle.dump(g_intersect_d, f)
            f.close()

        print "Extra filtering: from the G' not in D, select only those based on keywords"
        filtered = set()
        for r in g_minus_d:
            unigrams_bet = r.between
            unigrams_bef = r.before
            unigrams_aft = r.after
            bigrams_bet = extract_bigrams(' '.join(r.between))
            if any(x in rel_words_unigrams for x in unigrams_bet):
                filtered.add(r)
                continue
            if any(x in rel_words_unigrams for x in unigrams_bef):
                filtered.add(r)
                continue
            if any(x in rel_words_unigrams for x in unigrams_aft):
                filtered.add(r)
                continue
            elif any(x in rel_words_bigrams for x in bigrams_bet):
                filtered.add(r)
                continue
        g_minus_d = filtered
        print len(g_minus_d), "relationships in the corpus not in the KB"
        if len(g_minus_d) > 0:
            # dump G - D to file, relationships in the corpus not in KB
            f = open(rel_type + "_g_minus_d.pkl", "wb")
            cPickle.dump(g_minus_d, f)
            f.close()

    # having B and G_intersect_D => |c| = |G_intersect_D| - |b|
    c = g_intersect_d.difference(set(b))
    assert len(g_minus_d) > 0
    return c, g_minus_d


def calculate_d(g_minus_d, a, e1_type, e2_type, index, rel_type, rel_words_unigrams, rel_words_bigrams):
    # contains facts described in the corpus that are not in the system output nor in the database
    #
    # by applying the PMI of the facts not in the database (i.e., G' \in D)
    # we determine |G \ D|, then we can estimate |d| = |G \ D| - |a|
    #
    # |G' \ D|
    # determine facts not in the database, with high PMI, that is, facts that are true and are not in the database

    # check if it was already calculated and stored in disk
    if os.path.isfile(rel_type + "_high_pmi_not_in_database.pkl"):
        f = open(rel_type + "_high_pmi_not_in_database.pkl")
        print "\nLoading high PMI facts not in the database", rel_type + "_high_pmi_not_in_database.pkl"
        g_minus_d = cPickle.load(f)
        f.close()
    else:
        m = multiprocessing.Manager()
        queue = m.Queue()
        results = [m.list() for _ in range(num_cpus)]

        for r in g_minus_d:
            queue.put(r)

        # calculate PMI for r not in database
        processes = [multiprocessing.Process(target=proximity_pmi_rel_word, args=(e1_type, e2_type, queue, index,
                                                                                  results[i], rel_words_unigrams,
                                                                                  rel_words_bigrams))
                     for i in range(num_cpus)]
        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        g_minus_d = set()
        for l in results:
            g_minus_d.update(l)

        print "High PMI facts not in the database", len(g_minus_d)

        # dump high PMI facts not in the database
        if len(g_minus_d) > 0:
            f = open(rel_type + "_high_pmi_not_in_database.pkl", "wb")
            print "Dumping high PMI facts not in the database to", rel_type + "_high_pmi_not_in_database.pkl"
            cPickle.dump(g_minus_d, f)
            f.close()

    return g_minus_d.difference(a)


########################################################################
# Paralelized functions: each function will run as a different process #
########################################################################
def proximity_pmi_rel_word(e1_type, e2_type, queue, index, results, rel_words_unigrams, rel_words_bigrams):
    idx = open_dir(index)
    count = 0
    distance = MAX_TOKENS_AWAY
    with idx.searcher() as searcher:
        while True:
            try:
                r = queue.get_nowait()
                if count % 50 == 0:
                    print multiprocessing.current_process(), "In Queue", queue.qsize(), "Total Matched: ", len(results)

                #TODO: fazer uma cache

                t1 = query.Term('sentence', "<" + e1_type + ">" + r.e1 + "</" + e1_type + ">")
                t3 = query.Term('sentence', "<" + e2_type + ">" + r.e2 + "</" + e2_type + ">")

                # Entities proximity query without relational words
                q1 = spans.SpanNear2([t1, t3], slop=distance, ordered=True, mindist=1)
                hits = searcher.search(q1, limit=q_limit)

                # Entities proximity considering relational words
                # From the results above count how many contain a relational word
                hits_with_r = 0
                total_hits = 0

                for s in hits:
                    sentence = s.get("sentence")
                    s = SentenceEvaluation(sentence, e1_type, e2_type, MAX_TOKENS_AWAY, MIN_TOKENS_AWAY, CONTEXT_WINDOW, stopwords)
                    for s_r in s.relationships:
                        if r.e1.decode("utf8") == s_r.e1 and r.e2.decode("utf8") == s_r.e2:
                            total_hits += 1
                            unigrams_rel_words = s_r.between
                            bigrams_rel_words = extract_bigrams(' '.join(s_r.between))
                            if any(x in rel_words_unigrams for x in unigrams_rel_words):
                                hits_with_r += 1
                                continue

                            if any(x in rel_words_bigrams for x in bigrams_rel_words):
                                hits_with_r += 1
                                continue

                assert total_hits >= hits_with_r

                if total_hits > 0:
                    pmi = float(hits_with_r) / float(total_hits)
                    if pmi >= PMI:
                        results.append(r)
                count += 1
            except Queue.Empty:
                break


def proximity_pmi_a(e1_type, e2_type, queue, index, results, not_found, rel_words_unigrams, rel_words_bigrams):
    idx = open_dir(index)
    count = 0
    # cache to store already evaluted triples
    cache_correct = set()
    cache_incorrect = set()

    with idx.searcher() as searcher:
        while True:
            try:
                r = queue.get_nowait()
                t = Triple(r.e1, r.e2, r.bet_words)

                count += 1
                if count % 50 == 0:
                    sys.stdout.write(str(multiprocessing.current_process())+" To Process: "+str(queue.qsize()) +
                                     " Correct found: "+str(len(results))+'\n')
                    sys.stdout.flush()

                if t in cache_correct:
                    results.append(r)
                    continue

                if t in cache_incorrect:
                    not_found.append(r)
                    continue

                # relational phrase/word of the relationship/triple
                fact_bet_words_tokens = word_tokenize(r.bet_words)

                entity1 = "<" + e1_type + ">" + r.e1 + "</" + e1_type + ">"
                entity2 = "<" + e2_type + ">" + r.e2 + "</" + e2_type + ">"
                t1 = query.Term('sentence', entity1)
                t3 = query.Term('sentence', entity2)

                # First count the proximity (MAX_TOKENS_AWAY) occurrences of entities r.e1 and r.e2
                q1 = spans.SpanNear2([t1, t3], slop=MAX_TOKENS_AWAY, ordered=True, mindist=1)
                hits = searcher.search(q1, limit=q_limit)

                # Entities proximity considering relational word
                # From the results above count how many contain the relational word/phrase
                hits_with_r = 0

                for s in hits:
                    sentence = s.get("sentence")
                    start_e1 = sentence.rindex(entity1)
                    end_e1 = start_e1+len(entity1)
                    start_e2 = sentence.rindex(entity2)
                    bet = sentence[end_e1:start_e2]
                    bet_tokens = word_tokenize(bet)

                    if not (MIN_TOKENS_AWAY <= len(bet_tokens) <= MAX_TOKENS_AWAY):
                        continue

                    if fact_bet_words_tokens == bet_tokens:
                        hits_with_r += 1

                assert len(hits) >= hits_with_r

                if len(hits) > 0:
                    pmi = float(hits_with_r) / float(len(hits))
                    if pmi >= PMI:
                        results.append(r)
                        cache_correct.add(t)
                        """
                        print "**VALID**:", entity1, '\t', entity2
                        print "hits_without_r ", float(hits_without_r)
                        print "hits_with_r ", float(hits_with_r)
                        print "discarded", discarded
                        print "Index hits", len(hits)
                        print "PMI", pmi
                        print r.sentence
                        print r.bet_words
                        print
                        """

                    else:
                        # check against a list
                        if r.bet_words.strip() in rel_words_unigrams:
                            results.append(r)
                            cache_correct.add(t)

                        elif r.bet_words.strip() in rel_words_bigrams:
                            results.append(r)
                            cache_correct.add(t)

                        else:
                            not_found.append(r)
                            cache_incorrect.add(t)
                            """
                            print "**INVALID**:"
                            print 'ExtractedFact:', entity1, '\t', entity2
                            print r.bet_words.strip()
                            print
                            """
                else:
                    not_found.append(r)
                    cache_incorrect.add(t)

            except Queue.Empty:
                break


def find_intersection(results, no_matches, database, queue):
    count = 0
    while True:
        try:
            r = queue.get_nowait()
            count += 1
            if count % 50 == 0:
                sys.stdout.write(str(multiprocessing.current_process())+" To Process: "+str(queue.qsize()) +
                                 " Correct found: "+str(len(results))+'\n')
                sys.stdout.flush()

            if r.e1 in database.keys():
                if r.e2 in database[r.e1]:
                    results.append(r)
                else:
                    no_matches.append(r)
            else:
                no_matches.append(r)

        except Queue.Empty:
            break


def process_corpus(queue, g_dash, e1_type, e2_type):
    count = 0
    added = 0
    while True:
        try:
            if count % 25000 == 0:
                print multiprocessing.current_process(), "In Queue", queue.qsize(), "Total added: ", added
            line = queue.get_nowait()
            s = SentenceEvaluation(line.strip(), e1_type, e2_type, MAX_TOKENS_AWAY, MIN_TOKENS_AWAY, CONTEXT_WINDOW)
            for r in s.relationships:
                g_dash.append(r)
                added += 1
            count += 1
        except Queue.Empty:
            break


def main():
    # "Automatic Evaluation of Relation Extraction Systems on Large-scale"
    # https://akbcwekex2012.files.wordpress.com/2012/05/8_paper.pdf
    #
    # S  - system output
    # D  - database (freebase)
    # G  - will be the resulting ground truth
    # G' - superset, contains true facts, and wrong facts
    # a  - contains correct facts from the system output
    #
    # b  - intersection between the system output and the database (i.e., freebase),
    #      it is assumed that every fact in this region is correct
    # c  - contains the database facts described in the corpus but not extracted by the system
    # d  - contains the facts described in the corpus that are not in the system output nor in the database
    #
    # Precision = |a|+|b| / |S|
    # Recall    = |a|+|b| / |a| + |b| + |c| + |d|
    # F1        = 2*P*R / P+R

    if len(sys.argv) == 1:
        print "No arguments"
        print "Use: evaluation.py system_output rel_type #CPUs"
        print "\n"
        sys.exit(0)

    rel_type = sys.argv[2]

    global num_cpus
    num_cpus = int(sys.argv[3])

    # load relationships extracted by the system
    system_output = process_output(sys.argv[1])
    print "System output relationships   :", len(system_output)

    # corpus from which the system extracted relationships
    #corpus = "/home/dsbatista/gigaword/set_a_matched.txt"
    corpus = "/home/dsbatista/gigaword/set_b_matched.txt"

    # index to be used to estimate proximity PMI
    index = "/home/dsbatista/gigaword/automatic-evaluation/index_full"

    # directory with files containing relationships from KB
    base_dir = "/home/dsbatista/gigaword/automatic-evaluation/relationships_gold/"

    freebase_ground_truth = []
    dbpedia_ground_truth = []
    yago_ground_truth = []
    manually_added = []

    #TODO: inserir na BD de relações uma versao da relação sem _Coroporation, _Inc., etc.

    if rel_type == 'has-installations':
        e1_type = "ORG"
        e2_type = "LOC"
        rel_words_unigrams = installations_in_unigrams
        rel_words_bigrams = installations_in_bigrams
        freebase_ground_truth = [base_dir+"freebase_place_founded.txt"]
        dbpedia_ground_truth = [base_dir+"dbpedia_location.txt", base_dir+"dbpedia_headquarter.txt",
                                base_dir+"dbpedia_locationCity.txt", base_dir+"dbpedia_locationCountry.txt"]
        yago_ground_truth = [base_dir+"yago_isLocatedIn.txt"]
        manually_added = [base_dir+"manually_has-installations.txt"]

    elif rel_type == 'has-installations2':
        e1_type = "LOC"
        e2_type = "ORG"
        rel_words_unigrams = installations_in_unigrams
        rel_words_bigrams = installations_in_bigrams
        freebase_ground_truth = [base_dir+"freebase_place_founded.txt"]
        dbpedia_ground_truth = [base_dir+"dbpedia_location.txt", base_dir+"dbpedia_headquarter.txt",
                                base_dir+"dbpedia_locationCity.txt", base_dir+"dbpedia_locationCountry.txt"]
        yago_ground_truth = [base_dir+"yago_isLocatedIn.txt"]
        manually_added = [base_dir+"manually_has-installations2.txt"]

    elif rel_type == 'founder':
        e1_type = "ORG"
        e2_type = "PER"
        rel_words_unigrams = founder_unigrams
        rel_words_bigrams = founder_unigrams
        freebase_ground_truth = [base_dir+"freebase_founded.txt"]
        dbpedia_ground_truth = [base_dir+"dbpedia_founder.txt"]
        yago_ground_truth = [base_dir+"yago_created.txt"]
        manually_added = [base_dir+"manually_founded.txt"]

    elif rel_type == 'founder2':
        e1_type = "PER"
        e2_type = "ORG"
        rel_words_unigrams = founder_unigrams
        rel_words_bigrams = founder_unigrams
        freebase_ground_truth = [base_dir+"freebase_founded.txt"]
        dbpedia_ground_truth = [base_dir+"dbpedia_founder.txt"]
        yago_ground_truth = [base_dir+"yago_created.txt"]
        manually_added = [base_dir+"manually_founded2.txt"]

    elif rel_type == 'owns':
        e1_type = "ORG"
        e2_type = "ORG"
        rel_words_unigrams = owns_unigrams
        rel_words_bigrams = owns_unigrams
        freebase_ground_truth = [base_dir+"freebase_acquired.txt"]
        dbpedia_ground_truth = [base_dir+"dbpedia_subsidiary.txt"]
        yago_ground_truth = [base_dir+"yago_owns.txt"]
        manually_added = [base_dir+"manually_owns.txt"]

    elif rel_type == 'owns2':
        e1_type = "ORG"
        e2_type = "ORG"
        rel_words_unigrams = owns_unigrams
        rel_words_bigrams = owns_unigrams
        freebase_ground_truth = [base_dir+"freebase_acquired.txt"]
        dbpedia_ground_truth = [base_dir+"dbpedia_subsidiary.txt"]
        yago_ground_truth = [base_dir+"yago_owns.txt"]
        manually_added = [base_dir+"manually_owns2.txt"]

    elif rel_type == 'affiliation':
        e1_type = "ORG"
        e2_type = "PER"
        rel_words_unigrams = affiliation_unigrams
        rel_words_bigrams = affiliation_bigrams
        freebase_ground_truth = [base_dir+"freebase_employment.txt", base_dir+"freebase_governance.txt",
                                 base_dir+"freebase_leader_of.txt"]
        yago_ground_truth = [base_dir+"yago_isAffiliatedTo.txt", base_dir+"yago_worksAt.txt"]
        manually_added = [base_dir+"manually_affiliation.txt"]

    elif rel_type == 'affiliation2':
        e1_type = "PER"
        e2_type = "ORG"
        rel_words_unigrams = affiliation_unigrams
        rel_words_bigrams = affiliation_bigrams
        freebase_ground_truth = [base_dir+"freebase_employment.txt", base_dir+"freebase_governance.txt",
                                 base_dir+"freebase_leader_of.txt"]
        yago_ground_truth = [base_dir+"yago_isAffiliatedTo.txt", base_dir+"yago_worksAt.txt"]
        manually_added = [base_dir+"manually_affiliation2.txt"]

    elif rel_type == 'studied':
        e1_type = "PER"
        e2_type = "ORG"
        rel_words_unigrams = studied_unigrams
        rel_words_bigrams = studied_bigrams
        freebase_ground_truth = []
        dbpedia_ground_truth = [base_dir+"dbpedia_almaMater.txt"]
        yago_ground_truth = [base_dir+"yago_graduatedFrom.txt"]
        manually_added = [base_dir+"manually_studied.txt"]

    elif rel_type == 'studied2':
        e1_type = "ORG"
        e2_type = "PER"
        rel_words_unigrams = studied2_unigrams
        rel_words_bigrams = studied2_bigrams
        freebase_ground_truth = []
        dbpedia_ground_truth = [base_dir+"dbpedia_almaMater.txt"]
        yago_ground_truth = [base_dir+"yago_graduatedFrom.txt"]
        manually_added = [base_dir+"manually_studied2.txt"]

    elif rel_type == 'located-in':
        e1_type = "LOC"
        e2_type = "LOC"
        rel_words_unigrams = located_in_unigrams
        rel_words_bigrams = located_in_bigrams
        freebase_ground_truth = [base_dir+"freebase_location_citytown.txt"]
        dbpedia_ground_truth = [base_dir+"dbpedia_country.txt", base_dir+"dbpedia_capital.txt",
                                base_dir+"dbpedia_largestCity.txt"]

        yago_ground_truth = [base_dir+"yago_isLocatedIn.txt"]

    elif rel_type == 'spouse':
        e1_type = "PER"
        e2_type = "PER"
        rel_words_unigrams = spouse_unigrams
        rel_words_bigrams = spouse_bigrams
        dbpedia_ground_truth = []
        yago_ground_truth = []
        freebase_ground_truth = [base_dir+"freebase_married_to.txt", base_dir+"freebase_spouse_partner.txt"]
        manually_added = [base_dir+"manually_spouse.txt"]

    else:
        print "Invalid relationship type", rel_type
        sys.exit(0)

    print "\nRelationship Type:", rel_type
    print "Arg1 Type:", e1_type
    print "Arg2 Type:", e2_type

    # load relationships into database
    database = defaultdict(set)

    print "\nLoading relationships from Freebase"
    for f in freebase_ground_truth:
        print f.split('/')[-1],
        process_freebase(freebase_ground_truth, database, rel_type)
        print

    print "\nLoading relationships from DBpedia"
    if dbpedia_ground_truth is not None:
        for f in dbpedia_ground_truth:
            print f.split('/')[-1],
            process_dbpedia(dbpedia_ground_truth, database, f.split('/')[-1], rel_type)
            print

    print "\nLoading relationships from Yago"
    for f in yago_ground_truth:
        print f.split('/')[-1],
        process_yago(f, database, rel_type)
        print

    print "\nLoading manually added relationships"
    for f in manually_added:
        print f.split('/')[-1],
        process_manually(f, database)
        print

    print
    print len(database), "relationships loaded"

    print "\nCalculating set B: intersection between system output and database"
    b, not_in_database = calculate_b_parallel(system_output, database)

    print "System output      :", len(system_output)
    print "Found in database  :", len(b)
    print "Not found          :", len(not_in_database)
    assert len(system_output) == len(not_in_database) + len(b)

    print "\nCalculating set A: correct facts from system output not in the database (proximity PMI)"
    a, not_found = calculate_a(not_in_database, e1_type, e2_type, index, rel_words_unigrams, rel_words_bigrams)
    print "System output      :", len(system_output)
    print "Found in database  :", len(b)
    print "Correct in corpus  :", len(a)
    print "Not found          :", len(not_found)
    assert len(system_output) == len(a) + len(b) + len(not_found)

    # Estimate G \intersected D = |b| + |c|, looking for relationships in G' that match a relationship in D
    # once we have G \in D and |b|, |c| can be derived by: |c| = |G \in D| - |b|
    #  G' = superset of G, cartesian product of all possible entities and relations (i.e., G' = E x R x E)
    print "\nCalculating set C: database facts in the corpus but not extracted by the system"
    c, g_minus_d = calculate_c(corpus, database, b, e1_type, e2_type, rel_type, rel_words_unigrams, rel_words_bigrams)
    assert len(c) > 0

    uniq_c = set()
    for r in c:
        uniq_c.add((r.e1, r.e2))

    # By applying the PMI of the facts not in the database (i.e., G' \in D)
    # we determine |G \ D|, then we can estimate |d| = |G \ D| - |a|
    print "\nCalculating set D: facts described in the corpus not in the system output nor in the database"
    d = calculate_d(g_minus_d, a, e1_type, e2_type, index, rel_type, rel_words_unigrams, rel_words_bigrams)

    print "System output      :", len(system_output)
    print "Found in database  :", len(b)
    print "Correct in corpus  :", len(a)
    print "Not found          :", len(not_found)
    print "\n"

    assert len(d) > 0

    uniq_d = set()
    for r in d:
        uniq_d.add((r.e1, r.e2))

    print "|a| =", len(a)
    print "|b| =", len(b)
    print "|c| =", len(c), "(", len(uniq_c), ")"
    print "|d| =", len(d), "(", len(uniq_d), ")"
    print "|S| =", len(system_output)
    print "|G| =", len(set(a).union(set(b).union(set(c).union(set(d)))))
    print "Relationships not found:", len(set(not_found))

    # Write relationships not found in the Database nor with high PMI relatation words to disk
    f = open(rel_type + "_wrong.txt", "w")
    for r in sorted(set(not_found), reverse=True):
        f.write('instance : ' + r.e1 + '\t' + r.e2 + '\n')
        f.write('sentence : ' + r.sentence + '\n')
        f.write('bet_words: ' + r.bet_words + '\n')
        f.write('\n')
    f.close()

    a = set(a)
    b = set(b)
    output = set(system_output)
    if len(output) == 0:
        print "\nPrecision   : 0.0"
        print "Recall      : 0.0"
        print "F1          : 0.0"
        print "\n"
    elif float(len(a) + len(b)) == 0:
        print "\nPrecision   : 0.0"
        print "Recall      : 0.0"
        print "F1          : 0.0"
        print "\n"
    else:
        precision = float(len(a) + len(b)) / float(len(output))
        recall = float(len(a) + len(b)) / float(len(a) + len(b) + len(uniq_c) + len(uniq_d))
        f1 = 2 * (precision * recall) / (precision + recall)
        print "\nPrecision   : ", precision
        print "Recall      : ", recall
        print "F1          : ", f1
        print "\n"

if __name__ == "__main__":
    main()