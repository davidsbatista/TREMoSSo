#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import codecs
import re
import fileinput
import functools
import multiprocessing
import time
import sys
import os
import cPickle
import Queue

from whoosh.index import open_dir, os
from whoosh.query import spans
from whoosh import query
from nltk import word_tokenize, bigrams
from nltk.corpus import stopwords
from collections import defaultdict
from Sentence import Sentence

num_cpus = multiprocessing.cpu_count()
#num_cpus = 1

# relational words to be used in calculating the set C and D aplpying the  with the proximity PMI

founder_unigrams = ['founder', 'co-founder', 'cofounder', 'co-founded', 'cofounded', 'founded', 'founders']
founder_bigrams = ['started by']

acquired_unigrams = ['owns', 'acquired', 'bought', 'acquisition']
acquired_bigrams = []

installations_in_unigrams = ['headquarters', 'headquartered', 'offices', 'office', 'building', 'buildings', 'factory',
                             'plant', 'compund']

installations_in_bigrams = ['based in', 'located in', 'main office', ' main offices', 'offices in', 'building in',
                            'office in', 'branch in', 'store in', 'firm in', 'factory in', 'plant in', 'head office',
                            'head offices', 'in central', 'in downton', 'outskirts of', 'suburs of']

#TODO: melhorar esta lista, incluir mais profissões?
employment_unigrams = ['chief', 'scientist', 'professor', 'biologist', 'ceo', 'CEO', 'employer']
employment_bigrams = []

# tokens between entities which do net represent relationships
bad_tokens = [",", "(", ")", ";", "''",  "``", "'s", "-", "vs.", "v", "'", ":", ".", "--"]
stopwords = stopwords.words('english')
not_valid = bad_tokens + stopwords

# PMI value for proximity
PMI = 0.7

# Parameters for relationship extraction from Sentence
MAX_TOKENS_AWAY = 6
MIN_TOKENS_AWAY = 1
CONTEXT_WINDOW = 2

# DEBUG stuff
PRINT_NOT_FOUND = False

# stores all variations matched with database
manager = multiprocessing.Manager()
all_in_database = manager.dict()


class ExtractedFact(object):
    def __init__(self, _e1, _e2, _score, _bef, _bet, _aft, _sentence, _passive_voice):
        self.ent1 = _e1
        self.ent2 = _e2
        self.score = _score
        self.bef_words = _bef
        self.bet_words = _bet
        self.aft_words = _aft
        self.sentence = _sentence
        self.passive_voice = _passive_voice

    def __cmp__(self, other):
            if other.score > self.score:
                return -1
            elif other.score < self.score:
                return 1
            else:
                return 0

    def __hash__(self):
        sig = hash(self.ent1) ^ hash(self.ent2) ^ hash(self.bef_words) ^ hash(self.bet_words) ^ hash(self.aft_words) ^ \
            hash(self.score) ^ hash(self.sentence)
        return sig

    def __eq__(self, other):
        if self.ent1 == other.ent1 and self.ent2 == other.ent2 and self.score == other.score and self.bef_words == \
                other.bef_words and self.bet_words == other.bet_words and self.aft_words == other.aft_words \
                and self.sentence == other.sentence:
            return True
        else:
            return False


# ###########################################
# Misc., Utils, parsing corpus into memory #
# ###########################################

def timecall(f):
    @functools.wraps(f)
    def wrapper(*args, **kw):
        start = time.time()
        result = f(*args, **kw)
        end = time.time()
        # print "%s %.2f seconds" % (f.__name__, end - start)
        print "Time taken: %.2f seconds" % (end - start)
        return result

    return wrapper


def process_corpus(queue, g_dash, e1_type, e2_type):
    count = 0
    added = 0
    while True:
        try:
            if count % 25000 == 0:
                print multiprocessing.current_process(), "In Queue", queue.qsize(), "Total added: ", added
            line = queue.get_nowait()
            line = re.sub(r'>([^<]+</[A-Z]+>)', ">", line)
            s = Sentence(line.strip(), e1_type, e2_type, MAX_TOKENS_AWAY, MIN_TOKENS_AWAY, CONTEXT_WINDOW, stopwords)
            for r in s.relationships:
                """
                print r.sentence
                print r.before
                print r.between
                print r.after
                print r.ent1
                print r.ent2
                print r.arg1type
                print r.arg2type
                """

                tokens = word_tokenize(r.between)
                if all(x in not_valid for x in word_tokenize(r.between)):
                    continue
                elif "," in tokens and tokens[0] != ',':
                    continue
                else:
                    g_dash.append(r)
                    added += 1
            count += 1
        except Queue.Empty:
            break


def process_output(data, threshold, rel_type):
    """
    parses the file with the relationships extracted by the system
    each relationship is transformed into a ExtracteFact class
    """
    system_output = list()
    for line in fileinput.input(data):
        if line.startswith('instance'):
            instance_parts, score = line.split("score:")
            e1, e2 = instance_parts.split("instance:")[1].strip().split('\t')

        if line.startswith('sentence'):
            sentence = line.split("sentence:")[1].strip()

        if line.startswith('pattern_bef:'):
            bef = line.split("pattern_bef:")[1].strip()

        if line.startswith('pattern_bet:'):
            bet = line.split("pattern_bet:")[1].strip()

        if line.startswith('pattern_aft:'):
            aft = line.split("pattern_aft:")[1].strip()

        if line.startswith('passive voice:'):
            tmp = line.split("passive voice:")[1].strip()
            if tmp == 'False':
                passive_voice = False
            elif tmp == 'True':
                passive_voice = True

        if line.startswith('\n') and float(score) >= threshold:
            if 'bef' not in locals():
                bef = ''
            if 'aft' not in locals():
                aft = ''
            if passive_voice is True and rel_type in ['acquired', 'installations-in']:
                r = ExtractedFact(e2, e1, float(score), bef, bet, aft, sentence, passive_voice)
            else:
                r = ExtractedFact(e1, e2, float(score), bef, bet, aft, sentence, passive_voice)

            if ("'s parent" in bet or 'subsidiary of' in bet or bet == 'subsidiary') and rel_type == 'acquired':
                r = ExtractedFact(e2, e1, float(score), bef, bet, aft, sentence, passive_voice)
            system_output.append(r)

    fileinput.close()
    return system_output


def process_dbpedia(data, database, file_rel_type):
    for line in fileinput.input(data):
        try:
            e1, rel, e2, p = line.strip().split('>')
            e1 = e1.replace("<http://dbpedia.org/resource/", "http://en.wikipedia.org/wiki/")
            e2 = e2.replace("<http://dbpedia.org/resource/", "http://en.wikipedia.org/wiki/")

        except ValueError:
            print "Error parsing", line
            sys.exit(0)

        # follow the order of the relationships as in the output
        if file_rel_type in ['dbpedia_capital.txt', 'dbpedia_largestCity.txt']:
            database[e2.strip().decode("utf8")].append(e1.strip().decode("utf8"))
        else:
            database[e1.strip().decode("utf8")].append(e2.strip().decode("utf8"))
    fileinput.close()


def process_yago(data, database, rel_type):
    for line in fileinput.input(data):
        try:
            e1, rel, e2 = line.strip().split('\t')
            e1 = e1.replace("<", "").replace(">", "")
            e2 = e2.replace("<", "").replace(">", "").strip().replace(".", "")
        except ValueError:
            print "Error parsing", line
            sys.exit(0)

        # to have the same URL links has from the output and DBpedia
        e1 = "http://en.wikipedia.org/wiki/"+e1
        e2 = "http://en.wikipedia.org/wiki/"+e2

        # follow the order of the relationships as in the output
        if rel_type in ['founder', 'affiliation']:
            database[e2.strip().decode("utf8")].append(e1.strip().decode("utf8"))
        else:
            database[e1.strip().decode("utf8")].append(e2.strip().decode("utf8"))
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

    # Estimate G \in D, look for facts in G' that a match a fact in the database
    # check if already exists for this particular relationship
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
        for r in g_dash_set:
            queue.put(r)

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
            unigrams_bet = word_tokenize(r.between)
            unigrams_bef = word_tokenize(r.before)
            unigrams_aft = word_tokenize(r.after)
            bigrams_bet = extract_bigrams(r.between)
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
            unigrams_bet = word_tokenize(r.between)
            unigrams_bef = word_tokenize(r.before)
            unigrams_aft = word_tokenize(r.after)
            bigrams_bet = extract_bigrams(r.between)
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
    q_limit = 500
    with idx.searcher() as searcher:
        while True:
            try:
                r = queue.get_nowait()
                """
                if count % 50 == 0:
                    print multiprocessing.current_process(), "In Queue", queue.qsize(), "Total Matched: ", len(results)
                """

                t1 = query.Term('sentence', "<"+e1_type+" url="+r.ent1+">")
                t3 = query.Term('sentence', "<"+e2_type+" url="+r.ent2+">")

                # Entities proximity query without relational words
                q1 = spans.SpanNear2([t1, t3], slop=distance, ordered=True, mindist=1)
                hits = searcher.search(q1, limit=q_limit)

                # Entities proximity considering relational words
                # From the results above count how many contain a relational word

                hits_with_r = 0
                hits_without_r = 0

                for s in hits:
                    sentence = s.get("sentence")
                    s = Sentence(sentence, e1_type, e2_type, MAX_TOKENS_AWAY, MIN_TOKENS_AWAY, CONTEXT_WINDOW,
                                 stopwords)
                    for s_r in s.relationships:
                        if r.ent1.decode("utf8") == s_r.ent1 and r.ent2.decode("utf8") == s_r.ent2:
                            unigrams_rel_words = word_tokenize(s_r.between)
                            bigrams_rel_words = extract_bigrams(s_r.between)
                            if all(x in not_valid for x in unigrams_rel_words):
                                hits_without_r += 1
                                continue
                            elif any(x in rel_words_unigrams for x in unigrams_rel_words):
                                """
                                print "UNIGRAMS HIT"
                                print s_r.sentence
                                print s_r.ent1
                                print s_r.ent2
                                print s_r.between
                                print "\n"
                                """
                                hits_with_r += 1
                            elif any(x in rel_words_bigrams for x in bigrams_rel_words):
                                """
                                print "BIGRAMS HIT"
                                print s_r.sentence
                                print s_r.ent1
                                print s_r.ent2
                                print s_r.between
                                print "\n"
                                """
                                hits_with_r += 1
                            else:
                                hits_without_r += 1

                if hits_with_r > 0 and hits_without_r > 0:
                    pmi = float(hits_with_r) / float(hits_without_r)
                    if pmi >= PMI:
                        if word_tokenize(s_r.between)[-1] == 'by':
                            tmp = s_r.ent2
                            s_r.ent2 = s_r.ent1
                            s_r.ent1 = tmp
                        results.append(r)
                        """
                        print "**ADDED**:", entity1, '\t', entity2
                        print "hits_without_r ", float(hits_without_r)
                        print "hits_with_r ", float(hits_with_r)
                        print "PMI", pmi
                        print
                        """
                count += 1
            except Queue.Empty:
                break


def proximity_pmi_a(e1_type, e2_type, queue, index, results, not_found, rel_words_unigrams, rel_words_bigrams):
    idx = open_dir(index)
    count = 0
    q_limit = 500
    with idx.searcher() as searcher:
        while True:
            try:
                r = queue.get_nowait()
                count += 1
                if count % 50 == 0:
                    print multiprocessing.current_process(), "To Process", queue.qsize(), "Correct found:", len(results)

                # if its not in the database calculate the PMI
                entity1 = "<"+e1_type+" url="+r.ent1+">"
                entity2 = "<"+e2_type+" url="+r.ent2+">"
                t1 = query.Term('sentence', entity1)
                t3 = query.Term('sentence', entity2)

                # First count the proximity (MAX_TOKENS_AWAY) occurrences of entities r.e1 and r.e2
                q1 = spans.SpanNear2([t1, t3], slop=MAX_TOKENS_AWAY, ordered=True, mindist=1)
                hits = searcher.search(q1, limit=q_limit)

                # Entities proximity considering relational words
                # From the results above count how many contain a relational word
                #print entity1, '\t', entity2, len(hits), "\n"

                hits_with_r = 0
                hits_without_r = 0
                fact_bet_words_tokens = word_tokenize(r.bet_words)

                for s in hits:
                    sentence = s.get("sentence")
                    start_e1 = sentence.rindex(entity1)
                    end_e1 = start_e1+len(entity1)
                    start_e2 = sentence.rindex(entity2)
                    end_e2 = start_e2+len(entity2)
                    bef = sentence[:start_e1]
                    aft = sentence[end_e2:]
                    bet = sentence[end_e1:start_e2]
                    bef_tokens = word_tokenize(bef)
                    bet_tokens = word_tokenize(bet)
                    aft_tokens = word_tokenize(aft)

                    if not (MIN_TOKENS_AWAY >= len(bet_tokens) <= MAX_TOKENS_AWAY):
                        continue
                    else:
                        bef_tokens = bef_tokens[CONTEXT_WINDOW:]
                        aft_tokens = aft_tokens[:CONTEXT_WINDOW]

                    unigrams_bef_words = bef_tokens
                    unigrams_bet_words = bet_tokens
                    unigrams_aft_words = aft_tokens
                    bigrams_rel_words = extract_bigrams(bet)

                    if fact_bet_words_tokens == unigrams_bet_words:
                        #print "****HIT**** 2"
                        hits_with_r += 1

                    elif any(x in rel_words_unigrams for x in unigrams_bef_words):
                        #print "****HIT**** 4"
                        hits_with_r += 1

                    elif any(x in rel_words_unigrams for x in unigrams_bet_words):
                        #print "****HIT**** 5"
                        hits_with_r += 1

                    elif any(x in rel_words_unigrams for x in unigrams_aft_words):
                        #print "****HIT**** 6"
                        hits_with_r += 1

                    elif rel_words_bigrams == bigrams_rel_words:
                        #print "****HIT**** 7"
                        hits_with_r += 1
                    else:
                        hits_without_r += 1

                if hits_with_r > 0 and hits_without_r > 0:
                    pmi = float(hits_with_r) / float(hits_without_r)
                    if pmi >= PMI:
                        results.append(r)
                        """
                        print "**VALID**:", entity1, '\t', entity2
                        print "hits_without_r ", float(hits_without_r)
                        print "hits_with_r ", float(hits_with_r)
                        print "PMI", pmi
                        print r.sentence
                        print r.bet_words
                        print
                        """
                    else:
                        not_found.append(r)
                        #TODO: confirmar os invalids
                        """
                        print "**INVALID**:"
                        print 'ExtractedFact:', entity1, '\t', entity2
                        print r.sentence
                        fact_bef_words_tokens = word_tokenize(r.bef_words)
                        fact_bet_words_tokens = word_tokenize(r.bet_words)
                        fact_aft_words_tokens = word_tokenize(r.aft_words)
                        print "BEF", r.bef_words, fact_bef_words_tokens
                        print "BET", r.bet_words, fact_bet_words_tokens
                        print "AFT", r.aft_words, fact_aft_words_tokens
                        print "hits_without_r ", float(hits_without_r)
                        print "hits_with_r ", float(hits_with_r)
                        print "PMI", pmi
                        print "Index hits", len(hits)
                        print
                        """
                else:
                    not_found.append(r)
                count += 1

            except Queue.Empty:
                break


def find_intersection(results, no_matches, database, queue):
    while True:
        try:
            r = queue.get_nowait()
            if r.ent1 in database.keys():
                if r.ent2 in database[r.ent1]:
                    results.append(r)
                else:
                    no_matches.append(r)
            else:
                no_matches.append(r)

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
        print "Use: evaluation.py threshold system_output rel_type database"
        print "\n"
        sys.exit(0)

    threhsold = float(sys.argv[1])
    rel_type = sys.argv[3]

    # load relationships extracted by the system
    system_output = process_output(sys.argv[2], threhsold, rel_type)
    print "Relationships score threshold :", threhsold
    print "System output relationships   :", len(system_output)

    # corpus from which the system extracted relationships
    corpus = "/home/dsbatista/gigaword/AFP-AIDA-Linked/documents/all_sentences_6_months.txt"

    # index to be used to estimate proximity PMI
    index = "/home/dsbatista/gigaword/AFP-AIDA-Linked/evaluation/index_full"

    """
    affiliation                           & PER & ORG
        DBpedia:        <http://dbpedia.org/ontology/affiliation>
        YagoFacts.ttl:  <isAffiliatedTo>


    founded-by                            & ORG & PER
        DBpedia:        <http://dbpedia.org/ontology/founder>
        YagoFacts.ttl:  <created>


    installations-in                      & ORG & LOC
        DBpedia:        <http://dbpedia.org/ontology/location>
                        <http://dbpedia.org/ontology/headquarter>
                        <http://dbpedia.org/ontology/locationCity>
                        <http://dbpedia.org/ontology/locationCountry>

                        testar com esta: <http://dbpedia.org/ontology/locatedInArea> ?

        YagoFacts.ttl:  <isLocatedIn>

    has-shares-of                         & ORG & ORG
        DBpedia:        <http://dbpedia.org/ontology/subsidiary>
        YagoFacts.ttl:  <owns>

    located-in                            & LOC & LOC
        DBpedia:        <http://dbpedia.org/ontology/locatedInArea> # has organisations and locations
                        <http://dbpedia.org/ontology/country>
                        <http://dbpedia.org/ontology/capital>       # swap
                        <http://dbpedia.org/ontology/largestCity>   # swap

                        #TODO:
                        <http://dbpedia.org/ontology/municipality>
                        <http://dbpedia.org/ontology/archipelago>
                        <http://dbpedia.org/ontology/subregion>
                        <http://dbpedia.org/ontology/federalState>
                        <http://dbpedia.org/ontology/district>
                        <http://dbpedia.org/ontology/region>
                        <http://dbpedia.org/ontology/province>
                        <http://dbpedia.org/ontology/state>
                        <http://dbpedia.org/ontology/county>
                        <http://dbpedia.org/ontology/map>
                        <http://dbpedia.org/ontology/campus>
                        <http://dbpedia.org/ontology/garrison>
                        <http://dbpedia.org/ontology/department>

                        #<http://dbpedia.org/ontology/capitalCountry>
                        #<http://dbpedia.org/ontology/city>
                        #<http://dbpedia.org/ontology/location>

        YagoFacts.ttl:  <isLocatedIn>

    study-at                              & PER & ORG
        DBpedia:         <http://dbpedia.org/ontology/almaMater>
        YagoFacts.ttl:   <graduatedFrom>

    agrees-with                           & PER & PER
    disagrees-with                        & PER & PER
    """

    base_dir = "/home/dsbatista/gigaword/AFP-AIDA-Linked/evaluation/ground-truth/"

    if rel_type == 'installations-in':
        e1_type = "ORG"
        e2_type = "LOC"
        rel_words_unigrams = installations_in_unigrams
        rel_words_bigrams = installations_in_bigrams
        dbpedia_ground_truth = [base_dir+"dbpedia_location.txt", base_dir+"dbpedia_headquarter.txt",
                                base_dir+"dbpedia_locationCity.txt", base_dir+"dbpedia_locationCountry.txt"]
        yago_ground_truth = [base_dir+"yago_isLocatedIn.txt"]

    elif rel_type == 'studied':
        e1_type = "ORG"
        e2_type = "PER"
        rel_words_unigrams = None
        rel_words_bigrams = None
        dbpedia_ground_truth = [base_dir+"dbpedia_almaMater.txt"]
        yago_ground_truth = [base_dir+"yago_graduatedFrom.txt"]

    elif rel_type == 'founder':
        e1_type = "ORG"
        e2_type = "PER"
        rel_words_unigrams = founder_unigrams
        rel_words_bigrams = founder_unigrams
        dbpedia_ground_truth = [base_dir+"dbpedia_founder.txt"]
        yago_ground_truth = [base_dir+"yago_created.txt"]

    elif rel_type == 'has-shares-of':
        e1_type = "ORG"
        e2_type = "ORG"
        rel_words_unigrams = acquired_unigrams
        rel_words_bigrams = acquired_unigrams
        dbpedia_ground_truth = [base_dir+"dbpedia_subsidiary.txt"]
        yago_ground_truth = [base_dir+"yago_owns.txt"]

    elif rel_type == 'located-in':
        e1_type = "LOC"
        e2_type = "LOC"
        rel_words_unigrams = None
        rel_words_bigrams = None
        dbpedia_ground_truth = [base_dir+"dbpedia_locatedInArea.txt"]
        yago_ground_truth = [base_dir+"yago_isLocatedIn.txt"]

    elif rel_type == 'affiliation':
        e1_type = "ORG"
        e2_type = "PER"
        rel_words_unigrams = employment_unigrams
        rel_words_bigrams = employment_bigrams
        dbpedia_ground_truth = [base_dir+"dbpedia_affiliation.txt"]
        yago_ground_truth = [base_dir+"yago_affiliation.txt", base_dir+"yago_worksAt.txt"]
    else:
        print "Invalid relationship type", rel_type
        sys.exit(0)

    print "\nRelationship Type:", rel_type
    print "Arg1 Type:", e1_type
    print "Arg2 Type:", e2_type

    # load relationships into database
    database = defaultdict(list)
    print "\nLoading relationships from DBpedia"
    for f in dbpedia_ground_truth:
        print f.split('/')[-1],
        process_dbpedia(dbpedia_ground_truth, database, f.split('/')[-1])
        print

    print "\nLoading relationships from Yago"
    for f in yago_ground_truth:
        print f.split('/')[-1],
        process_yago(f, database, rel_type)
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
    print "\n"
    assert len(system_output) == len(a) + len(b) + len(not_found)

    # Estimate G \intersected D = |b| + |c|, looking for relationships in G' that match a relationship in D
    # once we have G \in D and |b|, |c| can be derived by: |c| = |G \in D| - |b|
    #  G' = superset of G, cartesian product of all possible entities and relations (i.e., G' = E x R x E)
    print "\nCalculating set C: database facts in the corpus but not extracted by the system"
    c, g_minus_d = calculate_c(corpus, database, b, e1_type, e2_type, rel_type,
                               rel_words_unigrams, rel_words_bigrams)
    assert len(c) > 0

    uniq_c = set()
    for r in c:
        uniq_c.add((r.ent1, r.ent2))

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
        uniq_d.add((r.ent1, r.ent2))

    print "|a| =", len(a)
    print "|b| =", len(b)
    print "|c| =", len(c), "(", len(uniq_c), ")"
    print "|d| =", len(d), "(", len(uniq_d), ")"
    print "|S| =", len(system_output)
    print "|G| =", len(set(a).union(set(b).union(set(c).union(set(d)))))
    print "Relationships not found:", len(set(not_found))

    # Write relationships not found in the Database nor with high PMI relatation words to disk
    f = open(rel_type + "_negative.txt", "w")
    for r in sorted(set(not_found), reverse=True):
        f.write('instance :' + r.ent1 + '\t' + r.ent2 + '\t' + str(r.score) + '\n')
        f.write('sentence :' + r.sentence + '\n')
        f.write('bef_words:' + r.bef_words + '\n')
        f.write('bet_words:' + r.bet_words + '\n')
        f.write('aft_words:' + r.aft_words + '\n')
        f.write('\n')
    f.close()

    # Write all correct relationships (sentence, entities and score) to file
    f = open(rel_type + "_positive.txt", "w")
    for r in sorted(set(a).union(b), reverse=True):
        f.write('instance :' + r.ent1 + '\t' + r.ent2 + '\t' + str(r.score) + '\n')
        f.write('sentence :' + r.sentence + '\n')
        f.write('bef_words:' + r.bef_words + '\n')
        f.write('bet_words:' + r.bet_words + '\n')
        f.write('aft_words:' + r.aft_words + '\n')
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