#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import os
import sys
import time
import codecs

from whoosh.analysis import RegexTokenizer
from whoosh.fields import Schema, TEXT
from whoosh.index import create_in
from whoosh.query import *


def timecall(f):
    @functools.wraps(f)
    def wrapper(*args, **kw):
        start = time.time()
        result = f(*args, **kw)
        end = time.time()
        print "%s %.2f seconds" % (f.__name__, end - start)
        return result
    return wrapper


def create_index():
    # tokenize using the full url as a unique token, i.e.:
    # ['said', 'authorities', 'in', '<LOC url=http://en.wikipedia.org/wiki/United_Kingdom>', 'must', 'be', 'held',
    # 'to', 'account']
    regex_tokenize = re.compile('<[^>]+>|\w+(?:-\w+)+|\w+', re.U)
    tokenizer = RegexTokenizer(regex_tokenize)
    schema = Schema(sentence=TEXT(stored=True, analyzer=tokenizer))
    if not os.path.exists("index_full"):
        os.mkdir("index_full")
    idx = create_in("index_full", schema)
    return idx


@timecall
def index_sentences(writer):
    count = 0
    f = codecs.open(sys.argv[1], "r", "utf-8")
    for line in f:
        try:
            # remove entity surface string, keep the URL only, e.g.:
            # "...said authorities in <LOC url=http://en.wikipedia.org/wiki/United_Kingdom>London</LOC> must be held..."
            # becomes
            # "...said authorities in <LOC url=http://en.wikipedia.org/wiki/United_Kingdom> must be held to account..."
            line = re.sub(r'>([^<]+</[A-Z]+>)', ">", line)
            writer.add_document(sentence=line.strip())
        except UnicodeDecodeError, e:
            print e
            print l
            sys.exit(0)

        count += 1
        if count % 50000 == 0:
            print count, "lines processed"
    f.close()
    writer.commit()


def main():
    idx = create_index()
    writer = idx.writer(limitmb=2048, procs=5, multisegment=True)
    index_sentences(writer)
    idx.close()


if __name__ == "__main__":
    main()