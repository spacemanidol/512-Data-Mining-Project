import os
import argparse
import numpy as np
from collections import Counter

def load_qrels(filename):
    q2click = {} # query to the documents that are clicks on it
    with open(filename) as f:
        for l in f:
            l = l.strip().split(' ')
            document = l[2]
            query = l[0]
            if query not in q2click:
                q2click[query] = set()
            q2click[query].add(document)
    return q2click

def get_q2click_stats(q2click):
    doc_counts = []
    doc2query = {}
    for q in q2click:
        doc_counts.append(len(q2click[q]))
        for doc in q2click[q]:
            if doc not in doc2query:
                doc2query[doc] = set()
            doc2query[doc].add(q)
    query_counts = []
    for doc in doc2query:
        query_counts.append(len(doc2query[doc]))
    print("There are {} queries and {} documents.".format(len(q2click), len(doc2query)))
    print("The average query has {} documents, the mean is {} and the max is {}, and the min is {}".format(np.average(doc_counts), np.mean(doc_counts), np.max(doc_counts), np.min(doc_counts)))
    print("The average document has {} queries, the mean has {}, the max is {} and the min is {}".format(np.average(query_counts), np.mean(query_counts), np.max(query_counts), np.min(query_counts)))
    return doc2query
def load_queries(data_dir):
    qid2query = {}
    query_files = ['queries.train.tsv', 'queries.eval.tsv', 'queries.dev.tsv', 'orcas-doctrain-queries.tsv']
    for q_file in query_files:
        with open(os.path.join(data_dir, q_file), 'r') as f:
            for l in f:
                l = l.strip().split('\t')
                if len(l) > 1:
                    qid = l[0]
                    query = l[1]
                    qid2query[qid] = query
    return qid2query
def get_query_statistics(qid2query):
    vocab = Counter()
    query_word_length = []
    query_char_length = []
    one_word_query = 0
    for qid in qid2query:
        query = qid2query[qid]
        query_char_length.append(len(query))
        query_word_length.append(len(query.split(' ')))
        if query_word_length[-1] == 1:
            one_word_query += 1
        for word in query:
            vocab[word] += 1
    print("There are {} unique queries with an average length of {} words and {} characters".format(len(qid2query), np.average(query_word_length), np.average(query_char_length)))
    print("The longest query is {} words long and there are {} queries with a single word".format(np.max(query_word_length). one_word_query))
    print("There are {} unique words and the 15 most common are {}".format(len(vocab), vocab.most_common(15)))

def main(args):
    qid2query = load_queries(args.queries_dir)
    q2click = load_qrels(args.qrel_file)
    doc2query = get_q2click_stats(q2click)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Turn Orcas into clickgraph')   
    parser.add_argument('--qrel_file', default='data/orcas-doctrain-qrels.tsv', type=str, help='Location of QREL file to make graph from')
    parser.add_argument('--queries_dir', default='data', type=str, help='Directory where queries files are located')
    parser.add_argument('--min_query_lenght', default=2, type=int, help='Minimum query length')
    parser.add_argument('--min_support', default=5, type=int, help='Minimum number of connecting documents to mean q1 and q2 have a edge')
    args = parser.parse_args()
    main(args)