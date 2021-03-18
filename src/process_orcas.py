import os
import random
import argparse
import numpy as np
from collections import Counter
import networkx as nx
import itertools

def load_qrels(filename):
    qid2click, doc2query = {}, {} # query to the documents that are clicks on it
    with open(filename) as f:
        for l in f:
            l = l.strip().split(' ')
            document = l[2]
            query = l[0]
            if query not in qid2click:
                qid2click[query] = set()
            qid2click[query].add(document)
    for q in qid2click:
        for doc in qid2click[q]:
            if doc not in doc2query:
                doc2query[doc] = set()
            doc2query[doc].add(q)
    return qid2click, doc2query

def get_click_stats(q2click, doc2query):
    doc_counts = []
    for q in q2click:
        doc_counts.append(len(q2click[q]))
    query_counts = []
    for doc in doc2query:
        query_counts.append(len(doc2query[doc]))
    print("There are {} queries and {} documents.".format(len(q2click), len(doc2query)))
    print("The average query has {} documents, the median is {} and the max is {}, and the min is {}".format(np.average(doc_counts), np.median(doc_counts), np.max(doc_counts), np.min(doc_counts)))
    print("The average document has {} queries, the median has {}, the max is {} and the min is {}".format(np.average(query_counts), np.median(query_counts), np.max(query_counts), np.min(query_counts)))

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

def get_query_stats(qid2query):
    vocab = Counter()
    query_word_length = []
    query_char_length = []
    one_word_query = 0
    for qid in qid2query:
        query = qid2query[qid]
        query_char_length.append(len(query))
        query = query.split(' ')
        query_word_length.append(len(query))
        if query_word_length[-1] == 1:
            one_word_query += 1
        for word in query:
            vocab[word] += 1
    print("There are {} unique queries with an average length of {} words and {} characters".format(len(qid2query), np.average(query_word_length), np.average(query_char_length)))
    print("The longest query is {} words long and there are {} queries with a single word".format(np.max(query_word_length), one_word_query))
    print("There are {} unique words and the 30 most common are {}".format(len(vocab), vocab.most_common(30)))

def filter_queries(args, qid2query, doc2query):
    filtered_qid2query = {}
    for qid in qid2query:
        query = qid2query[qid].split(' ')
        if len(query) > args.min_query_length and len(query) < args.max_query_length:
            filtered_qid2query[qid] = ' '.join(query)
    print("Originally there are {} queries. With filtering there are {} queries.".format(len(qid2query), len(filtered_qid2query)))
    return filtered_qid2query

def document_based_clustering(args, qid2query, doc2query):
    qids = list(qid2query.keys())
    dataset = []
    for doc in doc2query:
        if len(doc2query[doc ]) > 1: # only look for documents that have > 1 query co clicks
            combinations = get_combinations(doc2query[doc], qid2query)
            source_samples = []# a list of current sampled sources
            for sample in combinations:
                source_samples.append("{}\t{}\t1\n".format(sample[0],sample[1]))
                random_negative = random.choice(qids) #negative random sampling
                if np.random.randint(2) == 0:
                    source_samples.append("{}\t{}\t0\n".format(sample[0], random_negative)) # negative sample
                else:
                    source_samples.append("{}\t{}\t0\n".format(sample[1], random_negative)) # negative sample
            random.shuffle(source_samples)
            dataset += source_samples[:args.per_source_samples]
    random.shuffle(dataset)
    return dataset[:args.dataset_size]

def get_combinations(qids, qid2query):
    combinations = []
    for qid in qids:
        for second_qid in qids:
            if qid != second_qid and qid in qid2query and second_qid in qid2query:
                combinations.append((qid, second_qid))
    return combinations

def get_edges(source, g, depth):
    edges, related_nodes = [],[]
    for edge in g.edges(source):
        edges.append(edge)
        related_nodes.append(edge[1])
    if depth < 2:
        return edges
    else:
        depth -= 1
        for neighbor in related_nodes:
            edges += get_edges(neighbor, g, depth)
    return edges

def make_doc_graph(args, qid2query, doc2query, qid2click):
    g = nx.Graph()
    for doc in doc2query:
        g.add_node(doc)
    for doc in doc2query:
        for qid in doc2query[doc]:
            if qid in qid2query:
                associated_docs = qid2click[qid]
                for doc2 in associated_docs:
                    if doc2 != doc:
                        g.add_edge(doc, doc2)
                        g.add_edge(doc2, doc)
    print("Graph Construction Done. There are {} nodes and {} edges".format(g.number_of_nodes(),g.number_of_edges()))
    return g

def make_query_graph(args, qid2query, doc2query, qid2click):
    g = nx.Graph()
    for qid in qid2query:
        g.add_node(qid)
    for doc in doc2query:
        if len(doc2query[doc]) > 1:
            for pair in itertools.combinations(doc2query[doc], 2):
                if pair[0] in qid2query and pair[1] in qid2query:
                    g.add_edge(pair[0], pair[1])
                    g.add_edge(pair[1], pair[0])
    print("Graph Construction Done. There are {} nodes and {} edges".format(g.number_of_nodes(),g.number_of_edges()))
    return g
    
def get_document_graph_neighbors(args, g):
    pairs = []
    potential_nodes = list(g.nodes)
    while len(pairs) < args.dataset.size():
        source = random.choice(potential_nodes)
        edge_to_sample_from = get_edges(source, g, args.graph_sample_depth)
        random.shuffle(edges_to_sample)
        edges_for_data_sampling = edge_to_sample[:args.per_source_sample_size]
        # select a node
        # get edges
        #
def write_dataset(args, dataset):
    with open(args.output_name, 'w') as w:
        for item in dataset:
            w.write(item)
        
def main(args): 
    print("Loading Queries")
    qid2query = load_queries(args.queries_dir)
    print("Loading Orcas")
    qid2click, doc2query = load_qrels(args.qrel_file)
    if args.stats:
        print("Printing data statistics")
        get_click_stats(qid2click, doc2query)
        get_query_stats(qid2query)         
    print("Filtering data to min/max constraints")
    filtered_qid2query= filter_queries(args, qid2query, doc2query)
    if args.doc_clustering:
        print("Creating data via document click clusters")
        dataset = document_based_clustering(args, qid2query, doc2query)
    if args.do_query_graph:
        print("Creating query graph")
        qg = make_query_graph(args, qid2query, doc2query, qid2click)
        print("Creating data based on graph")
    if args.do_document_graph:
        print("Creating document graph")
        dg = make_doc_graph(args, qid2query, doc2query, qid2click)
        print("Creating data based on document graph")
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Turn Orcas into clickgraph')
    parser.add_argument('--stats', action='store_true', help='Run to calculate dataset stats')
    parser.add_argument('--graph_sample_depth', type=int, default = 1, help='how far away to consider neighbors when edge similarity_clustering')
    parser.add_argument('--dataset_size', type=int, default=1000000, help='How big should the target dataset be')
    parser.add_argument('--per_source_samples', type=int, default=10, help='How many data samples will we max take. Avoids overpresenting highly connected nodes')
    parser.add_argument('--doc_clustering', action='store_true', help='Run dataset creation via document clusetring(queries that share a document click are considered similair')
    parser.add_argument('--do_query_graph', action='store_true', help='Run dataset creation via query graph')
    parser.add_argument('--do_document_graph', action='store_true', help='Run dataset creation via document graph')
    parser.add_argument('--edge_hops', default=2, type=int, help='Edge hops for dataset creation')   
    parser.add_argument('--qrel_file', default='data/orcas-doctrain-qrels.tsv', type=str, help='Location of QREL file to make graph from')
    parser.add_argument('--queries_dir', default='data', type=str, help='Directory where queries files are located')
    parser.add_argument('--min_query_length', default=2, type=int, help='Minimum query length')
    parser.add_argument('--max_query_length', default=20, type=int, help='Maximum queyr legth for dataset processing')
    parser.add_argument('--edge_skip', default=2, type=int, help='How far away in co clicks do queries need to be for similarity')
    parser.add_argument('--doc_cluster_output', default='data/doc_cluster_labels.tsv', type=str, help='File where outputed data from clustering')
    args = parser.parse_args()
    main(args)