import ir_datasets
dataset = ir_datasets.load('msmarco-document/orcas')
for query in dataset.queries_iter():
    query # namedtuple<query_id, text>

for doc in dataset.docs_iter():
    doc # namedtuple<doc_id, url, title, body>

for qrel in dataset.qrels_iter():
    qrel # namedtuple<query_id, doc_id, relevance, iteration>

for scoreddoc in dataset.scoreddocs_iter():
    scoreddoc # namedtuple<query_id, doc_id, score>