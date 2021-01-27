#!usr/bin/python
from queue import PriorityQueue
from sklearn.metrics.pairwise import cosine_similarity
from bert_serving.client import BertClient
import random
import pandas as pd
from AttentiveRec import Config
from py2neo import Node, Relationship, Graph, NodeMatcher, RelationshipMatcher, Subgraph
from preprocess import util
from ner import entity_link

bert_client = BertClient(ip="127.0.0.1", check_version=False, port=5557, port_out=5558)

class NodeWithDeep():
    def __init__(self, node, deep):
        self.node = node
        self.deep = deep

def get_top_k(pqueue, top_k):
    result = PriorityQueue()
    while pqueue.qsize() > 0:
        try:
            cur = pqueue.get()
            result.put(cur)
            if result.qsize() >= top_k:
                 break
        except:
            continue
    return result

def get_relational_nodes(graph, entity):
    matcher = NodeMatcher(graph)
    start_node = matcher.match("node", name=entity).first()
    if start_node != None:
        return [start_node]
    else:
        return []

def get_fuzzy_relational_nodes(graph, entity, iter=5, first_pick=10, num_limit = 20, max_deep=5, k=10, threshold=0.2):
    result = []
    embedding = bert_client.encode([entity])

    all_picked = PriorityQueue()
    all_visited = {}
    node_num = 16000000

    for i in range(iter):
        bfs_picked = PriorityQueue()
        met_nodes = {}
        first_pick_nodes_id = random.sample(range(node_num), first_pick)
        first_pick_nodes = list(graph.run("match (n:node) WHERE ID(n) IN %s return n" % str(first_pick_nodes_id)))
        for node in first_pick_nodes:
            node = node[0]
            node_embedding = bert_client.encode([node["name"]])
            sim = cosine_similarity(node_embedding, embedding)[0][0]
            bfs_picked.put([1 - sim, NodeWithDeep(node, 0)])
            met_nodes[node] = True
            if node not in all_visited:
                all_picked.put([1 - sim, node])
                all_visited[node] = True

        while bfs_picked.qsize() > 0:
            temp = bfs_picked.get()[1]
            cur_node, deep = temp.node, temp.deep
            if deep > max_deep:
                break

            next_nodes = graph.run(
                "MATCH (next:node)<-[:relation]-(cur:node {name: '%s'}) Return next limit %d" % (cur_node['name'], 10))
            for next_node in next_nodes:
                next_node = next_node[0]
                if next_node not in met_nodes:
                    sim = cosine_similarity(bert_client.encode([next_node["name"]]), embedding)[0][0]
                    try:
                        bfs_picked.put([1 - sim, NodeWithDeep(next_node, deep + 1)])
                    except:
                        continue
                    met_nodes[next_node] = True
                if next_node not in all_visited:
                    all_picked.put([1 - sim, next_node])
                    all_visited[next_node] = True
            bfs_picked = get_top_k(bfs_picked, num_limit)



    for _ in range(k):
        if not all_picked:
            break
        sim, node = all_picked.get()
        if (sim < threshold):
            result.append(node)
        else:
            break

    return result

if __name__ == "__main__":
    df_teacher = pd.read_csv(Config.all_p_path)
    graph = util.get_neo4j_graph()

    matcher = NodeMatcher(graph)
    for index, row in df_teacher.iterrows():
        cur_project = matcher.match("project", p_name=row['p_name']).first()
        if cur_project:
            continue

        ner_results = entity_link.ner([row['p_name']], 5)[0]
        print(ner_results)
        node = Node("project", p_name=row["p_name"], acc_id=row["id"], p_institute=row["p_institute"])
        graph.create(node)
        relations = []
        for entity in ner_results:
            relational_nodes = get_relational_nodes(graph, entity)
            for r in relational_nodes: # r (node, deep)
                relation = Relationship(node, "field", r)
                relations.append(relation)

        if relations:
            temp = Subgraph(relationships=relations)
            graph.create(temp)

        if index % 100 == 0:
            print(index)

