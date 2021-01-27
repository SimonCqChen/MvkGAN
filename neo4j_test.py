from py2neo import Node, Relationship, Graph, NodeMatcher, RelationshipMatcher
import numpy as np
import tf_geometric as tfg

username = "username"
password = "***"
host = "1.1.1.1:1111"
feature_num = 1

def build_nn_graph():
    graph = Graph(host, username=username, password=password)
    matcher_1 = NodeMatcher(graph)

    start_node = matcher_1.match("category", name="软件类目").first()

#     BFS
    unvisited = [start_node]
    node_count = 0
    edge_index = [[], []]
    node_index_id_mapping = {}
    while unvisited:
        start = unvisited.pop(0)
        start_i = node_count
        node_index_id_mapping[start_i] = start.identity
        next_rlts = list(graph.match(start))
        for rlt in next_rlts:
            node_count += 1
            edge_index[0].append(start)
            edge_index[1].append(node_count)
            end_iid = rlt.end_node
            node_index_id_mapping[node_count] = end_iid
            unvisited.append(matcher_1.get(end_iid))

    graph = tfg.Graph(
        x=np.random.randn(5, feature_num),  # 5 nodes, 20 features,
        edge_index=edge_index  # 4 undirected edges
    )

    return graph









