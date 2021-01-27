from bert_serving.client import BertClient
from py2neo import Node, Relationship, Graph, NodeMatcher, RelationshipMatcher, Subgraph
from preprocess import util
import pandas as pd
from AttentiveRec import Config
from ner import entity_link
from kg_nearest_k import get_relational_nodes

username = "***"
password = "***"
host = "*.*.*.*:****"

def node_encode():
    server_ip = "127.0.0.1"
    bert_client = BertClient(ip=server_ip, check_version=False, port=5557, port_out=5558)

    graph = Graph(host, username=username, password=password)
    tx = graph.begin()
    matcher_1 = NodeMatcher(graph)
    nodes = matcher_1.match("node")

    i = 0
    for node in nodes:
        if 'embedding' not in node:
            # try:
            node['embedding'] = bert_client.encode([node['name']])            # except:
            #     print("编码出错")
        i += 1
        if i % 100 == 0:
            print(i)

    tx.push(nodes)  # push到服务器
    # or
    tx.commit()

def save_teacher_nodes():
    graph = util.get_neo4j_graph()
    df_teacher = pd.read_csv(Config.all_t_path)
    matcher = NodeMatcher(graph)
    for index, row in df_teacher.iterrows():
        cur_teacher = matcher.match("teacher", id=row['id']).first()

        if cur_teacher:
            continue

        node = Node("teacher", id=row["id"], t_id=row["t_id"],
                    t_name=row["t_name"], department=row["department"],
                    t_institute=row["t_institute"])
        graph.create(node)

        if index % 100 == 0:
            print(index)

def save_teacher_department_relation():
    graph = util.get_neo4j_graph()
    df_teacher = pd.read_csv(Config.all_t_path)
    matcher = NodeMatcher(graph)
    for index, row in df_teacher.iterrows():
        cur_teacher = matcher.match("teacher", id=row['id']).first()

        if not cur_teacher:
            continue

        ner_results = set(entity_link.ner([row['department']], 3)[0])
        ner_results_inst = set(entity_link.ner([row['t_institute']], 3)[0])
        ner_results = ner_results | ner_results_inst
        print(ner_results)
        relations = []
        for entity in ner_results:
            relational_nodes = get_relational_nodes(graph, entity)
            for r in relational_nodes:  # r (node, deep)
                relation = Relationship(cur_teacher, "department", r)
                relations.append(relation)

        if relations:
            temp = Subgraph(relationships=relations)
            graph.create(temp)

        if index % 100 == 0:
            print(index)

def save_copyright_teacher_relation():
    df_copyrights = pd.read_csv(Config.copyright_path)
    graph = util.get_neo4j_graph()

    matcher = NodeMatcher(graph)
    for index, row in df_copyrights.iterrows():
        node = matcher.match("copyright", copyright_id=row['copyright_id']).first()

        if not node:
            ner_results = entity_link.ner([row['software_name']], 5)[0]
            print(ner_results)

            node = Node("copyright", copyright_id=row["copyright_id"], software_name=row["software_name"],
                        complete_time=row["complete_time"])
            graph.create(node)
            relations = []
            for entity in ner_results:
                relational_nodes = get_relational_nodes(graph, entity)
                for r in relational_nodes:  # r (node, deep)
                    relation = Relationship(node, "field", r)
                    relations.append(relation)

            if relations:
                temp = Subgraph(relationships=relations)
                graph.create(temp)

        teacher = matcher.match("teacher", t_id=row['t_id']).first()

        if teacher:
            relation = Relationship(teacher, "have_copyright", node)
            graph.create(relation)

        if index % 100 == 0:
            print(index)

def generate_kg_final():
    kg_final = open(Config.mvkgan_data_path + "kg_final.txt", "w")

    relation_map_file = open(Config.mvkgan_data_path + "relation_list.txt", "w")
    relation_map = dict()

    relation_map_file.write("org_id remap_id\n")

    graph = util.get_neo4j_graph()
    matcher = NodeMatcher(graph)

    all_relations = graph.match()

    i = 0
    for r in all_relations:
        # print(r.relationships)
        if r["name"] not in relation_map:
            relation_map[r["name"]] = len(relation_map.keys())
            relation_map_file.write("%s %d\n" % (r["name"], relation_map[r["name"]]))

        kg_final.write("%d %d %d\n" % (r.start_node.identity, relation_map[r["name"]], r.end_node.identity))
        i += 1
        if i % 500 == 0:
            print(i)

    relation_map_file.close()
    kg_final.close()

if __name__ == "__main__":
    # save_teacher_nodes()
    # save_teacher_department_relation()
    generate_kg_final()
