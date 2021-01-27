import pandas as pd
import pickle
from py2neo import Graph
from AttentiveRec import Config

def generate_addition_data(df_path, paper_embed, save_path):
    df = pd.read_csv(open(df_path))
    advise_data = []
    for index, row in df.iterrows():
        t_name = row[-8]
        try:
            advise_data.append(paper_embed[t_name])
        except:
            advise_data.append([])

    # TODO 补全缺失，使每一条数据对应的附加信息向量数量一致
    pickle.dump(advise_data, open(save_path, 'wb'))

def get_neo4j_graph():
    graph = Graph(Config.neo4j_host, username=Config.neo4j_username, password=Config.neo4j_password)
    return graph

if __name__ == '__main__':
    file = open('../feature_data/teacher_name_paper_embed_dict')
    paper_embed = pickle.load(file)
    generate_addition_data('../feature_data/pt_same_depart_train.csv', paper_embed, '../feature_data/pt_train_addition')
    generate_addition_data('../feature_data/pt_same_depart_test_balance.csv', paper_embed,
                           '../feature_data/pt_test_addition')
