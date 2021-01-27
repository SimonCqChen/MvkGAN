import pandas as pd
import pickle
from collections import Counter
from AttentiveRec import Config
from preprocess import util
from py2neo import NodeMatcher

def feature_count():
    all_ffm = open('../feature_data/all_with_bert_embed_ffm.csv')

    biggest = 0

    for line in all_ffm:
        features = line.strip().split(' ')
        print(len(features))
        for f in features:
            nums = f.split(':')
            if len(nums) == 3:
                biggest = int(nums[1]) if int(nums[1]) > biggest else biggest

        print('biggest:', biggest)


def comma_to_blank():
    all_ffm = open('../feature_data/all_with_bert_embed_ffm_1.csv')
    all_ffm_blank = open('../feature_data/all_with_bert_embed_ffm.csv', 'w')

    for line in all_ffm:
        temp = line.replace(',', ' ')
        all_ffm_blank.write(temp)

    all_ffm_blank.close()


def split_ffm_data():
    all_ffm = open('../feature_data/all_with_bert_embed_ffm.csv')
    train = open('../feature_data/train_with_bert_embed_ffm.csv', 'w')
    test = open('../feature_data/test_with_bert_embed_ffm.csv', 'w')

    lines = all_ffm.readlines()

    train_lines = lines[:57203]
    test_lines = lines[57203:]

    train.writelines(train_lines)
    test.writelines(test_lines)

    train.close()
    test.close()


def add_id():
    train = open('../feature_data/train_with_bert_embed_ffm.csv')
    test = open('../feature_data/test_with_bert_embed_ffm.csv')
    train_id = open('../feature_data/train_with_bert_embed_ffm_ID.csv', 'w')
    test_id = open('../feature_data/test_with_bert_embed_ffm_ID.csv', 'w')

    id = 0

    for line in train:
        with_id = '%s%%ID_%d\n' % (line.strip(), id)
        train_id.write(with_id)
        id += 1
    train_id.close()
    for line in test:
        with_id = '%s%%ID_%d\n' % (line.strip(), id)
        test_id.write(with_id)
        id += 1
    test_id.close()


def id_transfer(teachers, projects):
    t2id = {}
    p2id = {}
    for index, row in teachers.iterrows():
        t2id[row['t_id']] = row['id']

    for index, row in projects.iterrows():
        p2id[row['project_id']] = row['id']

    pickle.dump(t2id, open("../data/teacher_original_id2id", "wb"))
    pickle.dump(p2id, open("../data/project_original_id2id", "wb"))


def statistic_t_num():
    t_p_match_path = '/mnt/sdb/ccq/persona-recommend/feature_data/' + "t_p_match_dict.pickle"
    match = pickle.load(open(t_p_match_path, "rb"))
    result = Counter()
    for t in match["p"]:
        result[len(match["p"][t])] += 1

    print(result)

def generate_mvkgan_data():
    match = pickle.load(open(Config.t_p_match_path, "rb"))
    train = open(Config.mvkgan_data_path + "train.txt", "w")
    test = open(Config.mvkgan_data_path + "test_original_id.txt", "w")
    for t in match["t"]:
        if t % 5 < 4:
            temp = train
        else:
            temp = test
        temp.write(str(t))
        for interact_p in match["t"][t]:
            temp.write(" ")
            temp.write(str(interact_p))
        temp.write("\n")

    train.close()
    test.close()

def generate_mvkgan_data():
    match = pickle.load(open(Config.t_p_match_path, "rb"))
    train = open(Config.mvkgan_data_path + "train.txt", "w")
    test = open(Config.mvkgan_data_path + "test.txt", "w")
    for t in match["t"]:
        if t % 5 < 4:
            temp = train
        else:
            temp = test
        temp.write(str(t))
        for interact_p in match["t"][t]:
            temp.write(" ")
            temp.write(str(interact_p))
        temp.write("\n")

    train.close()
    test.close()

def generate_mvkgan_data_with_graph():
    graph = util.get_neo4j_graph()
    matcher = NodeMatcher(graph)

    match = pickle.load(open(Config.t_p_match_path, "rb"))
    train = open(Config.mvkgan_data_path + "train.txt", "w")
    test = open(Config.mvkgan_data_path + "test.txt", "w")

    i = 0
    for t in match["t"]:
        if t % 5 < 4:
            temp = train
        else:
            temp = test
        t_node = matcher.match("teacher", id=t).first()
        temp.write(str(t_node.identity))
        for interact_p in match["t"][t]:
            p_node = matcher.match("project", acc_id=interact_p).first()
            if not p_node:
                continue

            temp.write(" ")
            temp.write(str(p_node.identity))
        temp.write("\n")
        i += 1
        if i % 10 == 0:
            print(i)

    train.close()
    test.close()

if __name__ == "__main__":
    # all_t = pd.read_csv("../feature_data/all_worker.csv")
    # all_p = pd.read_csv("../feature_data/all_task.csv")
    # id_transfer(all_t, all_p)
    # comma_to_blank()
    # feature_count()
    # split_ffm_data()
    # add_id()
    # statistic_t_num()
    generate_mvkgan_data_with_graph()
