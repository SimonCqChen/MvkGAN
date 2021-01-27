import pymysql
from bert_base.client import BertClient
import jieba
import jieba.analyse as ja
from py2neo import Graph
import random

def mysqlSearch(cursor, sql, params=()):
    cursor.execute(sql, params)
    return cursor.fetchall()

def getProjects(num):
    """
    从mysql得到科研项目的名称列表，长度为num
    """
    db = pymysql.connect("localhost", "***", "***", "***")
    cursor = db.cursor()
    projects = mysqlSearch(cursor, "select name from xls_handler_project limit %s", (num,))
    db.close()
    sList = []
    for project in projects:
        if project[0] in sList:
            continue
        sList.append(project[0])
    return sList

def getStopWord():
    stopWord = []
    with open("/mnt/sdb/ccq/persona-recommend/data/stopWord.txt", 'r', encoding='utf-8') as f:
        for s in f.readlines():
            stopWord.append(s.strip())
    return stopWord

def cut(sent):
    stopWord = getStopWord()
    wList = jieba.cut(sent)
    wList2 = []
    for w in wList:
        if w not in stopWord:
            wList2.append(w)
    return wList2

def ner(sList, keyNum):
    """
    利用BERT-BiLSMT-CRF-NER得到s中的命名实体列表
    """
    # https://blog.csdn.net/macanv/article/details/85684284?tdsourcetag=s_pctim_aiomsg
    # cmd cd D:\Code\python\Scripts
    # bert-base-serving-start -model_dir E:\test\train\old\output_dir -bert_model_dir E:\test\BERT_NER\chinese_L-12_H-768_A-12 -mode NER
    rList = []
    with BertClient(ip='127.0.0.1',
                    ner_model_dir='/mnt/sdb/ccq/persona-recommend/graph/ner/output_dir',
                    show_server_config=False,
                    check_version=False, check_length=False, mode='NER') as bc:
        iter = 0
        while iter*1000 < len(sList):
            subSList = sList[iter*1000:(iter+1)*1000]  # 最多一次处理1000条
            entityList = bc.encode(sList)
            iter += 1
            for s, entity in zip(subSList, entityList):
                wList = []
                word = ""
                flag = False
                for char, label in zip(s, entity):
                    if label != "O":
                        flag = True
                        word += char
                    elif flag:
                        flag = False
                        for w in cut(word):
                            if w not in wList:
                                wList.append(w)
                if len(wList) > keyNum:
                    wList = random.sample(wList, keyNum)
                rList.append(wList)
    return rList

def keyWord(s, wList, keyNum):
    """
    利用TF-IDF得到s中的关键词
    """
    kwList = ja.extract_tags(s, topK=keyNum, allowPOS=('n'))  # tf-idf
    for w in kwList:
        if w not in wList:
            wList.append(w)

def neoSearch(s, n=5):
    """
    搜索与s相邻的前n个node
    """
    # Node: ":ID", "name", ":TYPE"
    # Relation: ":START_ID", ":END_ID", "name", ":TYPE"
    # 导入: neo4j-admin load --from=neo4j.dump --database=neo4j
    # cmd E:\Tool\neo4j\neo4j-community-4.1.0-windows\neo4j-community-4.1.0\bin neo4j.bat console
    graph = Graph("http://localhost:7474", auth=("***","***"))
    data = graph.run('OPTIONAL match (p:node {name: "' + s + '"})--(q:node) return q limit '+str(n)).data()  # cypher
    nameList = []  # 返回相邻节点的name，若无相邻节点则返回none
    for nodeDic in data:
        if nodeDic['q'] is not None:
            nameList.append(nodeDic['q']["name"])
        else:
            nameList.append(None)
    return nameList

def control(num, keyNum=4):
    """
    实体链接主函数
    :param num: 从科研项目标题数据库表中提取的标题文本数量
    :param keyNum: 每条文本的关键词数量
    :return:
    sList: 标题文本列表
    rList: 每条文本的关键词和链接到的实体
    """
    sList = getProjects(num)
    entityList = ner(sList, keyNum)
    ja.set_idf_path("/mnt/sdb/ccq/persona-recommend/data/idf.txt")
    ja.set_stop_words("/mnt/sdb/ccq/persona-recommend/data/stopWord.txt")
    for i, wList in enumerate(entityList):
        keyWord(sList[i], wList, keyNum)
    rList = []
    for wList in entityList:
        r = []
        for w in wList:
            r.append((w, neoSearch(w)))
        rList.append(r)
    return sList, rList

def show(sList, rList):
    # 打印结果
    for i, s in enumerate(sList):
        rs = rList[i]
        print(s)
        for r in rs:
            print(r)
        print("*"*20)

if __name__ == "__main__":
    sList, rList = control(50)
    show(sList, rList)