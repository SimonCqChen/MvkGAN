# file
# train_file = 'ex_data/dataset1.csv'
# valid_file = 'ex_data/dataset2.csv'
# test_file = 'ex_data/dataset3.csv'
#
# train_save_file = 'ex_data/dataset1.txt'
# valid_save_file = 'ex_data/dataset2.txt'
# test_save_file = 'ex_data/dataset3.txt'

train_file = 'ex_data/train_with_bert_embed.csv'
valid_file = 'ex_data/test_with_bert_embed.csv'
test_file = 'ex_data/test_with_bert_embed.csv'

train_save_file = 'ex_data/train_with_bert_embed.txt'
valid_save_file = 'ex_data/test_with_bert_embed.txt'
test_save_file = 'ex_data/test_with_bert_embed.txt'

label_name = 'target'

# features
# numeric_features = ['all_launch_count', 'last_launch', 'all_video_count', 'last_video', 'all_video_day',
#                     'all_action_count', 'last_action',
#                     'all_action_day', 'register_day']
bert_embed_size = 768
numeric_features = ['funds', 'category']

for i in range(bert_embed_size):
    temp = ['copyright_embed' + str(i), 'p_embed' + str(i), 't_embed' + str(i), 'institute_embed' + str(i)]
    for col in temp:
        if col not in numeric_features:
            numeric_features.append(col)

single_features = ['main_category']

# single_features = ['register_type', 'device_type']
multi_features = []

num_embedding = False
single_feature_frequency = 10
multi_feature_frequency = 0

# model

FM_layer = True
DNN_layer = True
CIN_layer = False

use_numerical_embedding = False


embedding_size = 8

dnn_net_size = [128,64,32]
cross_layer_size = [10,10,10]
cross_direct = False
cross_output_size = 1

# train
batch_size = 4096
epochs = 40
learning_rate = 0.01

threshold = 0.70

paper_path = '/mnt/sdb/ccq/persona-recommend/Scholar-master/BaiduXueShu/data/'
feature_data_path = '/mnt/sdb/ccq/persona-recommend/feature_data/'
data_path = '/mnt/sdb/ccq/persona-recommend/data/'
teacher_paper_embedding_path = feature_data_path + 'teacher_name_paper_embed_dict'
teacher_papers_path = data_path + "teacher_papers_dict"
field_layer_path = data_path + "field_layer"
t_p_match_path = feature_data_path + "t_p_match_dict.pickle"
all_t_path = feature_data_path + "all_worker.csv"
all_p_path = feature_data_path + "all_task.csv"
copyright_path = feature_data_path + "copyright_teacher.csv"

train_data_path = feature_data_path + "pt_same_depart_train.csv"
test_data_path = feature_data_path + "pt_same_depart_test_balance.csv"

t_id_transfer_path = data_path + "teacher_original_id2id"
p_id_transfer_path = data_path + "../data/project_original_id2id"


d2 = 100
kH = 100
e0_d = 80074
e1_d = 164519
e2_d = 621
e3_d = 329
field_layer_d = [e1_d, e2_d, e3_d]

H = 4  # 层数

lambda_a = 1 / 3
lambda_x = 1 / 3
lambda_y = 1 / 3
k1 = 10

num_topics = 300

d3 = 10

# mvkgan 配置
neo4j_username = "***"
neo4j_password = "***"
neo4j_host = "0.0.0.0:0000"

dropout = 0.2

bert_server_ip = "127.0.0.1"

mvkgan_data_path = '/mnt/sdb/ccq/persona-recommend/graph/mvkgan/Data_example/sjtu-project-teacher/'

max_len = 20
max_nodes = 10000
bilstm_n_hidden = 128
