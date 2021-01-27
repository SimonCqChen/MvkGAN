import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.datasets import make_classification
import df2FFM

########################### Lets build some data and test ############################
###


# train, y = make_classification(n_samples=100, n_features=5, n_informative=2, n_redundant=2, n_classes=2, random_state=42)
#
# train=pd.DataFrame(train, columns=['int1','int2','int3','s1','s2'])
# train['int1'] = train['int1'].map(int)
# train['int2'] = train['int2'].map(int)
# train['int3'] = train['int3'].map(int)
# train['s1'] = round(np.log(abs(train['s1'] +1 ))).map(str)
# train['s2'] = round(np.log(abs(train['s2'] +1 ))).map(str)
# train['clicked'] = y
#
#
# ffm_train = FFMFormatPandas()
# ffm_train_data = ffm_train.fit_transform(train, y='clicked')
# print('Base data')
# print(train[0:10])
# print('FFM data')
# print(ffm_train_data[0:10])
#
# for item in ffm_train_data:
#     print(item)
#
# out = open('test_ffm.csv', 'w')
# out.write(str(ffm_train_data[0]))
# out.close()

TRAIN_FILE_EMBED = '../feature_data/train_with_bert_embed.csv'
TEST_FILE_EMBED = '../feature_data/test_with_bert_embed.csv'

IGNORE_COLS = [
    "id",  # "target",
    # "ps_calc_01", "ps_calc_02", "ps_calc_03", "ps_calc_04",
    # "ps_calc_05", "ps_calc_06", "ps_calc_07", "ps_calc_08",
    # "ps_calc_09", "ps_calc_10", "ps_calc_11", "ps_calc_12",
    # "ps_calc_13", "ps_calc_14",
    # "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin",
    # "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin"
    'p_id', 'project_id', 'p_name', 'type', 'second_type', 'start_time', 'end_time',
    'character', 'ranking', 'status', 't_id', 't_name', 'native_place',
    'employment_date', 'title', 'gender', 't_institute'
]

CATEGORICAL_COLS = [
    # 'ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat',
    # 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat',
    # 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat',
    # 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat',
    # 'ps_car_10_cat', 'ps_car_11_cat',
    'main_category', 'p_institute', 'department'
]


def _load_embed_data():
    dfTrain = pd.read_csv(TRAIN_FILE_EMBED)
    dfTest = pd.read_csv(TEST_FILE_EMBED)

    cols = [c for c in dfTrain.columns if (not c in IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain["target"].values
    X_test = dfTest[cols].values
    ids_test = dfTest["id"].values

    dfTrain = dfTrain[cols]
    dfTest = dfTest[cols]

    return dfTrain, dfTest, X_train, y_train, X_test, ids_test


df_train, df_test, _, _, _, _ = _load_embed_data()

df_train = df_train.append(df_test)

label_col = ['target']

ffm_train = df2FFM.ffm(df_train, CATEGORICAL_COLS, label_col, path='../feature_data/all_with_bert_embed_ffm_1.csv')

# train_ffm = open('../feature_data/train_with_bert_embed_ffm.csv')
# test_ffm = open('../feature_data/test_with_bert_embed_ffm.csv')
# train_ffm_change = open('../feature_data/train_with_bert_embed_ffm_1.csv', 'w')
# test_ffm_change = open('../feature_data/test_with_bert_embed_ffm_1.csv', 'w')
#
# for line in train_ffm:
#     temp = line.replace(',', ' ')
#     train_ffm_change.write(temp)
#
# train_ffm_change.close()
#
# for line in test_ffm:
#     temp = line.replace(',', ' ')
#     test_ffm_change.write(temp)
#
# test_ffm_change.close()

# test_ffm = open('../feature_data/test_with_bert_embed_ffm_1.csv')
#
# for line in test_ffm:
#     features = line.strip().split(' ')
#     print(len(features))

# ffm_train_file = open(, 'w')
# ffm_test_file = open(, 'w')
#
# for item in ffm_train:
#     ffm_train_file.write(str(item) + '\n')
#
# ffm_train_file.close()
#
# for item in ffm_test:
#     ffm_test_file.write(str(item) + '\n')
#
# ffm_test_file.close()

