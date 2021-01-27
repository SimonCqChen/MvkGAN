import pandas
import pickle
import random
import pandas as pd

# 生成同department负样本
def generate_same_depart_nega():
    projects = pandas.read_csv('feature_data/all_task.csv')
    teachers = pandas.read_csv('feature_data/all_worker.csv')
    institute_teachers = {}
    projects_id_teacher = {}
    projects_id_project = {}

    pt_institute_nega = open('feature_data/pt_depart_nega.csv', 'w')

    for index, row in projects.iterrows():
        if row['project_id'] not in projects_id_teacher:
            projects_id_teacher[row['project_id']] = [row['t_id']]
            projects_id_project[row['project_id']] = row
        else:
            projects_id_teacher[row['project_id']].append(row['t_id'])

    for index, row in teachers.iterrows():
        if row['t_institute'] not in institute_teachers:
            institute_teachers[row['t_institute']] = [row]
        else:
            institute_teachers[row['t_institute']].append(row)

    print('institute对应教师统计完成')

    project_i = 0
    for p_id in projects_id_teacher:
        try:
            p = projects_id_project[p_id]
            p_institute = p['p_institute']
            i = 0
            while i < min(5, len(institute_teachers[p_institute])):
                random_index = random.randint(0, len(institute_teachers[p_institute]))
                if institute_teachers[p_institute][random_index]['t_id'] not in projects_id_teacher[p_id]:
                    t = institute_teachers[p_institute][random_index]
                    pt_institute_nega.write(str(p['id']) + ',' + str(p['main_category']) + ',' + str(p['category']) + ',' + p['project_id'] + ',' +
                                      p['p_name'] + ',' + str(p['type']) + ',' + str(p['second_type']) + ',' + p['start_time'] + ',' +
                                      p['end_time'] + ',' + str(p['funds']) + ',' + str(p['character']) + ',' + str(p['ranking']) + ',' +
                                    str(p['status']) + ',' + p['p_institute'] + ',' + t['t_id'] + ',' + t['t_name'] + ',' +
                                    str(t['native_place']) + ',' + t['employment_date'] + ',' + t['title'] + ',' +
                                      t['department'] + ',' + t['gender'] + ',' + t['t_institute'] + ',0\n')
                    i = i + 1
            project_i += 1
            if project_i % 500 == 0:
                print('已写入%d个项目的负样本' % project_i)
        except:
            continue

    pt_institute_nega.close()


def generate_same_depart_train_test():
    pt_posi = open('feature_data/pt_positive.csv', encoding='utf-8')
    pt_nega = open('feature_data/pt_depart_nega.csv', encoding='utf-8')
    pt_test = open('feature_data/pt_same_depart_test.csv', 'w', encoding='utf-8')
    # pt_train = open('feature_data/pt_same_depart_train.csv', 'w', encoding='utf-8')
    #
    # pt_posi_train = pt_posi.readlines()
    # pt_nega_train = pt_nega.readlines()
    #
    # for i in range(0, 40000):
    #     pt_train.write(pt_posi_train[i])
    #     pt_train.write(pt_nega_train[i * 4])

    pt_posi_test = pt_posi.readlines()[40000:]
    pt_nega_test = pt_nega.readlines()[160004:]
    for line in pt_posi_test:
        pt_test.write(line)

    for line2 in pt_nega_test:
        pt_test.write(line2)

    pt_posi.close()
    pt_nega.close()
    pt_test.close()
    # pt_train.close()


def reduce_nega_instance():
    f = open('feature_data/pt_same_depart_test.csv', 'r', encoding='utf-8')
    balance = open('feature_data/pt_same_depart_test_balance.csv', 'w', encoding='utf-8')
    pt_posi = f.readlines()

    for i in range(len(pt_posi)):
        if i < 11000:
            balance.write(pt_posi[i])
        else:
            if i % 3 == 0:
                balance.write(pt_posi[i])

    f.close()
    balance.close()


def print_paper_crawler_keyword():
    teachers = pandas.read_csv('../feature_data/all_worker.csv')
    keyword = open('../data/crawler_keyword.csv', 'w', encoding='utf-8')
    for index, row in teachers.iterrows():
        t_info = '%s %s %s\n' % (row['t_name'], row['department'], '上海交通大学')
        keyword.write(t_info)
    keyword.close()


def print_scholar_crawler_keyword():
    teachers = pandas.read_csv('../feature_data/all_worker.csv')
    keyword = open('../data/scholar_keyword.csv', 'w', encoding='utf-8')
    for index, row in teachers.iterrows():
        t_info = '%s %s\n' % (row['t_name'], '上海交通大学' + row['department'])
        keyword.write(t_info)
    keyword.close()


def build_t_p_department(teachers, projects):
    department_t = {}
    department_p = {}

    for index, row in teachers.iterrows():
        department = row['t_institute']
        if department not in department_t:
            department_t[department] = []
        department_t[department].append(row['id'])

    for index, row in projects.iterrows():
        department = row['p_institute']
        if department not in department_p:
            department_p[department] = []
        department_p[department].append(row['id'])

    with open("../feature_data/department_t_dict", 'wb') as file:
        pickle.dump(department_t, file)
    with open("../feature_data/department_p_dict", 'wb') as file:
        pickle.dump(department_p, file)


all_t = pd.read_csv("../feature_data/all_worker.csv")
all_p = pd.read_csv("../feature_data/all_task.csv")
build_t_p_department(all_t, all_p)
