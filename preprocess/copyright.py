import pymysql
import pandas

conn = pymysql.connect(
    host="*.*.*.*",
    user="***",
    port=0000,
    password="***",
    database="persona_recommend")

cursor = conn.cursor()
copyright_original = pandas.read_csv('../data/copyright.CSV')

sql = 'insert into xls_handler_copyright(category, school_id, certificate, software_name, abbrev, version, ' \
      'complete_time, first_publish_time, serial_no, department) values'

for index, row in copyright_original.iterrows():
    depart = ''
    try:
        depart = str(int(row['所属学院1']))
    except:
        depart = 'nan'
    values = "(%d, '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s');" \
             % (int(row['成果大类']), row['学校编号'], row['证书号'], row['软件名称'], row['简称'], row['版本号'],
                row['开发完成日期'], row['首次发表日期'], row['流水号'], depart)
    now_sql = sql + values
    cursor.execute(now_sql)
    print(conn.insert_id())
    cur_copy_id = conn.insert_id()
    if row['著作权人/开发人员名单'] is not None:
        teachers = str(row['著作权人/开发人员名单'])
        teachers = teachers.replace('*', '')
        teachers = teachers.replace(' ', '')
        teachers = teachers.split('，')
        print(teachers)
        for teacher in teachers:
            find_teacher_sql = "select ID, department from xls_handler_teacher where name = '%s'" % teacher
            cursor.execute(find_teacher_sql)
            target_teachers = cursor.fetchall()
            try:
                if target_teachers is not None and len(target_teachers) != 0:
                    target_teacher = target_teachers[0]
                    for t in target_teachers:
                        if t[1] == row['所属学院1']:
                            target_teacher = t
                            break
                    copyright_teacher_sql = "insert into xls_handler_copyright_teacher(copyright_id, t_id) " \
                                            "values (%d, '%s')" % (cur_copy_id, target_teacher[0])
                    cursor.execute(copyright_teacher_sql)
            except:
                continue

conn.commit()
