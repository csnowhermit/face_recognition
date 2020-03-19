import pymysql
import base64

'''
    从数据库读取特征
'''

conn = pymysql.connect(host="localhost", port=3306, user="root", password="123456", db="test")
cursor = conn.cursor()

sql = "select name, feature_size, feature, create_time from asf_user_face_info"

cursor.execute(sql)

for r in cursor.fetchall():
    # print(r[0], r[1], r[2], r[3])
    s = r[2]
    print(type(s))
    s = s[s.index("'") + 1: -1]
    if len(s) % 2 != 0:
        s = s + "="
    print(base64.b64decode(s))

cursor.close()