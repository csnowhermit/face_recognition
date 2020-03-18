
import os

'''
    通用工具
'''

'''
    判断字符串中中文的个数
'''
def get_zhcn_number(s):
    count = 0
    for item in s:
        if 0x4E00 <= ord(item) <= 0x9FA5:
            count += 1
    return count