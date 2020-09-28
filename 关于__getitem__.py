class DataBase:
    '''Python 3 中的类'''

    def __init__(self, id, address):
        '''初始化方法'''
        self.id = id
        self.address = address
        self.d = {self.id: 1,
                  self.address: "192.168.1.1",
                  }

    def __getitem__(self, key):
        # return self.__dict__.get(key, "100")
        return self.d.get(key, "default")


data = DataBase(1, "192.168.2.11")
print(data["hi"])
print(data[data.id])



# import re
# RE_WORD = re.compile(r'\w+')
# class Sentence:
#     def __init__(self, text):
#         self.text = text
#         self.words = RE_WORD.findall(text)  # re.findall函数返回一个字符串列表，里面的元素是正则表达式的全部非重叠匹配
#     def __getitem__(self, index):
#         return self.words[index]
#
# s = Sentence('The time has come')
# print("s = ", s)
#
# for word in s:
#     print(word)
#
# print("*"*50)
# print(s[3])






# # -*- coding:utf-8 -*-
# class DataTest:
#     def __init__(self, id, address):
#         self.id = id
#         self.address = address
#         self.d = {self.id: 1,
#                   self.address: "192.168.1.1"
#                   }
#
#     def __getitem__(self, key):
#         return "hello"
#
#
# data = DataTest(1, "192.168.2.11")
# # print(len(data))  # TypeError: object of type 'DataTest' has no len()
# print(data[6])