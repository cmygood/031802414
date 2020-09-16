#！／user/bin/env python
#-*- coding:utf-8 -*-
#author:M10
import jieba
from gensim import corpora,models,similarities
from collections import defaultdict
import sys



if __name__ == '__main__':
    #获取由命令行输入的三个参数
    oriPath = sys.argv[1]
    copyPath = sys.argv[2]
    ansPath = sys.argv[3]
    doc1 = oriPath
#doc2 = '/Users/wangxingfan/Desktop/doc2.txt'
# d1 = open(doc1).read()
# #d2 = open(doc2).read()
# data1 = jieba.cut(d1)
# #data2 = jieba.cut(d2)
    try:
        with open(oriPath, encoding='UTF-8') as fp:
            ori = fp.read()
        with open(copyPath, encoding='UTF-8') as fp:
            copy = fp.read()
    except:
        print("路径错误")
    data1 = jieba.cut(ori)
    list1 = []
    list2 = []
    list = []
    for i in data1:
      list1.append(i)
#for i in data2:
 #   list2.append(i)
    list = [list1]
    frequency = defaultdict(int)#如果键不存在则返回N/A,而不是报错,获取分词后词的个数
    for i in list:
        for j in i:
          frequency[j] +=1
#创建词典
    dictionary = corpora.Dictionary(list)
    # dictionary.save('d:\软工\第一次软工作业\dnm.txt')
    # doc3 = 'd:\软工\第一次软工作业\orig_0.8_add.txt'
    d3 = copy
    data3 = jieba.cut(d3)
    data31 = []
    for i in data3:
        data31.append(i)
    new_doc = data31
#稀疏向量.dictionary.doc2bow(doc)是把文档doc变成一个稀疏向量，[(0, 1), (1, 1)]，表明id为0,1的词汇出现了1次，至于其他词汇，没有出现。
    new_vec = dictionary.doc2bow(new_doc)
# #获取语料库
    corpus = [dictionary.doc2bow(i) for i in list]

    tfidf = models.TfidfModel(corpus)
    # tfidf_vector = tfidf[corpus]
    # query = jieba.cut(copy)
    # query_bow = dictionary.doc2bow(query)
    # index = similarities.MatrixSimilarity(tfidf_vector)
    # sim = index[query_bow]
#特征数
    featureNUM = len(dictionary.token2id.keys())
#通过TfIdf对整个语料库进行转换并将其编入索引，以准备相似性查询
    index = similarities.MatrixSimilarity(tfidf[corpus],num_features=featureNUM)
#计算向量相似度
    sim = index[tfidf[new_vec]]
    # print(sim)
    # tfidf = models.TfidfModel
    try:
        with open(ansPath, "w+", encoding='UTF-8') as fp:
            fp.write(str(sim))
    except:
        print("路径错误")
