# -*- coding: utf-8 -*-
"""
Created on Tue May  5 02:08:38 2020

@author: dell
"""
from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
raw_file = 'corpus.txt' #导入处理后的数据
corpus = open(raw_file)
sentence = [s.split() for s in corpus] #进行分词
print (sentence)

path = get_tmpfile("word2vec.model")
model = Word2Vec(sentence,sg=1,min_count=10,negative=3, sample=0.001, hs=1) #构建模型

model.save("word2vec.model")
m= Word2Vec.load('word2vec.model')

model['rice'] #导出该词的词向量

for key in model.similar_by_word('gene',topn=10):
    print(key)
m.most_similar(positive=['rice', 'gene'], negative=['cdna'])


print(m.wv.most_similar(['rice'],topn=3))