
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import random
from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec

 

#因为词向量文件比较大，全部可视化就什么都看不见了，所以随机抽取一些词可视化
model = Word2Vec.load("word2vec.model")
words = list(model.wv.vocab)
random.shuffle(words)
vector = model[words]
tsne = TSNE(n_components=2,init='pca',verbose=1)
embedd = tsne.fit_transform(vector)

 #可视化
plt.figure(figsize=(14,10))

plt.scatter(embedd[:215,0], embedd[:215,1])

for i in range(215):

    x = embedd[i][0]

    y = embedd[i][1]

    plt.text(x, y, words[i])

plt.show()
