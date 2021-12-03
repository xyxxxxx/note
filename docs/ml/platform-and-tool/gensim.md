

# Tutorial

```python
# 示例语料库
>>> text_corpus = [
...     "Human machine interface for lab abc computer applications",
...     "A survey of user opinion of computer system response time",
...     "The EPS user interface management system",
...     "System and human system engineering testing of EPS",
...     "Relation of user perceived response time to error measurement",
...     "The generation of random binary unordered trees",
...     "The intersection graph of paths in trees",
...     "Graph minors IV Widths of trees and well quasi ordering",
...     "Graph minors A survey",
... ]

# 创建一个简单的停止词集合
>>> stoplist = set('for a of the and to in'.split(' '))
# 将每个文本小写,用空格划分,并过滤停止词
>>> texts = [[word for word in document.lower().split() if word not in stoplist]
...          for document in text_corpus]
>>> from pprint import pprint
>>> pprint(texts)
[['human', 'machine', 'interface', 'lab', 'abc', 'computer', 'applications'],
 ['survey', 'user', 'opinion', 'computer', 'system', 'response', 'time'],
 ['eps', 'user', 'interface', 'management', 'system'],
 ['system', 'human', 'system', 'engineering', 'testing', 'eps'],
 ['relation', 'user', 'perceived', 'response', 'time', 'error', 'measurement'],
 ['generation', 'random', 'binary', 'unordered', 'trees'],
 ['intersection', 'graph', 'paths', 'trees'],
 ['graph', 'minors', 'iv', 'widths', 'trees', 'well', 'quasi', 'ordering'],
 ['graph', 'minors', 'survey']]

# 计数词频
>>> from collections import defaultdict
>>> frequency = defaultdict(int)
>>> for text in texts:
...     for token in text:
...         frequency[token] += 1
... 
# 仅保留词频大于1的词
>>> processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
>>> pprint(processed_corpus)
[['human', 'interface', 'computer'],
 ['survey', 'user', 'computer', 'system', 'response', 'time'],
 ['eps', 'user', 'interface', 'system'],
 ['system', 'human', 'system', 'eps'],
 ['user', 'response', 'time'],
 ['trees'],
 ['graph', 'trees'],
 ['graph', 'minors', 'trees'],
 ['graph', 'minors', 'survey']]

# 为processed_corpus中的token建立索引
>>> from gensim import corpora
>>> dictionary = corpora.Dictionary(processed_corpus)
>>> print(dictionary)
Dictionary(12 unique tokens: ['computer', 'human', 'interface', 'response', 'survey']...)         # 共12个不同的token
>>> pprint(dictionary.token2id)   # 打印索引
{'computer': 0,
 'eps': 8,
 'graph': 10,
 'human': 1,
 'interface': 2,
 'minors': 11,
 'response': 3,
 'survey': 4,
 'system': 5,
 'time': 6,
 'trees': 9,
 'user': 7}

# 基于索引,将文本转换为词袋
>>> new_doc = "Human computer interaction"
>>> new_vec = dictionary.doc2bow(new_doc.lower().split())
>>> new_vec
[(0, 1), (1, 1)]
# 元组表示(索引,频率)
# 将语料库转换为词袋
>>> bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
>>> pprint(bow_corpus)
[[(0, 1), (1, 1), (2, 1)],
 [(0, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)],
 [(2, 1), (5, 1), (7, 1), (8, 1)],
 [(1, 1), (5, 2), (8, 1)],
 [(3, 1), (6, 1), (7, 1)],
 [(9, 1)],
 [(9, 1), (10, 1)],
 [(9, 1), (10, 1), (11, 1)],
 [(4, 1), (10, 1), (11, 1)]]

# 训练并使用tfidf模型
>>> from gensim import models
>>> tfidf = models.TfidfModel(bow_corpus)
>>> words = "system minors".lower().split()
>>> dictionary.doc2bow(words)
[(5, 1), (11, 1)]
>>> tfidf[dictionary.doc2bow(words)]
[(5, 0.5898341626740045), (11, 0.8075244024440723)]
# 元组表示(索引,tf-idf权重)

# 计算查询向量和各文本向量的相似度
>>> from gensim import similarities
>>> index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=12)
>>> query_document = 'system engineering'.split()
>>> query_bow = dictionary.doc2bow(query_document)
>>> query_bow
[(5, 1)]
>>> tfidf[query_bow]
[(5, 1.0)]
>>> sims = index[tfidf[query_bow]]
>>> list(enumerate(sims))
[(0, 0.0), (1, 0.32448703), (2, 0.41707572), (3, 0.7184812), (4, 0.0), (5, 0.0), (6, 0.0), (7, 0.0), (8, 0.0)]
```

### 语料库流Corpus Streaming – One Document at a Time

Gensim 的语料库可以使用任何可迭代对象，包括但不限于列表、numpy 数组、pandas dataframe 等，每次返回一个文本。因此我们不必将语料库一次读入内存，尤其是对于规模巨大的语料库而言。

```python
from gensim.test.utils import datapath
from gensim import utils

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""
    def __iter__(self):
        corpus_path = datapath('example.cor')
        for line in open(corpus_path):
            yield utils.simple_preprocess(line)
```

# genism.corpora

### Dictionary

为语料库中的 token 建立索引。

```python
>>> from gensim.corpora import Dictionary
>>> texts = [['human', 'interface', 'computer']]    # 简单的语料库
>>> dct = Dictionary(texts)
>>> dct.token2id                                    # 查看token索引
{'computer': 0, 'human': 1, 'interface': 2}         # 返回词典{str:int}

>>> dct.add_documents([["cat", "say", "meow"], ["dog", "say", "bark", "bark", "bark"]])   # 增加新的语料库
>>> dct.token2id
{'computer': 0, 'human': 1, 'interface': 2, 'cat': 3, 'meow': 4, 'say': 5, 'bark': 6, 'dog': 7}
>>> len(dct)
8

>>> dct.doc2bow(["dog", "computer", "non_existent_word"])  # 文本转词袋向量
[(0, 1), (7, 1)]
>>> dct.doc2idx(["dog", "computer", "non_existent_word"])  # token转索引
[7, 0, -1]                                                 # -1表示在词汇表之外

>>> dct.cfs
{1: 1, 2: 1, 0: 1, 3: 1, 5: 2, 4: 1, 7: 1, 6: 3}    # 各token的总频率(collection freq.)
>>> dct.dfs
{1: 1, 2: 1, 0: 1, 3: 1, 5: 2, 4: 1, 7: 1, 6: 1}    # 各token的文件频率(document freq.)
>>> dct.num_docs                                    # 共处理的文本数量
3
>>> dct.num_pos                                     # 共处理的token数量
11

# >>> dct.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)
# 仅保留出现频率不小于5次,不大于文本数0.5倍的次数,再保留频率前100000的token
# >>> dct.filter_n_most_frequent(2)
# 过滤掉频率前2的token
>>> dct.filter_tokens(bad_ids=[dct.token2id['say']])
# 删除指定索引的token
>>> dct.token2id
{'computer': 0, 'human': 1, 'interface': 2, 'cat': 3, 'meow': 4, 'bark': 5, 'dog': 6}
                                                    # 索引号对齐
    
# 保存和读取
>>> dct.save('dct1')                                # 保存为二进制文件
>>> loaded_dct = Dictionary.load('dct1')
>>> loaded_dct.token2id
>>> dct.save_as_text('dct2')
>>> loaded_dct = Dictionary.load_from_text('dct2')
>>> loaded_dct.cfs                                  # 丢失信息
{}
```

# gensim.models

## keyedvectors

### KeyedVectors

```python

```

## tfidfmodel

### TfidfModel

```python
>>> from gensim.models import TfidfModel
>>> from gensim.utils import simple_preprocess
>>> corpus = [
...     'This is the first document.',
...     'This document is the second document.',
...     'And this is the third one.',
...     'Is this the first document?',
... ]
>>> corpus_tokenized = [simple_preprocess(doc) for doc in corpus]
>>> corpus_tokenized
[['this', 'is', 'the', 'first', 'document'], ['this', 'document', 'is', 'the', 'second', 'document'], ['and', 'this', 'is', 'the', 'third', 'one'], ['is', 'this', 'the', 'first', 'document']]

>>> dct = Dictionary(corpus_tokenized)
>>> dct.token2id
{'document': 0, 'first': 1, 'is': 2, 'the': 3, 'this': 4, 'second': 5, 'and': 6, 'one': 7, 'third': 8}

>>> corpus_bow = [dct.doc2bow(doc) for doc in corpus_tokenized]
>>> pprint(corpus_bow)
[[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)],
 [(0, 2), (2, 1), (3, 1), (4, 1), (5, 1)],
 [(2, 1), (3, 1), (4, 1), (6, 1), (7, 1), (8, 1)],
 [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]]
>>> tfidf = TfidfModel(corpus_bow)                 # 接受一个元组(索引,频率)列表

>>> tfidf[corpus_bow[0]]                           # 训练完毕后可以作用于任何向量,
[(0, 0.383332888988391), (1, 0.9236102512530996)]  #    包括训练语料库
>> corpus_tfidf = tfidf[corpus_bow]
>>> for doc in corpus_tfidf:
...     print(doc)
... 
[(0, 0.383332888988391), (1, 0.9236102512530996)]
[(0, 0.383332888988391), (5, 0.9236102512530996)]
[(6, 0.5773502691896258), (7, 0.5773502691896258), (8, 0.5773502691896258)]
[(0, 0.383332888988391), (1, 0.9236102512530996)]
```

## word2vec

### Word2Vec

训练、使用和评价 word2vec 模型。

```python
>>> from gensim.utils import simple_preprocess
>>> from gensim.models import KeyedVectors, Word2Vec
>>> from gensim.test.utils import datapath
>>> class MyCorpus:
...     """An iterator that yields sentences (lists of str)."""
...     def __iter__(self):
...         corpus_path = datapath('lee_background.cor')
...         for line in open(corpus_path):
                # 语料库文件: 每一行为一个文本,token之间用空格分隔
...             yield simple_preprocess(line)
... 
>>> sentences = MyCorpus()
>>> model = Word2Vec(sentences=sentences)    # 训练模型

>>> model.wv['king']                         # wv指word vector
array([ 0.02872382,  0.03492628,  0.04034146, -0.07184649,  0.00168387,
        ...
        0.00480541,  0.01933789, -0.05256072,  0.00895547, -0.02865055],
      dtype=float32)

# 存储和读取
>>> model.save('word2vec1')                  # 存储模型
>>> model_loaded = Word2Vec.load('word2vec1')
>>> word_vectors = model.wv                  # 存储词向量(训练结果)
>>> word_vectors.save('wv1')
>>> wv = KeyedVectors.load('wv1', mmap='r')
>>> del model                                # 当不再需要模型状态,即训练结束时,
                                             # 可以从内存中释放该对象
    
>>> model.wv.similarity('king', 'man')       # 计算相似度
0.99777293
>>> model.wv.most_similar(positive=['car'], topn=5)   # 最相似的n个词
[('local', 0.9991850852966309), ('most', 0.9991558194160461), ('day', 0.9991497993469238), ('last', 0.9991437792778015), ('world', 0.9991411566734314)]

# 评价模型
#
```

```python
gensim.models.word2vec.Word2Vec(sentences=None, corpus_file=None, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=<built-in function hash>, epochs=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000, compute_loss=False, callbacks=(), comment=None, max_final_vocab=None)
# sentences       语料库,需要是一个可迭代对象
# size            嵌入维度
# alpha           初始学习率
# window          上下文词与目标词的最大距离
# min_count       忽略频率低于此参数的词
# max_vocab_size  最大词汇表规模
# workers         使用线程数
# min_alpha       最小学习率
# sg              1=skip gram, 其它=CBOW

```

使用预定义的 word2vec 模型：

```python
>>> import gensim.downloader
>>> print(list(gensim.downloader.info()['models'].keys()))  # 展示所有可用模型
['fasttext-wiki-news-subwords-300',
 'conceptnet-numberbatch-17-06-300',
 'word2vec-ruscorpora-300',
 'word2vec-google-news-300',
 'glove-wiki-gigaword-50',
 'glove-wiki-gigaword-100',
 'glove-wiki-gigaword-200',
 'glove-wiki-gigaword-300',
 'glove-twitter-25',
 'glove-twitter-50',
 'glove-twitter-100',
 'glove-twitter-200',
 '__testing_word2vec-matrix-synopsis']

>>> glove_vectors = gensim.downloader.load('glove-twitter-25')   # 下载并加载模型

>>> glove_vectors.most_similar('twitter')    # 即可正常使用
[('facebook', 0.948005199432373),
 ('tweet', 0.9403423070907593),
 ('fb', 0.9342358708381653),
 ('instagram', 0.9104824066162109),
 ('chat', 0.8964964747428894),
 ('hashtag', 0.8885937333106995),
 ('tweets', 0.8878158330917358),
 ('tl', 0.8778461217880249),
 ('link', 0.8778210878372192),
 ('internet', 0.8753897547721863)]
```

## 保存和读取

官方文档推荐的模型保存和读取方法。

```python
import tempfile

with tempfile.NamedTemporaryFile(prefix='gensim-model-', delete=False) as tmp:
    temporary_filepath = tmp.name          # like '/tmp/gensim-model-u1pbzss0'
    model.save(temporary_filepath)
    #
    # The model is now safely stored in the filepath.
    # You can copy it to other machines, share it with others, etc.
    #
    # To load a saved model:
    #                         for Word2Vec model
    new_model = gensim.models.Word2Vec.load(temporary_filepath)
```

# gensim.similarities

### MatrixSimilarity

计算查询向量和语料库各文本的余弦相似度，仅适用于内存占用较小的情形。

### Similarity

计算查询向量和语料库各文本的余弦相似度。

```python
>>> from gensim.models import TfidfModel
>>> from gensim.utils import simple_preprocess
>>> corpus = [
...     'This is the first document.',
...     'This document is the second document.',
...     'And this is the third one.',
...     'Is this the first document?',
... ]
>>> corpus_tokenized = [simple_preprocess(doc) for doc in corpus]
>>> dct = Dictionary(corpus_tokenized)
>>> corpus_bow = [dct.doc2bow(doc) for doc in corpus_tokenized]
>>> tfidf = TfidfModel(corpus_bow)

>>> from gensim import similarities
>>> index = similarities.SparseMatrixSimilarity(tfidf[corpus_bow], num_features=9)

# 存储和读取
>>> index.save('index1')
>>> index_loaded = similarities.SparseMatrixSimilarity.load('index1')

# 计算余弦相似度
>>> query_document = 'This is the second document'.split()
>>> query_bow = dct.doc2bow(query_document)
>>> sims = index[tfidf[query_bow]]
>>> list(enumerate(sims))
[(0, 0.07788932), (1, 0.9822325), (2, 0.0), (3, 0.07788932)]

# 将上述结果排序
>>> sims = sorted(enumerate(sims), key=lambda item: -item[1])
>>> for doc_position, doc_score in sims:
...     print(doc_score, corpus[doc_position])
... 
0.9822325 This document is the second document.
0.07788932 This is the first document.
0.07788932 Is this the first document?
0.0 And this is the third one.
```

### SparseMatrixSimilarity

计算查询向量和语料库各文本的余弦相似度，适用于稀疏向量（例如 tf-idf 向量）。

# gensim.utils

### simple_preprocess

将文本转换为小写的 token 列表。

```python
>>> from gensim.utils import simple_preprocess
>>> simple_preprocess('I LIKE IT')
['like', 'it']
```

```python
gensim.utils.simple_preprocess(doc, deacc=False, min_len=2, max_len=15)
# min_len    最小长度,小于该长度的token将被丢弃
# max_len    最大长度,大于该长度的token将被丢弃
```

### tokenize

迭代地 yield unicode 字符串形式的 token，可以去掉语调符号和小写。

```python
>>> from gensim.utils import tokenize
>>> list(tokenize('Nic nemůže letět rychlostí vyšší, než 300 tisíc kilometrů za sekundu!', deacc=True))
['Nic', 'nemuze', 'letet', 'rychlosti', 'vyssi', 'nez', 'tisic', 'kilometru', 'za', 'sekundu']
# 标点和数字被忽略
```

```python
gensim.utils.tokenize(text, lowercase=False, deacc=False, encoding='utf8', errors='strict', to_lower=False, lower=False)
# lowercase    小写
# deacc        去掉语调符号
```

