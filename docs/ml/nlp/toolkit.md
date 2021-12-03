

# tokenization分词

## 中文

```python
>>> from ltp import LTP
>>> ltp = LTP()
>>> segment, _ = ltp.seg(["上海市气象台发布大风黄色预警。"])
>>> segment
[['上海市', '气象台', '发布', '大风', '黄色', '预警', '。']]
```

## English

```python
>>> from nltk.tokenize import word_tokenize
>>> sentence = """At eight o'clock on Thursday morning
... Arthur didn't feel very good."""
>>> tokens = word_tokenize(sentence)
>>> tokens
['At', 'eight', "o'clock", 'on', 'Thursday', 'morning', 'Arthur', 'did', "n't", 'feel', 'very', 'good', '.']
```

# stem词干提取

## English

```python
>>> from nltk.stem import SnowballStemmer
>>> stemmer = SnowballStemmer('english')
>>> stemmer.stem('distributing')
'distribut'
```

# 词性标注

## 中文

```python
>>> from ltp import LTP
>>> ltp = LTP()
>>> segment, hidden = ltp.seg(["上海市气象台发布大风黄色预警。"])
>>> pos=ltp.pos(hidden)
>>> pos
[['ns', 'n', 'v', 'n', 'n', 'v', 'wp']]
```

# vectorization向量化

## CountVectorizer

`CountVectorizer`将训练文本转换为每种词汇在该文本中出现的频率。

```python
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> corpus = [
...     'This is the first document.',
...     'This document is the second document.',
...     'And this is the third one.',
...     'Is this the first document?',
... ]

>>> vectorizer = CountVectorizer()
>>> X = vectorizer.fit_transform(corpus)      # 传入字符串列表
>>> vectorizer.get_feature_names()            # 每个词为一个feature
['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
>>> X.toarray()                               # 词频矩阵
array([[0, 1, 1, 1, 0, 0, 1, 0, 1],
       [0, 2, 0, 1, 0, 1, 1, 0, 1],
       [1, 0, 0, 1, 1, 0, 1, 1, 1],
       [0, 1, 1, 1, 0, 0, 1, 0, 1]])
>>> X                                         # 本身是稀疏矩阵类型
<4x9 sparse matrix of type '<class 'numpy.float64'>'
        with 21 stored elements in Compressed Sparse Row format>
    
>>> vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
                                              # 设定ngram范围
>>> X2 = vectorizer2.fit_transform(corpus)
>>> vectorizer2.get_feature_names()           # 每个bigram为一个feature
['and this', 'document is', 'first document', 'is the', 'is this',
'second document', 'the first', 'the second', 'the third', 'third one',
 'this document', 'this is', 'this the']
 >>> print(X2.toarray())
 [[0 0 1 1 0 0 1 0 0 0 0 1 0]
  [0 1 0 1 0 1 0 1 0 0 1 0 0]
  [1 0 0 1 0 0 0 0 1 1 0 1 0]
  [0 0 1 0 1 0 1 0 0 0 0 0 1]]
```

```python
CountVectorizer(*, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, stop_words=None, token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=<class 'numpy.int64'>)
# encoding       传入字符串的编码方式
# lowercase      大写转换为小写
# max_df         构建词汇表时忽略频率高于该阈值的词,即构建停用词
# min_df         构建词汇表时忽略频率低于该阈值的词
# max_features   允许的最大特征数量
# ngram_range    设定作为特征的ngram范围
# stop_words     停用词
#   ='english'   使用内置的英文停用词列表
#   =list        使用给定的停用词列表
# token_pattern  用来界定token(分词)的正则表达式
```

## TfidfVectorizer

`TfidfVectorizer`使用TF-IDF算法。参考[文档](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)。

```python
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> corpus = [
...     'This is the first document.',
...     'This document is the second document.',
...     'And this is the third one.',
...     'Is this the first document?',
... ]

>>> vectorizer = TfidfVectorizer()
>>> X = vectorizer.fit_transform(corpus)
>>> print(vectorizer.get_feature_names())
['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
>>> print(X.toarray())
[[0.         0.46979139 0.58028582 0.38408524 0.         0.
  0.38408524 0.         0.38408524]
 [0.         0.6876236  0.         0.28108867 0.         0.53864762
  0.28108867 0.         0.28108867]
 [0.51184851 0.         0.         0.26710379 0.51184851 0.
  0.26710379 0.51184851 0.26710379]
 [0.         0.46979139 0.58028582 0.38408524 0.         0.
  0.38408524 0.         0.38408524]]
```

参数与`CountVectorizer`类似。

## word2vec

参考gensim。

# 模型

## 朴素贝叶斯

### MultinomialNB

适用于离散特征的分类问题。例如下面的例子中，特征为100个词的词频（取值0~4）。

但实际上对于tf-idf等算法得到的折减频率亦有效。

```python
>>> import numpy as np
>>> rng = np.random.RandomState(1)
>>> X = rng.randint(5, size=(6, 100))
>>> y = np.array([1, 2, 3, 4, 5, 6])
>>> from sklearn.naive_bayes import MultinomialNB
>>> clf = MultinomialNB()
>>> clf.fit(X, y)
MultinomialNB()
>>> print(clf.predict(X[2:3]))
[3]
```

# 停用词

# 工具列表

参考https://github.com/crownpku/Awesome-Chinese-NLP

## 中文

+ [LTP 语言技术平台](https://github.com/HIT-SCIR/ltp) by 哈工大 (C++) [pylyp](https://github.com/HIT-SCIR/pyltp) LTP的python封装

+ [BaiduLac](https://github.com/baidu/lac) by 百度 Baidu's open-source lexical analysis tool for Chinese, including word segmentation, part-of-speech tagging & named entity recognition.

+ [FudanNLP](https://github.com/FudanNLP/fnlp) by 复旦 (Java)

  （最后更新于2018年）

+ [THULAC 中文词法分析工具包](http://thulac.thunlp.org/) by 清华 (C++/Java/Python)

  （最后更新于2017年）

## 英文

+ [CoreNLP](https://github.com/stanfordnlp/CoreNLP) by Stanford (Java) A Java suite of core NLP tools.
+ [Stanza](https://github.com/stanfordnlp/stanza) by Stanford (Python) A Python NLP Library for Many Human Languages
+ [NLTK](http://www.nltk.org/) (Python) Natural Language Toolkit
+ [spaCy](https://spacy.io/) (Python) Industrial-Strength Natural Language Processing with a [online course](https://course.spacy.io/)
+ [textacy](https://github.com/chartbeat-labs/textacy) (Python) NLP, before and after spaCy
+ [OpenNLP](https://opennlp.apache.org/) (Java) A machine learning based toolkit for the processing of natural language text.
+ [gensim](https://github.com/RaRe-Technologies/gensim) (Python) Gensim is a Python library for topic modelling, document indexing and similarity retrieval with large corpora.
+ [Kashgari](https://github.com/BrikerMan/Kashgari) - Simple and powerful NLP framework, build your state-of-art model in 5 minutes for named entity recognition (NER), part-of-speech tagging (PoS) and text classification tasks. Includes BERT and word2vec embedding.

## 多语言

+ [langid](https://github.com/saffsd/langid.py) 离线的语言识别工具

+ 

  