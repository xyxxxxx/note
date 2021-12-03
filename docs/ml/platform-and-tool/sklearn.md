

# sklearn.feature_extraction

## text

### CountVectorizer

见 NLP 工具。

### TfidfVectorizer

见 NLP 工具。

# sklearn.linear_model

### LogisticRegression

逻辑回归模型。

```python
>>> from sklearn.datasets import load_iris
>>> from sklearn.linear_model import LogisticRegression
>>> X, y = load_iris(return_X_y=True)   # Iris数据集,三分类任务
>>> clf = LogisticRegression(max_iter=1000).fit(X, y)  # 训练模型
>>> clf.predict(X[:2, :])          # 预测结果
array([0, 0])
>>> clf.predict_proba(X[:2, :])    # 预测结果(返回概率值)
array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
       [9.7...e-01, 2.8...e-02, ...e-08]])
>>> clf.score(X, y)                # 测试模型
0.9733333333333334
```

```python
LogisticRegression(penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
# penalty     正则化项
#   ='l1'     l1正则化,'liblinear'支持l1和l2
#   ='l2'     l2正则化,‘newton-cg’,‘sag’,‘lbfgs’仅支持l2

# C           正则化系数λ的倒数

# random_state 当优化器为'sag','saga'或'liblinear'时打乱数据
#             使用其它优化器则没有影响

# solver      优化器
#   ='newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
#   对于小型数据集，‘liblinear’是一个不错的选择，而’sag’和’saga’对于大型数据集更快
#   对于多分类问题，只有'newton-cg','sag','saga'和'lbfgs'可以选择multinomial;'liblinear'仅限于ovr

# max_iter    最大迭代次数,若仍未收敛则返回错误

# multi_class 多分类方法
#   ='ovr'    一对剩余(one-vs-rest)的二分类
#   ='multinomial'  多项逻辑回归,即softmax回归
```

# sklearn.metrics

### accuracy_score()

计算准确率。

```python
>>> from sklearn.metrics import accuracy_score
>>> y_pred = [0, 2, 1, 3]
>>> y_true = [0, 1, 2, 3]
>>> accuracy_score(y_true, y_pred)
0.5
```

### confusion_matrix()

根据实际结果和预测结果计算混淆矩阵。

见 precision_score（），recall_score（），f1_score（）。

### precision_score(), recall_score(), f1_score()

计算精确率、召回率、F1 值。

```python
>>> from sklearn.metrics import precision_score
>>> from sklearn.metrics import recall_score
>>> from sklearn.metrics import f1_score
>>> from sklearn.metrics import confusion_matrix
>>> y_true = [0, 0, 0, 0, 1, 1, 1, 1]   # 二分类
>>> y_pred = [0, 0, 1, 1, 0, 1, 1, 1]
>>> confusion_matrix(y_true, y_pred)
# pred  0  1
array([[2, 2],   # true  0
       [1, 3]])  #       1
>>> precision_score(y_true, y_pred)
0.6              # 这里认为1为阳性,0为阴性
>>> recall_score(y_true, y_pred)
0.75
>>> f1_score(y_true, y_pred)
0.6666666666666665
```

```python
>>> from sklearn.metrics import precision_score
>>> from sklearn.metrics import recall_score
>>> from sklearn.metrics import f1_score
>>> from sklearn.metrics import confusion_matrix
>>> y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]  # 多分类
>>> y_pred = [0, 0, 0, 0, 1, 1, 1, 1, 2]
>>> confusion_matrix(y_true, y_pred)
# pred  0  1  2
array([[3, 0, 0],   # true  0
       [1, 2, 0],   #       1
       [0, 2, 1]])  #       2
>>> precision_score(y_true, y_pred, average=None)
array([0.75, 0.5 , 1.  ])   # 分别对于pred(列) 0,1,2
>>> precision_score(y_true, y_pred, average='macro')
0.75                        # 宏平均
>>> precision_score(y_true, y_pred, average='micro')
0.6666666666666666          # 微平均
>>> recall_score(y_true, y_pred, average=None)
array([1.        , 0.66666667, 0.33333333])   # 分别对于true(行) 0,1,2
>>> recall_score(y_true, y_pred, average='macro')
0.6666666666666666
>>> recall_score(y_true, y_pred, average='micro')
0.6666666666666666
>>> f1_score(y_true, y_pred, average=None)
array([0.85714286, 0.57142857, 0.5       ])
>>> f1_score(y_true, y_pred, average='macro')
0.6428571428571429
>>> f1_score(y_true, y_pred, average='micro')
0.6666666666666666
```

# sklearn.model_selection

### train_test_split

将输入和输出 list 划分为训练集和测试集。

```python
>>> from sklearn.model_selection import train_test_split
>>> import numpy as np
>>> x = np.arange(10000).reshape((5000,2))
>>> x
array([[   0,    1],
       [   2,    3],
       [   4,    5],
       ...,
       [9994, 9995],
       [9996, 9997],
       [9998, 9999]])
>>> y = np.arange(5000)      # 回归问题
>>> y
array([   0,    1,    2, ..., 4997, 4998, 4999])
>>> x_train, x_test, y_train, y_test = train_test_split(
...   x, y, test_size=0.2)   # .8为训练集,.2为测试集
>>> x_train.shape
(4000, 2)
>>> y_train.shape
(4000,)
>>> x_train                  # 随机打乱顺序
array([[4058, 4059],
       [9646, 9647],
       [7560, 7561],
       ...,
       [1550, 1551],
       [9506, 9507],
       [6836, 6837]])
>>> y_train
array([2029, 4823, 3780, ...,  775, 4753, 3418])
```

```python
train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
# test_size    测试集的规模,若为0~1之间的浮点数则作为划分比例,若为正整数则作为绝对样本数,若为None则根据train_size推导,若train_size也为None则默认为0.25
# train_size   训练集的规模,若为0~1之间的浮点数则作为划分比例,若为正整数则作为绝对样本数,若为None则根据test_size推导
# random_state 随机数种子
```

# sklearn.naive_bayes

### MultinomialNB

见 NLP 工具。

# sklearn.preprocessing

### minmax_scale

线性归一化到[0，1]区间。

```python
>>> from sklearn.preprocessing import minmax_scale
>>> import numpy as np
>>> x = np.arange(11)
>>> np.random.shuffle(x)
>>> x
array([ 9,  7,  1,  4,  8, 10,  5,  3,  6,  2,  0])
>>> minmax_scale(x)
array([0.9, 0.7, 0.1, 0.4, 0.8, 1. , 0.5, 0.3, 0.6, 0.2, 0. ])
```

### MinMaxScaler

线性归一化到[0，1]区间的归一器。

```python
>>> from sklearn.preprocessing import MinMaxScaler
>>> import numpy as np
>>> x = np.arange(10).reshape(5,2)
>>> x
array([[0, 1],
       [2, 3],
       [4, 5],
       [6, 7],
       [8, 9]])
>>> scaler = MinMaxScaler()
>>> scaler.fit(x)   # 接受二维数组,计算每列的最大和最小值
MinMaxScaler()
>>> scaler.transform([[0,1],[4,5],[8,9]])   # 对每列应用相应的最大和最小值进行归一化
array([[0. , 0. ],
       [0.5, 0.5],
       [1. , 1. ]])
```

### scale

归一化到 01 正态分布（每个值减去均值，除以标准差）。

```python
>>> from sklearn.preprocessing import scale
>>> import numpy as np
>>> x = np.arange(11)
>>> np.random.shuffle(x)
>>> x
array([ 9,  7,  1,  4,  8, 10,  5,  3,  6,  2,  0])
>>> scale(x)
array([ 1.26491106,  0.63245553, -1.26491106, -0.31622777,  0.9486833 ,
        1.58113883,  0.        , -0.63245553,  0.31622777, -0.9486833 ,
       -1.58113883])
```

### StandardScaler

归一化到 01 正态分布的归一器。

```python
>>> from sklearn.preprocessing import StandardScaler
>>> import numpy as np
>>> x = np.arange(10).reshape(5,2)
>>> x
array([[0, 1],
       [2, 3],
       [4, 5],
       [6, 7],
       [8, 9]])
>>> scaler = StandardScaler()
>>> scaler.fit(x)   # 接受二维数组,计算每列的均值和标准差
StandardScaler()
>>> scaler.transform([[0,1],[4,5],[8,9]])   # 对每列应用相应的均值和标准差进行归一化
array([[-1.41421356, -1.41421356],
       [ 0.        ,  0.        ],
       [ 1.41421356,  1.41421356]])
```

# sklearn.svm

### SVC

支持向量机模型。参考 [SVM.ipynb](https://colab.research.google.com/drive/1qg5I5Jzrcjm930i9jkdDpcSnxYZ8riQe?usp=sharing)。

```python
>>> from sklearn.datasets import load_iris
>>> from sklearn.svm import SVC
>>> X, y = load_iris(return_X_y=True)   # Iris数据集,三分类任务

>>> clf = SVC(kernel='linear').fit(X, y)  # 线性核
>>> clf.predict(X[:2, :])                 # 预测结果
array([0, 0])
>>> clf.score(X, y)                       # 测试模型
0.9933333333333333

>>> clf = SVC(kernel='rbf', gamma=1).fit(X, y)  # 高斯核
>>> clf.predict(X[:2, :])
array([0, 0])
>>> clf.score(X, y)
0.98
```

```python
SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)
# C           正则化系数λ的倒数,使用l2正则化
# kernel      核函数
#   ='linear' 线性核
#   ='rbf'    高斯核
#   ='polynomial' 多项式核
#   ='sigmoid'    sigmoid核
# degree,gamma,coef0
#             参考https://scikit-learn.org/stable/modules/svm.html#kernel-functions中的公式

```

