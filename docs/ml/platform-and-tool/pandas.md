# Pandas

[Pandas](https://pandas.pydata.org/) 是一个快速、强大、灵活且易用的开源数据分析和操作工具。很多机器学习框架都支持将 Pandas 数据结构作为输入。

!!! abstract "参考"
    * [User Guide](https://pandas.pydata.org/docs/user_guide/index.html#user-guide)
    * [API reference](https://pandas.pydata.org/docs/reference/index.html#api)

## 快速入门

### 创建 DataFrame

```python
# 使用numpy数组和字符串列表创建
>>> data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])
>>> columns = ['temperature', 'activity']
>>> df = pd.DataFrame(data=data, columns=columns)
>>> df
   temperature  activity
0            0         3
1           10         7
2           20         9
3           30        14
4           40        15

# 使用字典创建
>>> df = pd.DataFrame({
    'temperature': np.arange(0, 50, 10),
    'activity': [3, 7, 9, 14, 15]
})
>>> df
   temperature  activity
0            0         3
1           10         7
2           20         9
3           30        14
4           40        15
```

### 检查属性

```python
>>> df.dtypes           # 各Series数据类型
temperature    int64    # Dataframe的每一列就是一个Series
activity       int64
dtype: object
>>> df.shape            # 形状
(5, 2)
```


### 增加 Series

```python
# 基于已有的Series增加Series
>>> df['adjusted'] = df['activity'] + 2
>>> df
   temperature  activity  adjusted
0            0         3         5
1           10         7         9
2           20         9        11
3           30        14        16
4           40        15        17

# 增加新建的Series
>>> adjusted1 = pd.Series([5, 9, 11, 16, 17], name='adjusted1')
>>> df['adjusted1'] = adjusted1
>>> df
   temperature  activity  adjusted  adjusted1
0            0         3         5          5
1           10         7         9          9
2           20         9        11         11
3           30        14        16         16
4           40        15        17         17
```

### 选择和查询 DataFrame

```python
>>> df.head(3)     # 前3行
   temperature  activity  adjusted  adjusted1
0            0         3         5          5
1           10         7         9          9
2           20         9        11         11
>>> df[1:4]        # 行1-3
   temperature  activity  adjusted  adjusted1
1           10         7         9          9
2           20         9        11         11
3           30        14        16         16
>>> df.iloc[2]     # 行2
temperature    20
activity        9
adjusted       11
adjusted1      11
Name: 2, dtype: int64

>>> df['temperature']  # 指定Series
0     0
1    10
2    20
3    30
4    40
Name: temperature, dtype: int64
>>> df.temperature     # 指定Series
0     0
1    10
2    20
3    30
4    40
Name: temperature, dtype: int64
>>> df[['temperature', 'activity']]  # 指定多个Series
   temperature  activity
0            0         3
1           10         7
2           20         9
3           30        14
4           40        15

>>> df.iloc[2:4, 1:]   # 行2~3,列1~
   activity  adjusted  adjusted1
2         9        11         11
3        14        16         16

>>> df["adjusted"] > 10              # 条件查询,返回True或False
0    False
1    False
2     True
3     True
4     True
Name: adjusted, dtype: bool
>>> df["adjusted"].isin([15,16,17])  # 是否属于集合
0    False
1    False
2    False
3     True
4     True
Name: adjusted, dtype: bool
>>> df["adjusted"].notna()           # 是否为NaN
0    True
1    True
2    True
3    True
4    True
Name: adjusted, dtype: bool
        
>>> df[df["adjusted"] > 10]          # 条件选择
   temperature  activity  adjusted  adjusted1
2           20         9        11         11
3           30        14        16         16
4           40        15        17         17
```

### 查看统计量

```python
>>> df = pd.read_csv('https://raw.githubusercontent.com/jorisvandenbossche/pandas-tutorial/master/data/titanic.csv')
>>> df
     PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0              1         0       3  ...   7.2500   NaN         S
1              2         1       1  ...  71.2833   C85         C
2              3         1       3  ...   7.9250   NaN         S
3              4         1       1  ...  53.1000  C123         S
4              5         0       3  ...   8.0500   NaN         S
..           ...       ...     ...  ...      ...   ...       ...
886          887         0       2  ...  13.0000   NaN         S
887          888         1       1  ...  30.0000   B42         S
888          889         0       3  ...  23.4500   NaN         S
889          890         1       1  ...  30.0000  C148         C
890          891         0       3  ...   7.7500   NaN         Q

[891 rows x 12 columns]

>>> df.describe()            # 所有Series的基本统计量
       PassengerId    Survived      Pclass         Age       SibSp       Parch        Fare
count   891.000000  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean    446.000000    0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std     257.353842    0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
min       1.000000    0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%     223.500000    0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
50%     446.000000    0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%     668.500000    1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
max     891.000000    1.000000    3.000000   80.000000    8.000000    6.000000  512.329200

>>> df['Age'].describe()     # Series的基本统计量
count    714.000000
mean      29.699118
std       14.526497
min        0.420000
25%       20.125000
50%       28.000000
75%       38.000000
max       80.000000
Name: Age, dtype: float64

>>> df['Age'].max()          # Series的指定统计量
80.0

>>> df['Sex'].value_counts()     # Series计数
male      577
female    314
Name: Sex, dtype: int64

>>> df.groupby('Sex').mean()     # 按指定Series分组
        PassengerId  Survived    Pclass        Age     SibSp     Parch       Fare
Sex                                                                              
female   431.028662  0.742038  2.159236  27.915709  0.694268  0.649682  44.479818
male     454.147314  0.188908  2.389948  30.726645  0.429809  0.235702  25.523893
```

## API

### DataFrame

#### add(), sub(), mul(), div(), floordiv(), mod(), pow()

基本算术运算。符号 `+, -, *, /, //, %, **` 重载了这些方法。

```python
>>> df = pd.DataFrame([[1, 2],[3, 4],[5, 6]], columns=['A', 'B'])
>>> df
   A  B
0  1  2
1  3  4
2  5  6
>>> df + 1   # 或df.add(1)
   A  B
0  2  3
1  4  5
2  6  7
>>> df * 2   # 或df.mul(2)
    A   B
0   2   4
1   6   8
2  10  12
```

#### append()

增加一行。

```python
>>> df = pd.DataFrame([[1, 2],[3, 4],[5, 6]], columns=['A', 'B'])
>>> df.append({'A':7,'B':8}, ignore_index=True)
   A  B
0  1  2
1  3  4
2  5  6
3  7  8
>>> df.append({'A':7}, ignore_index=True)
     A    B
0  1.0  2.0
1  3.0  4.0
2  5.0  6.0
3  7.0  NaN
```

#### apply()

对所有元素应用函数。

```python
>>> df = pd.DataFrame([[1, 2],[3, 4],[5, 6]], columns=['A', 'B'])
>>> df
   A  B
0  1  2
1  3  4
2  5  6
>>> df.apply(lambda x: x**2)   # 作用于每个元素
    A   B
0   1   4
1   9  16
2  25  36
>>> df.apply(np.sum, axis=0)   # 作用于每列
A     9
B    12
dtype: int64
>>> df.apply(np.sum, axis=1)   # 作用于每行
0     3
1     7
2    11
dtype: int64
```

#### astype()

将 Series/DataFrame 转换为指定数据类型。

```python
>>> data = {'col1': [1, 2], 'col2': [3, 4]}
>>> df = pd.DataFrame(data=data)
>>> df.dtypes
col1    int64
col2    int64
dtype: object
>>> df.astype('int32').dtypes            # 所有列转换为int32类型
col1    int32
col2    int32
dtype: object
>>> df.astype({'col1': 'int32'}).dtypes  # 列col1转换为int32类型
col1    int32
col2    int64
dtype: object

>>> sr = pd.Series([1, 2], dtype='int32')
>>> sr.dtypes
dtype('int32')
>>> sr.astype('int64').dtypes
dtype('int64')
```

#### at()



```python

```

#### columns

返回 DataFrame 的列标签。

#### concat()

拼接 DataFrame。

```python
# Series相同的DataFrame的行拼接
>>> df1 = pd.DataFrame({'name': ['Alice', 'Bob', 'Cindy'],
                        'sex': ['F', 'M', 'F']})
>>> df1
    name sex
0  Alice   F
1    Bob   M
2  Cindy   F
>>> df2 = pd.DataFrame({'name': ['Dave', 'Elizabeth', 'Frank'],
                        'sex': ['M', 'F', 'M']})
>>> df2
        name sex
0       Dave   M
1  Elizabeth   F
2      Frank   M
>>> pd.concat([df1,df2])
        name sex
0      Alice   F
1        Bob   M
2      Cindy   F
0       Dave   M
1  Elizabeth   F
2      Frank   M
>>> pd.concat([df1,df2], ignore_index=True)  # 重新编号
        name sex
0      Alice   F
1        Bob   M
2      Cindy   F
3       Dave   M
4  Elizabeth   F
5      Frank   M
```

#### copy()

返回 Series/DataFrame 的深/浅拷贝。

```python
>>> s = pd.Series([1, 2], index=["a", "b"])
>>> deep = s.copy()                 # 深拷贝
>>> shallow = s.copy(deep=False)    # 浅拷贝
>>> s is deep
False
>>> s.values is deep.values or s.index is deep.index         # 不共享数据
False
>>> s is shallow
False
>>> s.values is shallow.values and s.index is shallow.index  # 共享数据
True
>>> s[0] = 3
>>> shallow[1] = 4
>>> s
a    3
b    4
dtype: int64
>>> shallow
a    3
b    4
dtype: int64
>>> deep
a    1
b    2
dtype: int64
```

#### drop()

删除指定行/列。

```python
>>> df = pd.DataFrame({'Country': ['US', 'China', 'Japan', 'Germany'],
...                    'GDP': [21.4, 14.3, 5.1, 3.8]})
>>> df.drop([0, 3])           # 删除多行
  Country   GDP
1   China  14.3
2   Japan   5.1
>>> df                        # 原DataFrame不变
   Country   GDP
0       US  21.4
1    China  14.3
2    Japan   5.1
3  Germany   3.8
>>> df.drop(columns=['GDP'])  # 删除列
   Country
0       US
1    China
2    Japan
3  Germany
```

#### dropna()

移除缺失值。

```python
>>> df = pd.DataFrame([[1, 2, 5, 0],
...                    [3, 4, np.nan, 1],
...                    [np.nan, np.nan, np.nan, 5],
...                    [np.nan, 3, np.nan, 4]],
...                    columns=list('ABCD'))
>>> df
     A    B    C  D
0  1.0  2.0  5.0  0
1  3.0  4.0  NaN  1
2  NaN  NaN  NaN  5
3  NaN  3.0  NaN  4
>>> df.dropna()                # 移除缺失至少一个值的行
     A    B    C  D
0  1.0  2.0  5.0  0
>>> df.dropna(axis='columns')  # 移除缺失至少一个值的列
   D
0  0
1  1
2  5
3  4
>>> df.dropna(how='all')       # 移除缺失所有值的行
     A    B    C  D
0  1.0  2.0  5.0  0
1  3.0  4.0  NaN  1
2  NaN  NaN  NaN  5
3  NaN  3.0  NaN  4
>>> df.dropna(thresh=2)        # 保留有至少两个值的行
     A    B    C  D
0  1.0  2.0  5.0  0
1  3.0  4.0  NaN  1
3  NaN  3.0  NaN  4
>>> df.dropna(subset=['A', 'B'])   # 仅检查指定列
     A    B    C  D
0  1.0  2.0  5.0  0
1  3.0  4.0  NaN  1
```

#### dtypes

返回 DataFrame 的（所有 Series 的）数据类型。

#### fillna()

替换 NaN。

```python
>>> df = pd.DataFrame([[1, 2, 5, 0],
...                    [3, 4, np.nan, 1],
...                    [np.nan, np.nan, np.nan, 5],
...                    [np.nan, 3, np.nan, 4]],
...                    columns=list('ABCD'))
>>> df
     A    B    C  D
0  1.0  2.0  5.0  0
1  3.0  4.0  NaN  1
2  NaN  NaN  NaN  5
3  NaN  3.0  NaN  4
>>> df.fillna(0)
     A    B    C  D
0  1.0  2.0  5.0  0
1  3.0  4.0  0.0  1
2  0.0  0.0  0.0  5
3  0.0  3.0  0.0  4
>>> values = {'A': 0, 'B': -1, 'C': -2, 'D': -3}
>>> df.fillna(value=values)  # 按列填充
     A    B    C  D
0  1.0  2.0  5.0  0
1  3.0  4.0 -2.0  1
2  0.0 -1.0 -2.0  5
3  0.0  3.0 -2.0  4
```

#### head(), tail()

查看 DataFrame 的前/后几行。

```python
>>> df = pd.DataFrame({'animal': ['alligator', 'bee', 'falcon', 'lion',
...                    'monkey', 'parrot', 'shark', 'whale', 'zebra']})
>>> df.head()      # 前5行
      animal
0  alligator
1        bee
2     falcon
3       lion
4     monkey
>>> df.head(3)     # 前3行
      animal
0  alligator
1        bee
2     falcon
>>> df.tail()      # 后5行
   animal
4  monkey
5  parrot
6   shark
7   whale
8   zebra
>>> df.tail(3)     # 后3行
  animal
6  shark
7  whale
8  zebra
```

#### iloc()

查看指定行/列。

```python
>>> mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
...           {'a': 100, 'b': 200, 'c': 300, 'd': 400},
...           {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000 }]
>>> df = pd.DataFrame(mydict)
>>> df
      a     b     c     d
0     1     2     3     4
1   100   200   300   400
2  1000  2000  3000  4000
>>> df.iloc[0]             # 单行
a    1
b    2
c    3
d    4
Name: 0, dtype: int64
>>> df.iloc[[0, 1]]        # 多行
     a    b    c    d
0    1    2    3    4
1  100  200  300  400
>>> df.iloc[1:]            # 行范围
      a     b     c     d
1   100   200   300   400
2  1000  2000  3000  4000

>>> df.iloc[0, 1]          # 单点
2
>>> df.iloc[[0, 2], [1, 3]]  # 多行列
      b     d
0     2     4
2  2000  4000
>>> df.iloc[1:3, 0:3]      # 行列范围
      a     b     c
1   100   200   300
2  1000  2000  3000
```

#### index

返回 DataFrame 的索引（行标签）。

#### info()

打印 DataFrame 的概要。

#### join()

与另一 DataFrame 做列拼接。

```python
>>> df1 = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],
                        'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})
>>> df2 = pd.DataFrame({'key': ['K0', 'K1', 'K2'],
                        'B': ['B0', 'B1', 'B2']})

>>> df1.join(df2, lsuffix='_caller', rsuffix='_other')  # 简单拼接,其中同名Series分别使用后缀
  key_caller   A key_other    B
0         K0  A0        K0   B0
1         K1  A1        K1   B1
2         K2  A2        K2   B2
3         K3  A3       NaN  NaN
4         K4  A4       NaN  NaN
5         K5  A5       NaN  NaN

>>> df1.set_index('key').join(df2.set_index('key'))     # 分别设置index再拼接
      A    B
key         
K0   A0   B0
K1   A1   B1
K2   A2   B2
K3   A3  NaN
K4   A4  NaN
K5   A5  NaN

>>> df1.join(df2.set_index('key'), on='key')            # 合并Series
  key   A    B
0  K0  A0   B0
1  K1  A1   B1
2  K2  A2   B2
3  K3  A3  NaN
4  K4  A4  NaN
5  K5  A5  NaN
```

#### memory()

返回 DataFrame 每一列的内存使用量，以字节为单位。

```python
>>> dtypes = ['int64', 'float64', 'complex128', 'object', 'bool']
>>> data = dict([(t, np.ones(shape=5000, dtype=int).astype(t))
...              for t in dtypes])
>>> df = pd.DataFrame(data)
>>> df.head()
   int64  float64  complex128 object  bool
0      1      1.0    1.0+0.0j      1  True
1      1      1.0    1.0+0.0j      1  True
2      1      1.0    1.0+0.0j      1  True
3      1      1.0    1.0+0.0j      1  True
4      1      1.0    1.0+0.0j      1  True
>>> df.memory_usage()
Index           128
int64         40000
float64       40000
complex128    80000
object        40000
bool           5000
dtype: int64
```

#### merge()

与另一 DataFrame 做数据库风格的列拼接，即匹配左键和右键。

```python
>>> df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'],
...                     'value': [1, 2, 3, 5]})
>>> df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'],
...                     'value': [5, 6, 7, 8]})
>>> df1.merge(df2, left_on='lkey', right_on='rkey')
  lkey  value_x rkey  value_y
0  foo        1  foo        5
1  foo        1  foo        8
2  foo        5  foo        5
3  foo        5  foo        8
4  bar        2  bar        6
5  baz        3  baz        7
```

#### ndim

若为 Series，返回 1；若为 DataFrame，返回 2。

#### plot()

对 DataFrame 或 Series 绘图。默认使用 matplotlib。

**散点图**

```python
>>> data = {'Unemployment_Rate': [6.1,5.8,5.7,5.7,5.8,5.6,5.5,5.3,5.2,5.2],
...         'Stock_Index_Price': [1500,1520,1525,1523,1515,1540,1545,1560,1555,1565]
...        }
>>> df = pd.DataFrame(data,columns=['Unemployment_Rate','Stock_Index_Price'])
>>> df.plot(kind = 'scatter', x ='Unemployment_Rate', y='Stock_Index_Price')
.show()<AxesSubplot:xlabel='Unemployment_Rate', ylabel='Stock_Index_Price'>
>>> plt.show()
```

![](https://datatofish.com/wp-content/uploads/2018/12/001_plot_df.png)

**折线图**

```python
>>> data = {'Year': [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010],
...         'Unemployment_Rate': [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3]
...        }
>>> df = pd.DataFrame(data,columns=['Year','Unemployment_Rate'])
', kind = 'line')
>>> df.plot(kind = 'line', x ='Year', y='Unemployment_Rate')
<AxesSubplot:xlabel='Year'>
>>> plt.show()
```

![](https://datatofish.com/wp-content/uploads/2018/12/002_plot_df.png)

**折线图**

```python
>>> data = {'Country': ['USA','Canada','Germany','UK','France'],
...         'GDP_Per_Capita': [45000,42000,52000,49000,47000]
...        }
>>> df = pd.DataFrame(data,columns=['Country','GDP_Per_Capita'])
>>> df.plot(kind = 'bar', x ='Country', y='GDP_Per_Capita')
<AxesSubplot:xlabel='Country'>
>>> plt.show()
```

![](https://datatofish.com/wp-content/uploads/2018/12/Capture_bar-1.jpg)

**扇形图**

```python
>>> data = {'Tasks': [300,500,700]}
>>> df = pd.DataFrame(data,columns=['Tasks'],index = ['Tasks Pending','Tasks Ongoing','Tasks Completed'])
>>> df
                 Tasks
Tasks Pending      300
Tasks Ongoing      500
Tasks Completed    700
>>> df.plot(kind='pie', y='Tasks', figsize=(5, 5), autopct='%1.1f%%', startangle=90, explode = [0.05,0.05,0.05], shadow=True)
<AxesSubplot:ylabel='Tasks'>
>>> plt.show()
# y: 选择DataFrame的Series
# autopct: 百分数显示格式
# startangle: 开始角度(从x轴正方向逆时针旋转),默认为0,逆时针划分

```

![](https://i.loli.net/2020/12/30/9V8sDLT7hjXJutQ.png)

#### pop()

删除 Series 并返回。

```python
>>> df = pd.DataFrame([('falcon', 'bird', 389.0),
...                    ('parrot', 'bird', 24.0),
...                    ('lion', 'mammal', 80.5),
...                    ('monkey', 'mammal', np.nan)],
...                   columns=('name', 'class', 'max_speed'))
>>> df
     name   class  max_speed
0  falcon    bird      389.0
1  parrot    bird       24.0
2    lion  mammal       80.5
3  monkey  mammal        NaN
>>> df.pop('class')
0      bird
1      bird
2    mammal
3    mammal
Name: class, dtype: object
>>> df
     name  max_speed
0  falcon      389.0
1  parrot       24.0
2    lion       80.5
3  monkey        NaN
```

#### rename()

重命名 Series。

```python
>>> df = pd.DataFrame([[1, 2],[3, 4],[5, 6]], columns=['A', 'B'])
>>> df.rename(columns={'A':'aaa','B':'bbb'})
   aaa  bbb
0    1    2
1    3    4
2    5    6
```

#### set_index(), reset_index()

使用既有的 Series 作为 index。重置 index。

```python
>>> df = pd.DataFrame({'month': [1, 4, 7, 10],
...                    'year': [2012, 2014, 2013, 2014],
...                    'sale': [55, 40, 84, 31]})
>>> df
   month  year  sale
0      1  2012    55
1      4  2014    40
2      7  2013    84
3     10  2014    31
>>> df.set_index('year')
      month  sale
year             
2012      1    55
2014      4    40
2013      7    84
2014     10    31
>>> df.set_index('year').reset_index()
   year  month  sale
0  2012      1    55
1  2014      4    40
2  2013      7    84
3  2014     10    31
```

#### size

返回 Series/DataFrame 的元素数量。

#### sort_values()

将各行根据指定 Series 排序。

```python
>>> df = pd.read_csv('https://raw.githubusercontent.com/jorisvandenbossche/pandas-tutorial/master/data/titanic.csv')

>>> df.sort_values(by='Age').head()   # 按Series 'Age' 
     PassengerId  Survived  Pclass                             Name     Sex   Age  ...
803          804         1       3  Thomas, Master. Assad Alexander    male  0.42  ...
755          756         1       2        Hamalainen, Master. Viljo    male  0.67  ...
644          645         1       3           Baclini, Miss. Eugenie  female  0.75  ...
469          470         1       3    Baclini, Miss. Helene Barbara  female  0.75  ...
78            79         1       2    Caldwell, Master. Alden Gates    male  0.83  ...

>>> df.sort_values(by=['Pclass', 'Age'], ascending=False).head()   # 按Series 'Pclass', 'Age' 分类,排降序
     PassengerId  Survived  Pclass                       Name     Sex   Age  ...
851          852         0       3        Svensson, Mr. Johan    male  74.0  ...
116          117         0       3       Connors, Mr. Patrick    male  70.5  ...
280          281         0       3           Duane, Mr. Frank    male  65.0  ...
483          484         1       3     Turkula, Mrs. (Hedwig)  female  63.0  ...
326          327         0       3  Nysveen, Mr. Johan Hansen    male  61.0  ...

```

#### sampling()

将 DataFrame 中的数据按比例做随机抽样。

```python
>>> df = pd.read_csv('https://raw.githubusercontent.com/jorisvandenbossche/pandas-tutorial/master/data/titanic.csv')

>>> train_df = df.sample(frac=0.8)  # 按比例随机抽样
>>> train_df
     PassengerId  Survived  Pclass                                         Name     Sex   ...
87            88         0       3                Slocovski, Mr. Selman Francis    male   ...
841          842         0       2                     Mudd, Mr. Thomas Charles    male   ...
..           ...       ...     ...                                          ...     ...   ...    ...    ...              ...       ...   ...      ...
352          353         0       3                           Elias, Mr. Tannous    male   ...

[713 rows x 12 columns]

>>> test_df = df.drop(train_df.index)  # 删除已取样的行
```

#### shape

返回以元组表示的 DataFrame 的形状。

```python
>>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4],
...                    'col3': [5, 6]})
>>> df.shape
(2, 3)
```

#### to_csv()

将 DataFrame 保存到 csv 文件。

```python

```

#### to_datetime()

#### values

返回 DataFrame 的 NumPy 表示。

```python
>>> df = pd.DataFrame({'age':    [ 3,  29],
...                    'height': [94, 170],
...                    'weight': [31, 115]})
>>> df
   age  height  weight
0    3      94      31
1   29     170     115
>>> df.dtypes
age       int64
height    int64
weight    int64
dtype: object
>>> df.values
array([[  3,  94,  31],
       [ 29, 170, 115]])

>>> df2 = pd.DataFrame([('parrot',   24.0, 'second'),
...                     ('lion',     80.5, 1),
...                     ('monkey', np.nan, None)],
...                   columns=('name', 'max_speed', 'rank'))
>>> df2
     name  max_speed    rank
0  parrot       24.0  second
1    lion       80.5       1
2  monkey        NaN    None
>>> df2.dtypes
name          object
max_speed    float64
rank          object
dtype: object
>>> df2.values
array([['parrot', 24.0, 'second'],
       ['lion', 80.5, 1],
       ['monkey', nan, None]], dtype=object)
```

### Series

#### apply()

对所有元素应用函数。

```python
>>> s = pd.Series([20, 21, 12],
...               index=['London', 'New York', 'Helsinki'])
>>> s
London      20
New York    21
Helsinki    12
dtype: int64
>>> s.apply(lambda x: x ** 2)
London      400
New York    441
Helsinki    144
dtype: int64
>>> s.apply(np.log)
London      2.995732
New York    3.044522
Helsinki    2.484907
dtype: float64
```

#### copy()

见 [DataFrame.copy](#copy)。

#### dropna()

见 [DataFrame.dropna](#dropna)。

#### ndim

见 [DataFrame.ndim](#ndim)。

#### size

见 [DataFrame.size](#size)。

### 导入数据

```python
# csv
>>> df = pd.read_csv("data/titanic.csv")  # 本地文件
>>> df = pd.read_csv('https://raw.githubusercontent.com/jorisvandenbossche/pandas-tutorial/master/data/titanic.csv')        # 在线下载

# xlsx
>>> df = pd.read_excel('titanic.xlsx', sheet_name='passengers')
```

