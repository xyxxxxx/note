Pandas是一种列存数据分析 API。它是用于处理和分析输入数据的强大工具，很多机器学习框架都支持将Pandas数据结构作为输入。

> tutorial参见[intro_to_pandas](https://colab.research.google.com/notebooks/mlcc/intro_to_pandas.ipynb)

## DateFrame & Series基本操作

```python
import numpy as np
import pandas as pd

# create and populate 5x2 NumPy array
my_data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])
# create Python list that holds the names of the two series
my_column_names = ['temperature', 'activity']
# create dataframe
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)
# or use dict to create
# my_dataframe = pd.DataFrame({'temperature':np.arange(0,50,10),'activity':[3,7,9,14,15]})

# print entire dataframe
print(my_dataframe)
#   temperature  activity
#0            0         3
#1           10         7
#2           20         9
#3           30        14
#4           40        15

# dataframe info
my_dataframe.dtypes          # data type of each series
my_dataframe.shape           # size of dataframe

# set index
print(my_dataframe.set_index('temperature')) # set series as index
print(my_dataframe.reset_index())            # reset

# create series
my_dataframe['adjusted'] = my_dataframe['activity'] + 2
print(my_dataframe)
# or
# adjusted = pd.Series([5,9,11,16,17], name='adjusted')
# my_dataframe['adjusted'] = adjusted

# select(query) dataframe
my_dataframe.head(3)         # row 0-2
my_dataframe[1:4]            # row 1-3

my_dataframe['temperature']  # series with given name
my_dataframe.temperature

my_dataframe.iloc[2]         # row 2
my_dataframe.iloc[2:4,1:]    # row 2-3, series 1-

my_dataframe["adjusted"]>10  # condition
# 0  False
# 1  False
# 2   True
# 3   True
# 4   True
my_dataframe[my_dataframe["adjusted"]>10]  # conditional select
#   temperature  activity  adjusted
#2           20         9        11
#3           30        14        16
#4           40        15        17
my_dataframe[my_dataframe["adjusted"].isin([15,16,17])]  # [15,16,17] is value set
my_dataframe[my_dataframe["adjusted"].notna()] # remove row with N/A value

# delete series
my_dataframe.pop('adjusted')
# or
df.drop(columns=['adjusted'])
```

### join & merge

```python
import pandas as pd

df = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],
                   'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})
other = pd.DataFrame({'key': ['K0', 'K1', 'K2'],
                      'B': ['B0', 'B1', 'B2']})
df.join(other, lsuffix='_caller', rsuffix='_other') # simple concatenate
# lsuffix and rsuffix distinguish series of same name
#   key_caller   A key_other    B
# 0         K0  A0        K0   B0
# 1         K1  A1        K1   B1
# 2         K2  A2        K2   B2
# 3         K3  A3       NaN  NaN
# 4         K4  A4       NaN  NaN
# 5         K5  A5       NaN  NaN

df.set_index('key').join(other.set_index('key'))
# set key as index
#       A    B
# key
# K0   A0   B0
# K1   A1   B1
# K2   A2   B2
# K3   A3  NaN
# K4   A4  NaN
# K5   A5  NaN

df.join(other.set_index('key'), on='key')
# set key as index of other
#   key   A    B
# 0  K0  A0   B0
# 1  K1  A1   B1
# 2  K2  A2   B2
# 3  K3  A3  NaN
# 4  K4  A4  NaN
# 5  K5  A5  NaN
```

```python
import pandas as pd

df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'],
                    'value': [1, 2, 3, 5]})
df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'],
                    'value': [5, 6, 7, 8]})
print(df1.merge(df2, left_on='lkey', right_on='rkey')) # database style join
# match 'lkey' in df1 and 'rkey' in df2
#   lkey  value_x rkey  value_y
# 0  foo        1  foo        5
# 1  foo        1  foo        8
# 2  foo        5  foo        5
# 3  foo        5  foo        8
# 4  bar        2  bar        6
# 5  baz        3  baz        7
```



## 查看统计量

```python
import numpy as np
import pandas as pd

titanic = pd.read_csv("data/titanic.csv")
titanic.head()
#    PassengerId  Survived  Pclass                                          Name ...
# 0            1         0       3                       Braund, Mr. Owen Harris 
# 1            2         1       1      Cumings, Mrs. John Bradley (Florence ... 
# 2            3         1       3                        Heikkinen, Miss. Laina 
# 3            4         1       1  Futrelle, Mrs. Jacques Heath (Lily May Peel) 
# 4            5         0       3                      Allen, Mr. William Henry 

# describe: basic statistics
titanic["Age", "Fare"].describe()
#               Age        Fare
# count  714.000000  891.000000
# mean    29.699118   32.204208
# std     14.526497   49.693429
# min      0.420000    0.000000
# 25%     20.125000    7.910400
# 50%     28.000000   14.454200
# 75%     38.000000   31.000000
# max     80.000000  512.329200

# single statistic: max, min, mean, median, std, 
titanic["Age", "Fare"].median()
# Age     28.0000
# Fare    14.4542

# grouped single statistic
titanic["Age"].groupby("Sex").mean()
# female    27.915709
# male      30.726645

# grouped count
titanic["sex"].groupby("Sex").count()
```



## 数据操作

### 基本运算: add, sub, mul, div, floordiv, mod, pow

```python
import numpy as np
import pandas as pd

df = pd.DataFrame([[1, 2],[3, 4],[5, 6]], columns=['A', 'B'])
print(df)
#    A  B
# 0  1  2
# 1  3  4
# 2  5  6

print(df + 1) # or df.add(1)
#    A  B
# 0  2  3
# 1  4  5
# 2  6  7
```

### append

```python
import pandas as pd
import numpy as np

df = pd.DataFrame(columns=['A', 'B', 'C']) # empty dataframe
for i in range(4):
    df.loc[i] = [np.random.randint(-1, 1) for n in range(3)]
print(df.append({'lib': 2, 'qty1': 3, 'qty2': 4}, ignore_index=True))
#   lib qty1 qty2
# 0  -1    0   -1
# 1   0    0    0
# 2  -1    0    0
# 3  -1    0    0
# 4   2    3    4
```

### drop

```python
import pandas as pd
df = pd.DataFrame({'Country': ['US', 'China', 'Japan', 'Germany'],
                   'GDP': [21.4, 14.3, 5.1, 3.8]},
                  )

print(df.drop([0,3])) # drop row 0 & 3
print(df)             # origin dataframe not changed
```

### N/A drop

```python
import pandas as pd
import numpy as np

df = pd.DataFrame([[1, 2, 5, 0],
                   [3, 4, np.nan, 1],
                   [np.nan, np.nan, np.nan, 5],
                   [np.nan, 3, np.nan, 4]],
                  columns=list('ABCD'))

print(df.dropna())
```

### N/A fill

```python
import pandas as pd
import numpy as np

df = pd.DataFrame([[np.nan, 2, np.nan, 0],
                   [3, 4, np.nan, 1],
                   [np.nan, np.nan, np.nan, 5],
                   [np.nan, 3, np.nan, 4]],
                  columns=list('ABCD'))
print(df)
#      A    B   C  D
# 0  NaN  2.0 NaN  0
# 1  3.0  4.0 NaN  1
# 2  NaN  NaN NaN  5
# 3  NaN  3.0 NaN  4

print(df.fillna(0)) # fill NaN
#     A   B   C   D
# 0   0.0 2.0 0.0 0
# 1   3.0 4.0 0.0 1
# 2   0.0 0.0 0.0 5
# 3   0.0 3.0 0.0 4

values = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
print(df.fillna(value=values)) # fill NaN by column
#     A   B   C   D
# 0   0.0 2.0 2.0 0
# 1   3.0 4.0 2.0 1
# 2   0.0 1.0 2.0 5
# 3   0.0 3.0 2.0 4
```

### sorting

```python
import pandas as pd

titanic = pd.read_csv("data/titanic.csv")

titanic.sort_values(by="Age").head()   # sort by series 'Age'
#      PassengerId  Survived  Pclass                             Name     Sex   Age ...
# 803          804         1       3  Thomas, Master. Assad Alexander    male  0.42   
# 755          756         1       2        Hamalainen, Master. Viljo    male  0.67   
# 644          645         1       3           Baclini, Miss. Eugenie  female  0.75   
# 469          470         1       3    Baclini, Miss. Helene Barbara  female  0.75   
# 78            79         1       2    Caldwell, Master. Alden Gates    male  0.83   

titanic.sort_values(by=['Pclass', 'Age'], ascending=False).head()  # sort by series 'Pclass', 'Age', descending order
#      PassengerId  Survived  Pclass                       Name     Sex   Age  SibSp ...
# 851          852         0       3        Svensson, Mr. Johan    male  74.0      0   
# 116          117         0       3       Connors, Mr. Patrick    male  70.5      0   
# 280          281         0       3           Duane, Mr. Frank    male  65.0      0   
# 483          484         1       3     Turkula, Mrs. (Hedwig)  female  63.0      0   
# 326          327         0       3  Nysveen, Mr. Johan Hansen    male  61.0      0   

```

### sampling

```python
# import large data as dataframe named dataset
train_dataset = dataset.sample(frac=0.8,random_state=0) # sampling by given ratio
test_dataset = dataset.drop(train_dataset.index)        # remove sampled row
```

### apply

```python
import numpy as np
import pandas as pd

df = pd.DataFrame([[1, 2],[3, 4],[5, 6]], columns=['A', 'B'])
#    A  B
# 0  1  2
# 1  3  4
# 2  5  6
print(df.apply(np.sum, axis=0))
# A     9
# B    12
# dtype: int64
print(df.apply(np.sum, axis=1))
# 0     3
# 1     7
# 2    11
# dtype: int64
print(df.apply(lambda x: [1, 2], axis=1, result_type='broadcast'))
#    A  B
# 0  1  2
# 1  1  2
# 2  1  2



```

### normalize

```python
import numpy as np
import pandas as pd
np.random.seed(1)
df_test = pd.DataFrame(np.random.randn(4,4)* 4 + 3)
print(df_test)
#           0         1         2         3
# 0  9.497381  0.552974  0.887313 -1.291874
# 1  6.461631 -6.206155  9.979247 -0.044828
# 2  4.276156  2.002518  8.848432 -5.240563
# 3  1.710331  1.463783  7.535078 -1.399565
print(df_test.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))))
#           0         1         2         3
# 0  1.000000  0.823413  0.000000  0.759986
# 1  0.610154  0.000000  1.000000  1.000000
# 2  0.329499  1.000000  0.875624  0.000000
# 3  0.000000  0.934370  0.731172  0.739260
```



## 导入数据

```python
# csv
titanic = pd.read_csv("data/titanic.csv")

# xlsx
titanic = pd.read_excel('titanic.xlsx', sheet_name='passengers')
```



## 绘图

```python
import numpy as np
import pandas as pd

air_quality = pd.read_csv("data/air_quality_no2.csv",
                           index_col=0, parse_dates=True)

air_quality.head()

#                      station_antwerp  station_paris  station_london
# datetime                                                           
# 2019-05-07 02:00:00              NaN            NaN            23.0
# 2019-05-07 03:00:00             50.5           25.0            19.0
# 2019-05-07 04:00:00             45.0           27.7            19.0

air_quality.plot()
air_quality["station_paris"].plot()
air_quality.plot.sca   0  345364  6.2375   NaN        Stter(x="station_london",
                          y="station_paris",
                          alpha=0.5)
air_quality.plot.box()
```



## 时间序列

```python
import numpy as np
import pandas as pd

air_quality = pd.read_csv("data/air_quality_no2_long.csv")

air_quality.head()
#     city country                   datetime location parameter  value   unit
# 0  Paris      FR  2019-06-21 00:00:00+00:00  FR04014       no2   20.0  µg/m³
# 1  Paris      FR  2019-06-20 23:00:00+00:00  FR04014       no2   21.8  µg/m³
# 2  Paris      FR  2019-06-20 22:00:00+00:00  FR04014       no2   26.5  µg/m³
# 3  Paris      FR  2019-06-20 21:00:00+00:00  FR04014       no2   24.9  µg/m³
# 4  Paris      FR  2019-06-20 20:00:00+00:00  FR04014       no2   21.4  µg/m³

air_quality["datetime"] = pd.to_datetime(air_quality["datetime"])
print(air_quality["datetime"])
# 0      2019-06-21 00:00:00+00:00
# 1      2019-06-20 23:00:00+00:00
# 2      2019-06-20 22:00:00+00:00
#                   ...           
# 2067   2019-05-07 01:00:00+00:00
# Name: datetime, Length: 2068, dtype: datetime64[ns, UTC]

print(air_quality["datetime"].max())
# Timestamp('2019-06-21 00:00:00+0000', tz='UTC')

print(air_quality["datetime"].max() - air_quality["datetime"].min())
# Timedelta('44 days 23:00:00')

print(air_quality["datetime"][0].dt.month)  # 6
```

