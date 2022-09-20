# 快速入门

## 创建 DataFrame

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

## 检查属性

```python
>>> df.dtypes           # 各Series数据类型
temperature    int64    # Dataframe的每一列就是一个Series
activity       int64
dtype: object
>>> df.shape            # 形状
(5, 2)
```

## 增加 Series

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

## 选择和查询 DataFrame

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

## 查看统计量

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
