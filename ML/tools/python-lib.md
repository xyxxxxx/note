# NumPy

NumPy包用于向量和矩阵运算。

> tutorial参见[Quickstart tutorial](https://numpy.org/doc/stable/user/quickstart.html)

```python
import numpy as np

one_dimensional_array = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])
print(one_dimensional_array)

two_dimensional_array = np.array([[6, 5], [11, 7], [4, 8]])
print(two_dimensional_array)
#[[ 6  5]
# [11  7]
# [ 4  8]]

sequence_of_integers = np.arange(5, 12)
print(sequence_of_integers)
#[ 5  6  7  8  9 10 11]

import numpy as np
random_integers_between_50_and_100 = np.random.randint(low=50, high=101, size=(6))
print(random_integers_between_50_and_100)
#[59 77 94 60 97 92]

random_floats_between_0_and_1 = np.random.random([6])
print(random_floats_between_0_and_1) 
#[0.19546204 0.18011937 0.41153588 0.45157418 0.16954296 0.63709503]

#NumPy uses broadcasting to virtually expand the smaller operand to dimensions compatible for linear algebra
random_floats_between_2_and_3 = random_floats_between_0_and_1 + 2.0
print(random_floats_between_2_and_3)
#[2.19546204 2.18011937 2.41153588 2.45157418 2.16954296 2.63709503]

```





# Pandas

Pandas是一种列存数据分析 API。它是用于处理和分析输入数据的强大工具，很多机器学习框架都支持将Pandas数据结构作为输入。

tutorial参见[intro_to_pandas](https://colab.research.google.com/notebooks/mlcc/intro_to_pandas.ipynb)

```python
import numpy as np
import pandas as pd

# Create and populate a 5x2 NumPy array.
my_data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])

# Create a Python list that holds the names of the two columns.
my_column_names = ['temperature', 'activity']

# Create a DataFrame.
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)

# Print the entire DataFrame
print(my_dataframe)
#   temperature  activity
#0            0         3
#1           10         7
#2           20         9
#3           30        14
#4           40        15

# Create a new column named adjusted.
my_dataframe["adjusted"] = my_dataframe["activity"] + 2

# Print the entire DataFrame
print(my_dataframe)

print("Rows #0, #1, and #2:")
print(my_dataframe.head(3), '\n')

print("Row #2:")
print(my_dataframe.iloc[[2]], '\n')

print("Rows #1, #2, and #3:")
print(my_dataframe[1:4], '\n')

print("Column 'temperature':")
print(my_dataframe['temperature'])


```





# matplotlib