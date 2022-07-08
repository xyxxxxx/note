[toc]

# Python 风格指南

!!! abstract "参考"
    * [PEP 8](https://www.python.org/dev/peps/pep-0008/)
    * PEP 257
    * [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

## 模块

### import

* 使用方法：
    * `import x`：导入包和模块，例如 `import torch`。
    * `from x import y`：从包中导入子包、模块或类，例如：
        * `from tensorflow import keras`
        * `from tensorflow.keras.datasets import mnist`
        * `from pytorch_lightning import LightningModule, LightningDataModule, Trainer`
    * `from x import y as z`：当导入两个同名为 `y` 的模块，或名称 `y` 非常长。
    * `import y as z`：仅当 `z` 是一个标准缩写，例如：
        * `import numpy as np`
        * `import torch.nn.functional as F`

  不要使用上述方法以外的方法，例如导入全部 `from x import *`。

* 不要使用相对路径导入，即使导入的模块在同一个包下，也使用完整的包名。

* 各 `import` 语句应分别占据一行：

  ```python
  # 正确的示范
  import os
  import sys
  from typing import Mapping, Sequence
  
  # 错误的示范
  import os, sys
  ```

* `import` 语句应总是放在文件的顶部，在模块的注释和 docstring 之后，全局变量和常量之前。

* `import ` 语句应按照包的类型分组，每个组内再按照字典序排序：

  ```python
  from __future__ import absolute_import   # `__future__`模块导入
  from __future__ import division
  from __future__ import print_function
  
  import collections                       # 标准库导入
  import logging
  import sys
  
  import torch                             # 第三方包导入
  from torch import ScriptModule, Tensor
  from torch.nn import Module
  from torch.optim.optimizer import Optimizer
  
                                           # 当前项目导入?
  ```

### 包

### 全局变量

一般应避免使用全局变量。但作为技术指标的全局变量是被允许和鼓励的，例如 `MAX_HOLY_HANDGRENADE_COUNT = 3`。

对外部隐藏全局变量应为变量名添加前缀 `_`，外部的访问需要通过公开的模块级别的函数完成。

## 类

### 属性

### 继承

尽量不要使用多重继承。

## 函数

### Lambda函数

对于非常简短的函数可以使用。如果lambda函数的长度超过80个字符，则应考虑使用一般的嵌套函数。

对于乘法这样的通常运算，使用`operator`模块中的函数而不要使用lambda函数，例如使用 `operator.mul` 而非 `lambda x, y: x * y`。

### 默认值参数

不要使用可变对象作为函数或方法定义的默认值。详见函数-参数-默认值参数。

## 控制

### 条件表达式

对于非常简短的判断可以使用。真值表达式、if语句、else语句，每个部分不得超过一行。若条件表达式过长，则应拆分为一般的条件语句。

## 容器

### 生成式

使用列表、字典、集合生成式的目的应是使代码简洁易读。映射表达式、for语句、if语句，每个部分不得超过一行。若生成式过长，或有多个for语句、if语句嵌套，则应拆分为一般的循环体。

## 命名

命名规则：

| Type        | Public               | Internal                          |
| ----------- | -------------------- | --------------------------------- |
| 包          | `lower_with_under`   |                                   |
| 模块        | `lower_with_under`   | `_lower_with_under`               |
| 类          | `CapWords`           | `_CapWords`                       |
| 异常        | `CapWords`           |                                   |
| 函数        | `lower_with_under()` | `_lower_with_under()`             |
| 全局/类常量 | `CAPS_WITH_UNDER`    | `_CAPS_WITH_UNDER`                |
| 全局/类变量 | `lower_with_under`   | `_lower_with_under`               |
| 实例变量    | `lower_with_under`   | `_lower_with_under` (protected)   |
| 方法        | `lower_with_under()` | `_lower_with_under()` (protected) |
| 函数参数    | `lower_with_under`   |                                   |
| 局部变量    | `lower_with_under`   |                                   |

函数名、变量名和模块名应该是描述性的，避免使用缩写，尤其是可能引起歧义或用户不熟悉的缩写，也不要通过在单词中删减字母的方式简略。

### 应避免的名称

* 单字符名称，除非一些特殊情况：
    * 计数或迭代变量（例如 `i`, `j`, `k`, `v` 等）
    * `e` 作为 `try/except` 语句中的异常标识符
    * `f` 作为 `with` 语句中的文件操作符
* 名称中使用横线 `-` 。
* 形如 `__my__` 的特殊属性名称（由 Python 保留）
* 冒犯性的名称

### 惯例

* “内部”指代模块的内部，或类内部的保护或私有属性。
* 使用单下划线前缀将提供一些保护模块变量和函数的支持。使用双下划线前缀将使得变量和函数为其类完全私有，但我们不鼓励使用它，因为它会影响代码的可读性和可测试性，而且并不是真正的*私有*。
* 将关联的类和顶级函数放置在一个模块中。不同于 Java，Python 没有限制一个模块只能有一个类。

## 类型

### 真值判断

## 字符串

* 使用 f-string，即 `%` 操作符或 `format()` 方法来格式化字符串，即使所有的参数都是字符串；使用 `+` 来拼接多个字符串；但介于其中的情形则需要基于你自己的判断，来决定使用格式化字符串还是拼接字符串。

  ```python
  # 正确的示范
  x = a + b                                         # 拼接字符串
  x = '%s, %s!' % (imperative, expletive)           # 格式化字符串
  x = '{}, {}'.format(first, second)
  x = 'name: %s; score: %d' % (name, n)
  x = 'name: {}; score: {}'.format(name, n)
  x = f'name: {name}; score: {n}'
  ```

  ```python
  # 错误的示范
  x = '%s%s' % (a, b)                               # use + in this case
  x = '{}{}'.format(a, b)                           # use + in this case
  x = first + ', ' + second
  x = 'name: ' + name + '; score: ' + str(n)
  ```

* 不要使用 `+` 或 `+=` 操作符在循环体中累积一个字符串。在一些情况下，这些累积一个字符串会导致平方的而非线形的运行时间。尽管 CPython 可能会对这种常见的累积方式进行优化，但这是一个实现细节的问题，在什么条件下会应用优化依然是不确定的。正确的方法是将所有子串添加到一个列表中，并在循环结束时调用 `''.join()`，或将每一个子串写入到 `io.StringIO` 缓冲区。这些方法恒定具有线性的运行时间复杂度。

  ```python
  # 正确的示范
  items = ['<table>']
  for last_name, first_name in employee_list:
      items.append('<tr><td>%s, %s</td></tr>' % (last_name, first_name))
  items.append('</table>')
  employee_table = ''.join(items)
  ```

  ```python
  # 错误的示范
  employee_table = '<table>'
  for last_name, first_name in employee_list:
      employee_table += '<tr><td>%s, %s</td></tr>' % (last_name, first_name)
  employee_table += '</table>'
  ```

* 在一个模块内，选择使用 `'` 或 `"` 引用字符串并坚持这一选择以保持格式一致。当字符串中出现某种引号时，可以使用另一种引号进行引用以避免使用转义符号。

  ```python
  # 正确的示范
  Python('Why are you hiding your eyes?')
  Gollum("I'm scared of lint errors.")
  Narrator('"Good!" thought a happy Python reviewer.')
  ```

* 多行字符串建议使用 `"""` 而非 `'''`。项目可以为非 docstring 的多行字符串使用 `'''`，当且仅当它们为一般字符串使用 `'`。docstring 必须使用 `"""`。

* 将多行字符串的缩进与代码保持一致可能会引入额外的空格，解决方法是用 `+` 拼接单行字符串、用圆括号隐式拼接单行字符串或用 `textwrap.dedent()` 去除多行字符串中每一行的空格前缀：

  ```python
  # 正确的示范
  >>> long_string = """This is fine if your use case can accept
      extraneous leading spaces."""
  >>> long_string
  'This is fine if your use case can accept\n    extraneous leading spaces.'   # 引入4个空格
  >>> long_string = ("And this is fine if you cannot accept\n" +
                  "extraneous leading spaces.")
  >>> long_string
  'And this is fine if you cannot accept\nextraneous leading spaces.'
  >>> long_string = ("And this too is fine if you cannot accept\n"
                 "extraneous leading spaces.")
  >>> long_string
  'And this too is fine if you cannot accept\nextraneous leading spaces.
  >>> import textwrap
  >>> long_string = textwrap.dedent("""\
      This is also fine, because textwrap.dedent()
      will collapse common leading spaces in each line.""")
  >>> long_string
  'This is also fine, because textwrap.dedent()\nwill collapse common leading spaces in each line.'
  
  # 错误的示范
  >>> long_string = """This is pretty ugly.
  Don't do this."""
  ```

### 日志

调用日志函数时，使用模式字符串（即包含 `%` 占位符）（而非 f-string）作为第一个参数，模式参数作为后续的参数。

```python
# 正确的示范
import tensorflow as tf
logger = tf.get_logger()
logger.info('TensorFlow Version is: %s', tf.__version__)
```

```python
# 正确的示范
import os
from absl import logging

logging.info('Current $PAGER is: %s', os.getenv('PAGER', default=''))

homedir = os.getenv('HOME')
if homedir is None or not os.access(homedir, os.W_OK):
logging.error('Cannot write to home directory, $HOME=%r', homedir)
```

### 错误信息

错误信息应遵守下列3条规则：

1. 信息应准确匹配实际错误条件
2. 插入的片段应总是可以清晰辨识
3. 应当允许简单的自动化处理

```python
# 正确的示范

raise TypeError('Iteration over a 0-d tensor')                              # 单行字符串

raise TypeError('Expected Operation, Variable, or Tensor, got ' + str(x))   # 拼接字符串

raise RuntimeError(                                                         # 拼接字符串
    'Script and require gradients is not supported at the moment.'
    'If you just want to do the forward, use .detach()'
    'on the input before calling the function.'
)

raise AttributeError(                                                       # 拼接字符串, f-string
    "Can't get __cuda_array_interface__ on non-CUDA tensor type: %s '
    "If CUDA data is required use tensor.cuda() to copy tensor to device memory." %
    self.type()
)

raise RuntimeError(
    "Only 2D tensors can be converted to the CSR format but got shape: ", shape)   # 多个字符串
```

## 格式

### 缩进

* 每个缩进层次为4个空格（而非一个tab`\t`）

* 不要使用tab；对于任何编辑器，将tab键设定为输入4个空格

* 对于拆分到多行的代码，应竖直对齐包装的所有元素，或使用4个空格的缩进并且第一行的括号后直接换行：

  ```python
  # 正确的示范
  # 竖直对齐元素
  foo = long_function_name(var_one, var_two,
                           var_three, var_four)
  meal = (spam,
          beans)
  
  # 字典中竖直对齐元素
  foo = {
      long_dictionary_key: value1 +
                           value2,
      ...
  }
  
  # 悬挂缩进4个空格;第一行括号后直接换行
  foo = long_function_name(
      var_one, var_two, var_three,
      var_four)
  meal = (
      spam,
      beans)
  
  # 字典中悬挂缩进4个空格
  foo = {
      long_dictionary_key:
          long_dictionary_value,
      ...
  }
  ```

  ```python
  # 错误的示范
  # 第一行有元素再换行悬挂缩进
  foo = long_function_name(var_one, var_two,
      var_three, var_four)
  meal = (spam,
      beans)
  
  # 悬挂缩进2个空格
  foo = long_function_name(
    var_one, var_two, var_three,
    var_four)
  
  # 字典中没有悬挂缩进
  foo = {
      long_dictionary_key:
      long_dictionary_value,
      ...
  }
  ```

### whitespace

### 行长度

* 一行代码的最大长度应为80个字符。

* 超过80个字符限制的常见例外情况包括：

    * 长的 `import` 语句
    * 注释中的 URL，路径名和长标记
    * 长的模块级别的字符串常量，因为不包含 whitespace 而不便于拆分到多行，例如 URL 或路径名

  ```python
  # 正确的示范
  # See details at
  # http://www.example.com/us/developer/documentation/api/content/v2.0/csv_file_name_extension_full_specification.html
  
  # 错误的示范
  # See details at
  # http://www.example.com/us/developer/documentation/api/content/\
  # v2.0/csv_file_name_extension_full_specification.html
  ```

* 利用 Python 隐式拼接括号内各行的特性，必要时可以使用圆括号包围表达式

  ```python
  # 正确的示范
  # 隐式拼接括号内各行
  foo_bar(self, width, height, color='black', design=None, x='foo',
          emphasis=None, highlight=0)
  
  # 使用圆括号包围表达式
  if (width == 0 and height == 0 and
      color == 'red' and emphasis == 'strong'):
  ```

* 当字符串在一行中容纳不下时，使用圆括号隐式拼接

  ```python
  # 正确的示范
  x = ('This will build a very long long '
       'long long long long long long string')
  
  url = ('http://www.example.com/us/developer/documentation/api/content/v2.0'
         '/csv_file_name_extension_full_specification.html')
  ```

* 不要使用反斜线拆分行，除非是 `with` 语句需要3个或更多的上下文管理器

  ```python
  # 正确的示范
  # 3个上下文管理器
  with very_long_first_expression_function() as spam, \
       very_long_second_expression_function() as beans, \
       third_thing() as eggs:
      place_order(eggs, beans, spam, beans)
  
  # 2个上下文管理器    
  with very_long_first_expression_function() as spam:
      with very_long_second_expression_function() as beans:
          place_order(beans, spam)    
  ```

  ```python
  # 错误的示范
  with very_long_first_expression_function() as spam, \
       very_long_second_expression_function() as beans, \
      place_order(beans, spam)
  ```

* 对于一行超过80个字符的所有其它情况，若 yapf 自动格式化器不能帮助拆分行，则该行可以超过限制。

### 圆括号

* 控制圆括号的使用。
* 可以对元组使用圆括号包围，但这不是必须的。

### 逗号

* 对于拆分到多行的代码，如果反括号与最后一个元素不在一行，则应在最后一个元素之后增加一个逗号。

  ```python
  # 正确的示范
  golomb4 = [
      0,
      1,
      4,
      6,
  ]
  
  # 错误的示范
  golomb4 = [
      0,
      1,
      4,
      6
  ]
  ```

## 文档字符串

Python 使用文档字符串 docstring 来为代码生成文档。一个 docstring 是包、模块、类、方法、函数的第一个声明字符串，这些字符串可以通过对象的 `__doc__` 属性自动提取，也被 `pydoc` 使用。所有公开的包、模块、类、方法、函数都应该有 docstring。docstring 的内容用一对 `"""` 包围。

docstring 应当被组织为：一行总结（不超过80个字符），以句号结尾；空一行，从第一行的第一个引号的位置开始，

* 函数和方法的文档字符串应当描述其功能、输入参数、返回值；如果有复杂的算法和实现，也需要写清楚或给出参考文献
* 对于多行的文档字符串，结束 `"""` 独占一行；对于单行的文档字符串，将开始和结束 `"""` 放在同一行

### Google Style

#### 模块

* 每一个文件都应当包含许可的模版内容，为项目选择合适的许可模板（例如 Apache 2.0，BSD，LGPL，GPL）。

* 文件应以一个 docstring 开始，描述模块的内容和使用方法。

  ```python
  """A one line summary of the module or program, terminated by a period.
  
  Leave one blank line.  The rest of this docstring should contain an
  overall description of the module or program.  Optionally, it may also
  contain a brief description of exported classes and functions and/or usage
  examples.
  
    Typical usage example:
  
    foo = ClassFoo()
    bar = foo.FunctionBar()
  """
  ```

  ```python
  # 来自TensorFlow的示例:
  
  """Adam optimizer implementation."""
  
  """TensorFlow-related utilities."""
  
  # 来自Lightning的示例:
  
  """Trainer to automate the training."""
  ```

#### 函数和方法

一个函数（或方法、生成器）必须有一个 docstring，除非满足下列所有条件：

* 外部不可见
* 非常短
* 功能显而易见

docstring 应当给出足够的信息，使调用函数无需阅读函数的代码。docstring 应当是记叙而非命令的风格（即动词使用单数第三人称形式），除了 `@property` 修饰的函数（遵循属性的风格）。docstring 应当描述函数的调用语法和它的语义，而非它的实现；对于复杂的实现，在代码附近注释比使用 docstring 更加合适。

重载基类方法的方法可能会有一个简单的 docstring 将读者指引到被重载的方法的 docstring，例如 `"""See base class."""`。其根本原因是没有必要将基类的 docstring 在文档中再多次重复。但是如果方法与被重载的方法有显著差异，或者有更多细节内容需要提供，那么需要 docstring 至少说明这些差异。

函数的某些部分应当在专门的小节中描述，列举如下。每个小节以标题行开始，跟随一个冒号并换行，小节的其余部分以标题为基准使用2个或4个空格的悬挂缩进（与模块内的其它 docstring 保持一致）。如果函数非常简单并且函数名和一行的 docstring 已经就可以给到充分的信息，这些小节可以省略。

* `Examples:`

  使用示例。对于模块中比较重要或者常用的 API 应首先给出。

* `Args:`

  列举每一个参数的名称，并跟随一段描述，中间用一个冒号和空格（或换行）分隔。如果描述长度超过80个字符，以参数名称为基准使用2个或4个空格的悬挂缩进（与模块内的其它 docstring 保持一致）。如果函数的形参列表不包含类型提示，描述应包含接受的实参类型；如果接受可变参数 `*foo` 和 `**bar`，就按原样列举 `*foo` 和 `**bar`。

* `Returns:` 或 `Yields:`（对于生成器）：

  描述返回值的类型和语义。如果函数返回 `None`，则省略这一小节；如果函数的 docstring 以“Returns”或“Yields”起首并且一句话就能够充分描述返回值，也可以省略这一小节。

* `Raises:`

  列举所有可能引发的异常，并跟随一段描述，具体格式与 `Args:` 部分相同。这一部分不应该包含不遵照 docstring 使用而可能引发的异常。

```python
def fetch_smalltable_rows(table_handle: smalltable.Table,         # 接受实参类型(仅起到提示作用,没有类型检查)
                          keys: Sequence[Union[bytes, str]],
                          require_all_keys: bool = False,
) -> Mapping[bytes, Tuple[str]]:                                  # 返回值类型
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        table_handle: An open smalltable.Table instance.          # 参数名称: 描述.
        keys: A sequence of strings representing the key of each table
          row to fetch.  String keys will be UTF-8 encoded.
        require_all_keys: Optional; If require_all_keys is True only
          rows with values set for all keys will be returned.

    Returns:
        A dict mapping keys to the corresponding table row data   # 返回值描述
        fetched. Each row is represented as a tuple of strings. For
        example:

        {b'Serak': ('Rigel VII', 'Preparer'),                     # 示例
         b'Zim': ('Irk', 'Invader'),
         b'Lrrr': ('Omicron Persei 8', 'Emperor')}

        Returned keys are always bytes.  If a key from the keys argument is   # 补充说明
        missing from the dictionary, then that row was not found in the
        table (and require_all_keys must have been False).

    Raises:
        IOError: An error occurred accessing the smalltable.      # 异常: 描述.
    """
```

```python
# 示例: `Args`的描述长度超过80个字符而使用悬挂缩进
def fetch_smalltable_rows(table_handle: smalltable.Table,
                          keys: Sequence[Union[bytes, str]],
                          require_all_keys: bool = False,
) -> Mapping[bytes, Tuple[str]]:
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
      table_handle:                                               # 统一缩进2个空格
        An open smalltable.Table instance.
      keys:
        A sequence of strings representing the key of each table row to
        fetch.  String keys will be UTF-8 encoded.
      require_all_keys:
        Optional; If require_all_keys is True only rows with values set
        for all keys will be returned.

    Returns:
      A dict mapping keys to the corresponding table row data
      fetched. Each row is represented as a tuple of strings. For
      example:

      {b'Serak': ('Rigel VII', 'Preparer'),
       b'Zim': ('Irk', 'Invader'),
       b'Lrrr': ('Omicron Persei 8', 'Emperor')}

      Returned keys are always bytes.  If a key from the keys argument is
      missing from the dictionary, then that row was not found in the
      table (and require_all_keys must have been False).

    Raises:
      IOError: An error occurred accessing the smalltable.
    """
```

#### 类

类应当有一个 docstring 描述之。类的某些部分应当在专门的小节中描述，列举如下。

* `Examples:`

  使用示例。对于模块中比较重要或者常用的公开的类可以给出。

* `Attributes:`

  列举每一个公开的实例属性的名称，并跟随一段描述，格式与函数的 `Args` 小节相同。

  > 用法混乱：有用 `Attributes` 描述类属性，也有用 `Attributes` 描述实例属性。

* `Notes:`

  注意事项。

* `Reference:`

  参考资料，通常是官方文档、算法论文等。

* `See also:`

  另见使用说明，通常是用户文档、使用示例等。

```python
class SampleClass:
    """Summary of class here.

    Longer class information....
    Longer class information....

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """

    def __init__(self, likes_spam=False):
        """Inits SampleClass with blah."""
        self.likes_spam = likes_spam
        self.eggs = 0

    def public_method(self):
        """Performs operation blah."""
```

```python
# 示例: `Attributes`描述@property包装的实例保护属性
# 来自https://github.com/tensorflow/tensorflow/blob/5dcfc51118817f27fad5246812d83e5dccdc5f72/tensorflow/examples/speech_commands/recognize_commands.py
class RecognizeResult(object):
  """Save recognition result temporarily.
  
  Attributes:
    founded_command: A string indicating the word just founded. Default value
      is '_silence_'                           # 将property对象作为实例公开属性描述
    score: An float representing the confidence of founded word. Default
      value is zero.
    is_new_command: A boolean indicating if the founded command is a new one
      against the last one. Default value is False.
  """

  def __init__(self):
    self._founded_command = "_silence_"        # 实例保护属性
    self._score = 0
    self._is_new_command = False

  @property
  def founded_command(self):                   # 包装为property对象
    return self._founded_command               # 省略docstring,因为类的docstring中已经记载此属性

  @founded_command.setter
  def founded_command(self, value):
    self._founded_command = value

  @property
  def score(self):
    return self._score

  @score.setter
  def score(self, value):
    self._score = value

  @property
  def is_new_command(self):
    return self._is_new_command

  @is_new_command.setter
  def is_new_command(self, value):
    self._is_new_command = value
```

```python
# 示例: `Attributes`描述实例保护属性
# 来自https://github.com/tensorflow/tensorflow/blob/5dcfc51118817f27fad5246812d83e5dccdc5f72/tensorflow/examples/speech_commands/recognize_commands.py
class RecognizeCommands(object):
  """Smooth the inference results by using average window.
  Maintain a slide window over the audio stream, which adds new result(a pair of
  the 1.confidences of all classes and 2.the start timestamp of input audio
  clip) directly the inference produces one and removes the most previous one
  and other abnormal values. Then it smooth the results in the window to get
  the most reliable command in this period.
  Attributes:
    _label: A list containing commands at corresponding lines.       # 尽管不建议用户调用,记载保护属性对于代码的阅读者
    _average_window_duration: The length of average window.          # 也能提供有用的参考
    _detection_threshold: A confidence threshold for filtering out unreliable
      command.
    _suppression_ms: Milliseconds every two reliable founded commands should
      apart.
    _minimum_count: An integer count indicating the minimum results the average
      window should cover.
    _previous_results: A deque to store previous results.
    _label_count: The length of label list.
    _previous_top_label: Last founded command. Initial value is '_silence_'.
    _previous_top_time: The timestamp of _previous results. Default is -np.inf.
  """

  def __init__(self, labels, average_window_duration_ms, detection_threshold,
               suppression_ms, minimum_count):
    """Init the RecognizeCommands with parameters used for smoothing."""
    # Configuration
    self._labels = labels
    self._average_window_duration_ms = average_window_duration_ms
    self._detection_threshold = detection_threshold
    self._suppression_ms = suppression_ms
    self._minimum_count = minimum_count
    # Working Variable
    self._previous_results = collections.deque()
    self._label_count = len(labels)
    self._previous_top_label = "_silence_"
    self._previous_top_time = -np.inf
```

```python
class WandbLogger(LightningLoggerBase):
    r"""
    Log using `Weights and Biases <https://www.wandb.com/>`_.
    
    Install it with pip:
    
    .. code-block:: bash
    
        pip install wandb
        
    Args:                     # 替代`__init__()`方法的Args小节
        name: Display name for the run.
        save_dir: Path where data is saved (wandb dir by default).
        offline: Run offline (data can be streamed later to wandb servers).
        id: Sets the version, mainly used to resume a previous run.
        version: Same as id.
        anonymous: Enables or explicitly disables anonymous logging.
        project: The name of the project to which this run will belong.
        log_model: Save checkpoints in wandb dir to upload on W&B servers.
        prefix: A string to put at the beginning of metric keys.
        experiment: WandB experiment object. Automatically set when creating a run.
        \**kwargs: Additional arguments like `entity`, `group`, `tags`, etc. used by
            :func:`wandb.init` can be passed as keyword arguments in this logger.
            
    Raises:                   # 总结所有函数和方法的`Raises`小节
        ImportError:
            If required WandB package is not installed on the device.
        MisconfigurationException:
            If both ``log_model`` and ``offline``is set to ``True``.
            
    Example::                 # 使用示例
        from pytorch_lightning.loggers import WandbLogger
        from pytorch_lightning import Trainer
        wandb_logger = WandbLogger()
        trainer = Trainer(logger=wandb_logger)

    Note:                     # 注意事项
    When logging manually through `wandb.log` or `trainer.logger.experiment.log`,
    make sure to use `commit=False` so the logging step does not increase.
    
    See Also:                 # 参见
        - `Tutorial <https://colab.research.google.com/drive/16d1uctGaw2y9KhGBlINNTsWpmlXdJwRW?usp=sharing>`__
          on how to use W&B with PyTorch Lightning
        - `W&B Documentation <https://docs.wandb.ai/integrations/lightning>`__
    """
```

```python
class Adam(optimizer_v2.OptimizerV2):
  r"""Optimizer that implements the Adam algorithm.
  
  Adam optimization is a stochastic gradient descent method that is based on
  adaptive estimation of first-order and second-order moments.
  
  According to [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
  the method is "*computationally efficient, has little memory requirement, 
  invariant to diagonal rescaling of gradients, and is well suited for 
  problems that are large in terms of data/parameters*".
  
  Args:                     # 替代`__init__()`方法的Args小节
    learning_rate: A `Tensor`, floating point value, or a schedule that is a
      `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
      that takes no arguments and returns the actual value to use, The
      learning rate. Defaults to 0.001.
    beta_1: A float value or a constant float tensor, or a callable
      that takes no arguments and returns the actual value to use. The
      exponential decay rate for the 1st moment estimates. Defaults to 0.9.
    beta_2: A float value or a constant float tensor, or a callable
      that takes no arguments and returns the actual value to use, The
      exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
    epsilon: A small constant for numerical stability. This epsilon is
      "epsilon hat" in the Kingma and Ba paper (in the formula just before
      Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
      1e-7.
    amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from
      the paper "On the Convergence of Adam and beyond". Defaults to `False`.
    name: Optional name for the operations created when applying gradients.
      Defaults to `"Adam"`.
    **kwargs: Keyword arguments. Allowed to be one of
      `"clipnorm"` or `"clipvalue"`.
      `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
      gradients by value.
      
  Usage:                     # 使用示例
  >>> opt = tf.keras.optimizers.Adam(learning_rate=0.1)
  >>> var1 = tf.Variable(10.0)
  >>> loss = lambda: (var1 ** 2)/2.0       # d(loss)/d(var1) == var1
  >>> step_count = opt.minimize(loss, [var1]).numpy()
  >>> # The first step is `-learning_rate*sign(grad)`
  >>> var1.numpy()
  9.9
  Reference:                 # 参考资料
    - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
    - [Reddi et al., 2018](
        https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.
  Notes:                     # 注意事项
  The default value of 1e-7 for epsilon might not be a good default in
  general. For example, when training an Inception network on ImageNet a
  current good choice is 1.0 or 0.1. Note that since Adam uses the
  formulation just before Section 2.1 of the Kingma and Ba paper rather than
  the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
  hat" in the paper.
  The sparse implementation of this algorithm (used when the gradient is an
  IndexedSlices object, typically because of `tf.gather` or an embedding
  lookup in the forward pass) does apply momentum to variable slices even if
  they were not used in the forward pass (meaning they have a gradient equal
  to zero). Momentum decay (beta1) is also applied to the entire momentum
  accumulator. This means that the sparse behavior is equivalent to the dense
  behavior (in contrast to some momentum implementations which ignore momentum
  unless a variable slice was actually used).
  """
```

#### 属性

### NumPy Style

## 注释

### 块注释和行内注释

注释应当在代码比较 tricky 的部分给出。如果你打算在下一次代码审查时解释某段代码，就应当在这里做注释。对于复杂的操作，在此操作刚出现之前给出几行注释进行说明；对于不显而易见的操作，在这一行的末尾给出行内注释。

```python
# We use a weighted dictionary search to find out where i is in
# the array.  We extrapolate position based on the largest num
# in the array and the array size and then do binary search to
# get the exact number.

if i & (i-1) == 0:  # True if i is 0 or a power of 2.
```

为了提高可读性，行内注释的 `#` 应距离代码至少2个空格，距离注释文本至少1个空格；块注释与其前后的代码都应间隔一行。

### TODO注释

对临时使用、仍待改善的代码使用 `TODO` 注释。`TODO` 注释以全大写的 `TODO` 开始，跟随一个圆括号包围的人名或id，email地址，或关联的issue，然后是解释接下来要做什么。

使用 `TODO` 注释的目的是有一个恒定的 `TODO` 格式可以用于搜索。`TODO` 注释并不代表承诺：括号中的这个人会修复这个问题。

```python
# TODO(kl@gmail.com): Use a "*" here for string repetition.
```

### 文法

注意 docstring 和注释的标点符号使用、单词拼写和文法，规范的文本让人更容易读。

注释应当和一般的陈述文本一样可读，在大多数情况下，完整的句子的可读性更好。行内注释使用短语可能会看起来比较不正式，但是更重要的是坚持使用一致的风格。

# Python Best Practice

* `dir(x)`查看当前变量`x`的所有方法，再使用`help(x.method)`获取方法的更多信息
* 读取文件时，逐行读取更常用，尤其是当读取文件较大，或每一行都需要单独处理时
* tuple通常用于单个对象的不同字段，list则通常用于多个对象

* 使用函数传递变量，避免全局变量
* 使用一个函数包装一个任务，以及相应的输入和输出变量
* 每定义一个函数使用多行注释给出文档。通常仅需一句话总结函数的作用，但如果需要给出更多信息，最好包含一个简单的例子
* 为输入和输出变量给出类型提示，这样更便于检查
* 对于布尔类型或可选的输入变量，一定要使用完整参数赋值语句以提高可读性
* 对函数和输入输出变量使用关键词命名以提高可读性
* 只捕获能够恢复正常运行的异常，或者显式指出程序不应该出现的情形，其余情况直接让程序崩溃
* 库文件最好有更大的灵活性，以实现广泛的功能

* 使用生成器的好处有：
    * 表示更清晰
    * 内存效率更高
    * 代码重用
      * 将产生迭代器与使用迭代器的位置分开

* Python的浮点数类型与C语言的double类型相同

## PEP8

### 代码布局

**空格**

* 注释符号`#`后面加一个空格，但第1行除外
* （使用vscode的格式化即可）

**空行**

* 函数内逻辑无关的段落之间空一行，但不要过度使用空行
* `if/for/while` 语句中，即使执行语句只有一句，也要另起一行
* （使用vscode的格式化即可）

**换行**

* 单行代码长度不超过79个字符

* 过长的行使用`\`或`(), [], {}`控制换行

* 当函数的参数过多需要换行时：

  ```python
  # good
  # 与左括号对齐
  foo = long_function_name(var_one, var_two,
                           var_three, var_four)
  
  # 用更多的缩进来与其他行区分
  def long_function_name(
          var_one, var_two, var_three,
          var_four):
      print(var_one)
  
  # 挂行缩进应该再换一行
  foo = long_function_name(
      var_one, var_two,
      var_three, var_four)
  
  # bad
  # 没有使用垂直对齐时，禁止把参数放在第一行
  foo = long_function_name(var_one, var_two,
      var_three, var_four)
  
  # 当缩进没有与其他行区分时，要增加缩进
  def long_function_name(
      var_one, var_two, var_three,
      var_four):
      print(var_one)
  ```

  

**命名**

* 不要使用`l, O, I`作为单字符变量名
* 模块名（`.py`文件名）应该使用短的、全小写的名字，可以加入下划线以提高可读性
* 包名应该使用短的、全小写的名字，但不鼓励使用下划线
* 类名使用首字母大写的驼峰命名
* 函数名，方法名和变量名使用全小写的名字，必要时加入下划线以提高可读性
* 常量名使用全大写的名字，必要时加入下划线以提高可读性

**import**

* 所有的 `import` 尽量放在文件开头，即 `docstring` 下方，全局变量定义的上方

* 不要使用 `from foo import *`

* `import` 需要分组，每组之间一个空行，每个分组内的顺序尽量采用字典序，分组顺序是：

  1. 标准库
  2. 第三方库
  3. 本项目的 package 和 module

* 对于不同的 package，一个 import 单独一行；同一个 package/module 下的内容可以写在一行：

  ```python
  # bad
  import sys, os, time
  # good
  import os
  import sys
  import time
  # ok
  from flask import Flask, render_template, jsonify
  ```

* 不要出现循环导入

**注释**

* 保持注释的正确性以及与代码的一致性；不要写无用的注释（废话）
* 注释如果是短语或句子，那么其第一个单词的首字母应该大写（除非为小写字母开头的命名）；如果句子较短，句尾的句号可以省略
* 注释使用英文，除非你能确保代码不会被其它语言的人阅读；如果英文水平不足，则注释全部使用中文
* 尽量少用行内注释；如果使用，则应和前面的语句隔开两个以上的空格

**异常**

* 不要轻易使用 `try/except`
* `except` 后面需要指定捕捉的异常，否则裸露的 `except` 会捕捉所有异常，隐藏潜在的问题
* `try/except` 里的内容不要太多，只在可能抛出异常的地方使用

**类**

* 显式地写明父类，如果不继承自别的类，就写明 `object`

**字符串**

* 坚持使用单引号或双引号中的一种；当字符串中出现单引号或双引号时，使用另外一种以避免使用 backslash

* 使用字符串的 `join` 方法拼接字符串

* 使用字符串类型的方法，而不是 `string` 模块的方法

* 使用 `startswith` 和 `endswith` 方法比较前缀和后缀

  ```python
  # good
  if foo.startswith('bar'):
  # bad
  if foo[:3] == 'bar':
  ```

* 使用 `format` 方法格式化字符串

**比较**

* 使用 `if` 判断某个字符串/列表/元组/词典是否为空：

  ```python
  # good
  if not seq:
  if seq:
  
  # bad
  if len(seq):
  if not len(seq):
  ```

* 使用`is not` 操作符而不是`not ... is `：

  ```python
  # good
  if foo is not None:
  # bad
  if not foo is None:
  ```

* 用 `isinstance` 判断类型

  ```python
  # good
  if isinstance(obj, int):
  # bad
  if type(obj) is type(1):
  ```

* 不要用 `==, !=` 与 `True, False` 比较

  ```python
  # good
  if condition:
  # bad
  if condition == True:
  # worse
  if condition is True:
  ```

  

**其它**

* 使用 `for item in list` 迭代 `list`，`for index, item in enumerate(list)` 迭代 `list` 并获取下标
* 使用 `logging` 记录日志，配置好格式和级别

