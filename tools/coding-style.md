[toc]

# Python

> 参考：
>
> [PEP 8](https://www.python.org/dev/peps/pep-0008/)
>
> PEP 257
>
> [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

## 格式

### 缩进

+ 每个缩进层次为4个空格（而非一个tab`\t`）

+ 不要使用tab；对于任何编辑器，将tab键设定为输入4个空格

+ 对于拆分到多行的代码，应竖直对齐包装的所有元素，或使用4个空格的缩进并且第一行的括号后直接换行：

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

+ 一行代码的最大长度应为80个字符。

+ 超过80个字符限制的常见例外情况包括：

  + 长的 `import` 语句
  + 注释中的 URL，路径名和长标记
  + 长的模块级别的字符串常量，因为不包含 whitespace 而不便于拆分到多行，例如 URL 或路径名

  ```python
  # 正确的示范
  # See details at
  # http://www.example.com/us/developer/documentation/api/content/v2.0/csv_file_name_extension_full_specification.html
  
  # 错误的示范
  # See details at
  # http://www.example.com/us/developer/documentation/api/content/\
  # v2.0/csv_file_name_extension_full_specification.html
  ```

+ 利用 Python 隐式拼接括号内各行的特性，必要时可以使用圆括号包围表达式

  ```python
  # 正确的示范
  # 隐式拼接括号内各行
  foo_bar(self, width, height, color='black', design=None, x='foo',
          emphasis=None, highlight=0)
  
  # 使用圆括号包围表达式
  if (width == 0 and height == 0 and
      color == 'red' and emphasis == 'strong'):
  ```

+ 当字符串在一行中容纳不下时，使用小括号隐式拼接

  ```python
  # 正确的示范
  x = ('This will build a very long long '
       'long long long long long long string')
  
  url = ('http://www.example.com/us/developer/documentation/api/content/v2.0'
         '/csv_file_name_extension_full_specification.html')
  ```

+ 不要使用反斜线拆分行，除非是 `with` 语句需要3个或更多的上下文管理器

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

+ 对于一行超过80个字符的所有其它情况，若 yapf 自动格式化器不能帮助拆分行，则该行可以超过限制。



### 圆括号

+ 控制圆括号的使用。
+ 可以对元组使用圆括号包围，但这不是必须的。



### 逗号

+ 对于拆分到多行的代码，如果反括号与最后一个元素不在一行，则应在最后一个元素之后增加一个逗号。

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

+ 单字符名称，除非一些特殊情况：
  + 计数或迭代变量（例如 `i`, `j`, `k`, `v` 等）
  + `e` 作为 `try/except` 语句中的异常标识符
  + `f` 作为 `with` 语句中的文件操作符
+ 名称中使用横线 `-` 。
+ 形如 `__my__` 的特殊属性名称（由 Python 保留）
+ 冒犯性的名称



### 惯例

+ “内部”指代模块的内部，或类内部的保护或私有属性。
+ 使用单下划线前缀将提供一些保护模块变量和函数的支持。使用双下划线前缀将使得变量和函数为其类完全私有，但我们不鼓励使用它，因为它会影响代码的可读性和可测试性，而且并不是真正的*私有*。
+ 将关联的类和顶级函数放置在一个模块中。不同于 Java，Python 没有限制一个模块只能有一个类。







## 类型

### 真值判断





## 字符串







## 容器

### 生成式

使用列表、字典、集合生成式的目的应是使代码简洁易读。映射表达式、for语句、if语句，每个部分不得超过一行。若生成式过长，或有多个for语句、if语句嵌套，则应拆分为一般的循环体。



## 控制

### 条件表达式

对于非常简短的判断可以使用。真值表达式、if语句、else语句，每个部分不得超过一行。若条件表达式过长，则应拆分为一般的条件语句。



## 函数

### Lambda函数

对于非常简短的函数可以使用。如果lambda函数的长度超过80个字符，则应考虑使用一般的嵌套函数。

对于乘法这样的通常运算，使用`operator`模块中的函数而不要使用lambda函数，例如使用 `operator.mul` 而非 `lambda x, y: x * y`。



### 默认值参数

不要使用可变对象作为函数或方法定义的默认值。详见函数-参数-默认值参数。







## 类

### 属性





### 继承

尽量不要使用多重继承。



## 模块

### import

使用方法：

+ `import x`：导入包和模块，例如 `import torch`。
+ `from x import y`：从包中导入子包、模块或类，例如 `from tensorflow import keras `，`from tensorflow.keras.datasets import mnist`，`from pytorch_lightning import LightningModule, LightningDataModule, Trainer`。
+ `from x import y as z`：当导入两个同名为 `y` 的模块，或名称 `y` 非常长。
+ `import y as z`：仅当 `z` 是一个标准缩写，例如 `import numpy as np`， `import torch.nn.functional as F`。

不要使用上述方法以外的方法，例如导入全部 `from x import *`。

不要使用相对路径导入，即使导入的模块在同一个包下，也使用完整的包名。



### 包





### 全局变量

一般应避免使用全局变量。但作为技术指标的全局变量是被允许和鼓励的，例如 `MAX_HOLY_HANDGRENADE_COUNT = 3`。

对外部隐藏全局变量应为变量名添加前缀 `_`，外部的访问需要通过公开的模块级别的函数完成。





## 文档字符串和注释

Python 使用文档字符串 docstring 来为代码生成文档。一个 docstring 是包、模块、类、方法、函数的第一个声明字符串，这些字符串可以通过对象的 `__doc__` 属性自动提取，也被 `pydoc` 使用。所有公开的包、模块、类、方法、函数都应该有 docstring。docstring 的内容用一对 `"""` 包围。

docstring 应当被组织为：一行总结（不超过80个字符），以句号结尾；空一行，从第一行的第一个引号的位置开始，



+ 函数和方法的文档字符串应当描述其功能、输入参数、返回值；如果有复杂的算法和实现，也需要写清楚或给出参考文献
+ 对于多行的文档字符串，结束 `"""` 独占一行；对于单行的文档字符串，将开始和结束 `"""` 放在同一行



### 模块

+ 每一个文件都应当包含许可的模版内容，为项目选择合适的许可模板（例如 Apache 2.0，BSD，LGPL，GPL）。

+ 文件应以一个 docstring 开始，描述模块的内容和使用方法。

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
  # TensorFlow 代码示例
  
  """Adam optimizer implementation."""
  
  """TensorFlow-related utilities."""
  
  # Lightning 代码示例
  
  """Trainer to automate the training."""
  ```



### 函数和方法







# Python Best Practice

+ `dir(x)`查看当前变量`x`的所有方法，再使用`help(x.method)`获取方法的更多信息
+ 读取文件时，逐行读取更常用，尤其是当读取文件较大，或每一行都需要单独处理时
+ tuple通常用于单个对象的不同字段，list则通常用于多个对象

+ 使用函数传递变量，避免全局变量
+ 使用一个函数包装一个任务，以及相应的输入和输出变量
+ 每定义一个函数使用多行注释给出文档。通常仅需一句话总结函数的作用，但如果需要给出更多信息，最好包含一个简单的例子
+ 为输入和输出变量给出类型提示，这样更便于检查
+ 对于布尔类型或可选的输入变量，一定要使用完整参数赋值语句以提高可读性
+ 对函数和输入输出变量使用关键词命名以提高可读性
+ 只捕获能够恢复正常运行的异常，或者显式指出程序不应该出现的情形，其余情况直接让程序崩溃
+ 库文件最好有更大的灵活性，以实现广泛的功能

+ 使用生成器的好处有：
  + 表示更清晰
  + 内存效率更高
  + 代码重用
    + 将产生迭代器与使用迭代器的位置分开



+ Python的浮点数类型与C语言的double类型相同



## PEP8

### 代码布局



**空格**

+ 注释符号`#`后面加一个空格，但第1行除外
+ （使用vscode的格式化即可）



**空行**

+ 函数内逻辑无关的段落之间空一行，但不要过度使用空行
+ `if/for/while` 语句中，即使执行语句只有一句，也要另起一行
+ （使用vscode的格式化即可）



**换行**

+ 单行代码长度不超过79个字符

+ 过长的行使用`\`或`(), [], {}`控制换行

+ 当函数的参数过多需要换行时：

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

+ 不要使用`l, O, I`作为单字符变量名
+ 模块名（`.py`文件名）应该使用短的、全小写的名字，可以加入下划线以提高可读性
+ 包名应该使用短的、全小写的名字，但不鼓励使用下划线
+ 类名使用首字母大写的驼峰命名
+ 函数名，方法名和变量名使用全小写的名字，必要时加入下划线以提高可读性
+ 常量名使用全大写的名字，必要时加入下划线以提高可读性



**import**

+ 所有的 `import` 尽量放在文件开头，即 `docstring` 下方，全局变量定义的上方

+ 不要使用 `from foo import *`

+ `import` 需要分组，每组之间一个空行，每个分组内的顺序尽量采用字典序，分组顺序是：

  1. 标准库
  2. 第三方库
  3. 本项目的 package 和 module

+ 对于不同的 package，一个 import 单独一行；同一个 package/module 下的内容可以写在一行：

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

+ 不要出现循环导入



**注释**

+ 保持注释的正确性以及与代码的一致性；不要写无用的注释（废话）
+ 注释如果是短语或句子，那么其第一个单词的首字母应该大写（除非为小写字母开头的命名）；如果句子较短，句尾的句号可以省略
+ 注释使用英文，除非你能确保代码不会被其它语言的人阅读；如果英文水平不足，则注释全部使用中文
+ 尽量少用行内注释；如果使用，则应和前面的语句隔开两个以上的空格



**异常**

+ 不要轻易使用 `try/except`
+ `except` 后面需要指定捕捉的异常，否则裸露的 `except` 会捕捉所有异常，隐藏潜在的问题
+ `try/except` 里的内容不要太多，只在可能抛出异常的地方使用



**类**

+ 显式地写明父类，如果不继承自别的类，就写明 `object`



**字符串**

+ 坚持使用单引号或双引号中的一种；当字符串中出现单引号或双引号时，使用另外一种以避免使用 backslash

+ 使用字符串的 `join` 方法拼接字符串

+ 使用字符串类型的方法，而不是 `string` 模块的方法

+ 使用 `startswith` 和 `endswith` 方法比较前缀和后缀

  ```python
  # good
  if foo.startswith('bar'):
  # bad
  if foo[:3] == 'bar':
  ```

+ 使用 `format` 方法格式化字符串



**比较**

+ 使用 `if` 判断某个字符串/列表/元组/词典是否为空：

  ```python
  # good
  if not seq:
  if seq:
  
  # bad
  if len(seq):
  if not len(seq):
  ```

+ 使用`is not` 操作符而不是`not ... is `：

  ```python
  # good
  if foo is not None:
  # bad
  if not foo is None:
  ```

+ 用 `isinstance` 判断类型

  ```python
  # good
  if isinstance(obj, int):
  # bad
  if type(obj) is type(1):
  ```

+ 不要用 `==, !=` 与 `True, False` 比较

  ```python
  # good
  if condition:
  # bad
  if condition == True:
  # worse
  if condition is True:
  ```

  

**其它**

+ 使用 `for item in list` 迭代 `list`，`for index, item in enumerate(list)` 迭代 `list` 并获取下标
+ 使用 `logging` 记录日志，配置好格式和级别


