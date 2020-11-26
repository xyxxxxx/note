# Python Best Practice

+ 每级缩进使用4个空格而不是1个tab`\t`。
+ `dir(x)`查看当前变量`x`的所有方法，再使用`help(x.method)`获取方法的更多信息
+ 读取文件时，逐行读取更常用，尤其是当读取文件较大，或每一行都需要单独处理时
+ 
+ tuple通常用于单个对象的不同字段，list则通常用于多个对象

+ 使用函数传递变量，避免全局变量
+ 使用一个函数包装一个任务，以及相应的输入和输出变量
+ 每定义一个函数使用多行注释给出文档。通常仅需一句话总结函数的作用，但如果需要给出更多信息，最好包含一个简单的例子
+ 为输入和输出变量给出类型提示，这样更便于检查
+ 对于布尔类型或可选的输入变量，一定要使用完整参数赋值语句以提高可读性
+ 对函数和输入输出变量使用关键词命名以提高可读性
+ 只捕获能够恢复正常运行的异常，或者显式指出程序不应该出现的情形，其余情况直接让程序崩溃
+ 库文件最好有更大的灵活性，以实现广泛的功能

+ 尽量不要多重继承
+ 使用生成器的好处有：
  + 表示更清晰
  + 内存效率更高
  + 代码重用
    + 将产生迭代器与使用迭代器的位置分开



+ Python的浮点数类型与C语言的double类型相同



## PEP8

### 代码布局

**缩进**

+ 每个缩进层次为4个空格
+ 不要使用tab；对于任何编辑器，将tab键设定为输入4个空格;



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



## PEP257

+ 文档字符串 `docstring` 是包、模块、类、方法、函数的注释，所有公开的包、模块、类、方法、函数都应该有文档字符串，可以通过 `__doc__` 方法访问，注释内容在一对 `'''` 或一对 `"""` 之间，建议使用后者
+ 函数和方法的文档字符串应当描述其功能、输入参数、返回值；如果有复杂的算法和实现，也需要写清楚或给出参考文献
+ 对于多行的文档字符串，结束 `"""` 独占一行；对于单行的文档字符串，将开始和结束 `"""` 放在同一行

