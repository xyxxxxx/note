[toc]

# 模块

一个 `.py` 源文件就是一个模块（module）。

## 导入模块

```python
import math	     # 导入math包，即执行该文件
import math as m # 导入math包并在本地赋名
from math import cos, sin # 导入math包并将cos, sin添加到本地命名空间

x = math.sin(math.pi/2)
```

> Python标准库https://docs.python.org/zh-cn/3/library/index.html



### 导入\__future__模块

`__future__` 模块包含了一些将在未来 Python 版本中成为标准特性的语法，让我们可以在之前的版本中提前使用该特性。它的目的是使得向引入了不兼容改变的新版本迁移更加容易。

```python
# in python 2.x
from __future__ import division
print 8 / 7    # 除法
print 8 // 7   # 整数除法取商
```

导入 `__future__` 模块的语句必须位于模块的顶部，在模块的 `docstring` 和注释之后，其它 `import` 语句之前。

`__future__` 模块包含以下特性：

| 特性名称         | 引入版本 | 强制版本 | 效果简介          | 详细的 PEP 文章                                       |
| :--------------- | :------- | :------- | :---------------- | :---------------------------------------------------- |
| nested_scopes    | 2.1.0b1  | 2.2      | 嵌套作用域        | [PEP-0227](https://www.python.org/dev/peps/pep-0227/) |
| generators       | 2.2.0a1  | 2.3      | 生成器语法        | [PEP-0255](https://www.python.org/dev/peps/pep-0255/) |
| division         | 2.2.0a2  | 3.0      | 强制浮点数除法    | [PEP-0238](https://www.python.org/dev/peps/pep-0238/) |
| absolute_import  | 2.5.0a1  | 3.0      | 绝对引入          | [PEP-0328](https://www.python.org/dev/peps/pep-0328/) |
| with_statement   | 2.5.0.a1 | 2.6      | with 声明         | [PEP-0343](https://www.python.org/dev/peps/pep-343/)  |
| print_function   | 2.6.0a2  | 3.0      | 强制 print 为函数 | [PEP-3105](https://www.python.org/dev/peps/pep-3105/) |
| unicode_literals | 2.6.0a2  | 3.0      | 默认为 unicode    | [PEP-3112](https://www.python.org/dev/peps/pep-3112/) |
| generator_stop   | 3.5.0b1  | 3.7      | 终止生成器        | [PEP-0479](https://www.python.org/dev/peps/pep-0479/) |
| annotations      | 3.7.0b1  | 3.10     | 注解              | [PEP-0563](https://www.python.org/dev/peps/pep-0563/) |



## 定义模块

```python
#!/usr/bin/env python3			# 标准注释:py3文件
# -*- coding: utf-8 -*-			# 标准注释:使用UTF-8编码

"""a test module."""			  # 文档注释

__author__ = 'Michael Liao'	 # 作者名

import sys						      # 正文

def test():
    args = sys.argv
    if len(args)==1:
        print('Hello, world!')
    elif len(args)==2:
        print('Hello, %s!' % args[1])
    else:
        print('Too many arguments!')

if __name__=='__main__':        # 执行该模块时运行
    test()
```



## 作用域

```python
# abc		public变量
# _abc		public变量，但惯例不直接引用
# __abc		private变量，不可直接引用
# __abc__	特殊变量，可以直接引用

def _private_1(name):		# 内部函数
    return 'Hello, %s' % name

def _private_2(name):		
    return 'Hi, %s' % name

def greeting(name):			# 外部接口
    if len(name) > 3:
        return _private_1(name)
    else:
        return _private_2(name)
```





# 包

对于更大规模的库文件，通常的做法是将模块组织成包（package）。

```python
# from this
util1.py
util2.py
util3.py

# to this
util/
    __init__.py
    util1.py
    util2.py
    util3.py
    
# use package
import util
util.util1.func1()

from util import util1
util1.func1()

from util.util1 import func1
func1()
```

同一包中各个模块的互相导入方法需要改变：

```python
# from this
# util1.py
import util2

# to this
# util1.py
from . import util2
```

运行包中这些模块的命令也需要改变：

```shell
# from this
$ python3 util/util1.py

# to this
$ python3 -m util.util1
```

```python
# __init__.py
from .util1 import func1

# 上述声明使func1成为util命名空间下的顶级名称
import util
util.func1()

from util import func1
func1()
```





