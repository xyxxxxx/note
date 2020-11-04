# 模块

一个`.py`源文件就是一个模块（module）。

## 导入模块

```python
import math	     # 导入math包，即执行该文件
import math as m # 导入math包并在本地赋名
from math import cos, sin # 导入math包并将cos, sin添加到本地命名空间

x=math.sin(math.pi/2)
```

> Python标准库https://docs.python.org/zh-cn/3/library/index.html



## 定义模块

```python
#!/usr/bin/env python3			# 标准注释:py3文件
# -*- coding: utf-8 -*-			# 标准注释:使用UTF-8编码

' a test module '				# 文档注释

__author__ = 'Michael Liao'		# 作者名

import sys						# 正文

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

