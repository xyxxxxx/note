[toc]

# 模块

模块（module）是包含 Python 定义和语句的文件，文件名为模块名加上后缀名 `.py`。



## 导入模块

定义一个简单的模块 `module1.py`：

```python
GLOBAL = 100

class Class1(object):
    pass

def func1():
    print('This is module1.func1')

def func2():
    print(GLOBAL)

def _func3():
    print('This is protected module1._func3')

print('module1 initialized.')

```

进入 Python 解释器，使用 `import` 语句导入该模块：

```python
>>> import module1
module1 initialized.
```

模块包含可执行语句以及类和函数的定义，这些语句用于初始化模块，在且仅在 `import` 语句第一次遇到模块名时执行。再次导入该模块时，不再调用 `print()` 函数：

```python
>>> import module1
```

导入模块之后，使用模块名访问模块中的类和函数：

```python
>>> module1.func1()
This is module1.func1
>>> instance1 = module1.Class1()
```

模块有自己的私有符号表，用作模块中所有类和函数的全局符号表，因此在模块内使用全局变量时，不用担心与用户定义的全局变量发生冲突；与此同时，可以用与访问模块类和函数一样的标记法访问模块的全局变量：

```python
>>> module1.GLOBAL     # 模块的全局变量
100
>>> GLOBAL = 50        # 用户定义的全局变量
>>> module1.func2()    # 使用模块的私有符号表
100
>>> dir()              # 当前解释器作用域(模块)的属性列表(也可以理解为全局符号表);刚才定义的变量是属性之一
['GLOBAL', '__annotations__', '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__', 'module1']
>>> dir(module1)       # 模块的属性列表;模块的全局变量是属性之一
['Class1', 'GLOBAL', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '_func3', 'func1', 'func2']
```

可以在模块中导入其它模块，被导入模块的名称会存在导入模块的全局符号表里。这里再在同一目录下定义模块 `module2.py`：

```python
import module1         # 按照惯例,`import`语句都放在模块开头

class Class2(object):
    pass

def func4():
    print('This is module2.func4')

print('module2 initialized.')

```

导入该模块：

```python
>>> import module2
module1 initialized.       # 初始化`module1`
module2 initialized.       # 初始化`module2`
>>> dir(module2)
['Class2', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'func4', 'module1']    # `module1`作为`module2`的属性
```



### import语句的变体

`import` 语句有如下几种变体：

+ `import ...`：导入模块：

  ```python
  >>> import module1
  module1 initialized.
  >>> dir()               # 导入了模块`module1`,即将模块实例加入到当前作用域的属性列表
  ['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__', 'module1']
  >>> dir(module1)        # 模块`module1`具有下列属性
  ['Class1', 'GLOBAL', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '_func3', 'func1', 'func2']
  ```

+ `from ... import ...`：直接导入模块内的（类、函数或全局变量的）名称，被导入的名称会存在导入模块的全局符号表里：

  ```python
  >>> from module1 import func2
  module1 initialized.    # 执行了`module1`的所有语句
  >>> dir()               # 属性列表仅增加了`func2`,没有`module1`
  ['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__', 'func2']
  >>> func2()             # 可以访问`module1`中的全局变量
  100                     # `from ... import ...`与`import ...`的区别仅在于,前者直接添加模块的部分名称
                          # 到当前作用域的属性列表中
  ```

+ `from ... import *`：直接导入模块内定义的所有（不以`_`开头的）名称：

  ```python
  >>> from module1 import *
  module1 initialized.
  >>> dir()
  ['Class1', 'GLOBAL', '__annotations__', '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__', 'func1', 'func2']        # '_func3'作为保护属性未被导入
  ```

  一般情况下，不建议使用这个功能，因为这种方式导入的一批未知的名称可能会与当前模块的其它名称相互覆盖，并且让读者难以得知到底导入了哪些名称。

+ `import ... as ...`：使用 `as` 之后的名称代表被导入模块：

  ```python
  >>> import module1 as m1
  module1 initialized.
  >>> dir()
  ['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__', 'm1']
  >>> m1.func1()
  This is module1.func1
  ```

+ `from ... import ... as ...`：使用 `as` 之后的名称代表被导入的名称。

  ```python
  >>> from module1 import func1 as f1
  module1 initialized.
  >>> f1()
  This is module1.func1
  ```



### 模块搜索路径

导入模块 `foo` 时，Python 解释器从变量 `sys.path` 的目录列表里查找文件 `foo.py` 。`sys.path` 初始化时包含以下位置：

+ 调用 Python 解释器的脚本所在的目录，或以交互方式调用解释器时的当前目录
+ 环境变量 `PYTHONPATH` 包含的路径（语法与 `PATH` 相同）
+ 当前 Python 解释器的库

可以根据需要任意修改该列表，例如：

```python
>>> import sys
>>> sys.path.append('/ufs/guido/lib/python')
```



### 已编译文件

为了快速加载模块，Python 把模块的编译版缓存在 `__pycache__` 目录中，文件名为 `module.*version*.pyc`，*version* 对编译文件格式进行编码，一般是 Python 的版本号。例如，CPython 的 3.3 发行版中，spam.py 的编译版本缓存为 `__pycache__/spam.cpython-33.pyc`。使用这种命名惯例，可以让不同 Python 发行版及不同版本的已编译模块共存。

Python 对比编译版本与源码的修改日期，查看它是否已过期，是否要重新编译，此过程完全自动化。此外，编译模块与平台无关，因此，可在不同架构系统之间共享相同的支持库。

Python 在两种情况下不检查缓存：其一，将模块作为脚本执行，这时只重新编译，不保存编译结果；其二，没有源模块，故不会检查缓存。为了支持无源文件（仅编译）发行版本，编译模块必须在源目录下，并且绝不能有源模块。



### 标准库

Python 附带了标准（模块）库，其中内置了一些模块，用于访问不属于语言核心的内置操作，其目的主要是为了提高运行效率，或访问系统调用等操作系统原语。详见标准库。



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



## 执行模块

Python 是一种动态语言，或称为脚本语言，即读一行解释一行执行一行。在交互的 Python 解释器中执行若干条语句，与将这些语句放入一个模块中作为脚本执行实际上没有差别。

> 脚本（script）相当于电影或者戏剧的脚本（或剧本），演员读完脚本就知道要表演什么，而计算机读完脚本就知道要完成什么操作。
>
> 脚本指短小的、用来让计算机自动化完成一系列工作的程序，这类程序可以用文本编辑器修改，并且通常是解释运行的（亦即，计算机中有一个程序能将脚本里的内容一句一句地转换成CPU能理解的指令，而且它每次只解释一句，待CPU执行完再解释下一句）。

以下命令将模块作为脚本执行：

```shell
$ python module1.py [arguments]
```

这项操作将执行模块里的所有代码，和导入模块一样，唯一的区别是会将模块的 `__name__` 属性修改为 `'__main__'`。在模块 `module1.py` 末尾添加一行 `print(__name__)`，再分别导入和作为脚本执行：

```shell
$ python
>>> import module1
This is module1.
module1
>>> module1.__name__    # 导入模块时,其`__name__`属性为模块名称
'module1'

$ python module1.py     # 作为脚本执行时,该属性被修改为'__main__'
This is module1.
__main__
```

一种常见的做法是将下列代码添加到模块末尾：

```python
if __name__ == "__main__":
    pass
```

这样当模块作为脚本执行时，会执行条件体下的代码，而导入该模块时则不会执行。



## 标准模块定义

```python
#!/usr/bin/env python3			 # 标准注释:py3文件
# -*- coding: utf-8 -*-			 # 标准注释:使用UTF-8编码

"""a test module."""			   # 文档注释

__author__ = 'Michael Liao'	 # 作者名

import sys						       # 正文

def test():
    args = sys.argv
    if len(args)==1:
        print('Hello, world!')
    elif len(args)==2:
        print('Hello, %s!' % args[1])
    else:
        print('Too many arguments!')

if __name__=='__main__':     # 执行该模块时运行
    test()
```





# 包

包（package）是一种使用“点式模块名”构造 Python 模块命名空间的方法，例如模块名 `A.B` 表示包 `A` 中名为 `B` 的子模块。正如模块可以区分不同模块的全局变量名称一样，点式模块名可以区分 NumPy 或 Pillow 等不同多模块包的模块名称。



## 导入包

定义一个简单的包 `package1`：

```
package1                   # 顶级包
├───__init__.py            # 包的初始化代码
├───subpackage1              # 子包
│   ├───__init__.py          # 子包的初始化代码
│   ├───module1.py             # 模块
│   └───module2.py
└───subpackage2
    ├───__init__.py
    ├───module3.py
    └───module4.py
```

其中所有 `__init__.py` 只有一行 `print(__name__)`，`module1.py`, `module2.py` 与[模块](#导入模块)部分定义的相同，`module3.py`, `module4.py` 定义如下：

```python
# package1/subpackage2/module3.py
class Class1(object):
    pass

def func1():
    print('This is module3.func1')

print('module3 initialized.')

```

```python
# package1/subpackage2/module4.py
class Class2(object):
    pass

def func2():
    print('This is module4.func2')

print('module4 initialized.')    

```

Python 只把含有 `__init__.py` 文件的目录当作包。最简单的情况下，`__init__.py` 只是一个空文件，但该文件也可以执行包的初始化代码，或设置 `__all__` 变量。



与导入模块相同，使用 `import` 语句导入包：

```python
>>> import package1
```

也可以直接导入子包或模块：

```python
>>> import package1.subpackage1            # 导入子包.此处的点表示文件系统的上下级关系,导入之后成为实例与属性的关系
```

```python
>>> import package1.subpackage1.module1    # 导入模块
module1 initialized.
package1.subpackage1.module1
```

但是无法使用包名访问子包和模块：

```python
>>> package1.subpackage2.module3.func1()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: module 'package1' has no attribute 'subpackage2'
```

这是因为包 `package1` 未进行初始化，子包 `subpackage1` 和 `subpackage2` 都不是它的属性。



### import语句的变体

`import` 语句有如下几种变体：

+ `import`：导入包、子包或模块：

  ```python
  >>> import package1
  package1
  >>> dir()               # 导入了包`package1`,即将模块实例加入到当前作用域的属性列表
  ['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__', 'package1']
  >>> dir(package1)       # 包`package1`仅有特殊属性
  ['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__']
  ```

  ```python
  >>> import package1.subpackage1
  package1
  package1.subpackage1
  >>> dir()               # 导入了包`package1`
  ['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__', 'package1']
  >>> dir(package1)       # 包`package1`有属性`subpackage1`
  ['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'subpackage1']
  >>> dir(package1.subpackage1)   # 子包`subpackage1`仅有特殊属性
  ['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__']
  ```

  ```python
  >>> import package1.subpackage1.module1
  package1
  package1.subpackage1
  module1 initialized.
  package1.subpackage1.module1
  >>> dir()               # 导入了包`package1`
  ['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__', 'package1']
  >>> dir(package1)       # 包`package1`有属性`subpackage1`
  ['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'subpackage1']
  >>> dir(package1.subpackage1)   # 子包`subpackage1`有属性`module1`
  ['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'module1']
  >>> dir(package1.subpackage1.module1)   # 模块`module1`有下列属性
  ['Class1', 'GLOBAL', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '_func3', 'func1', 'func2']
  ```

+ `from ... import ...`：从包导入子包/模块/名称，或从模块导入模块内的（类、函数或全局变量的）名称，被导入的名称会存在导入模块的全局符号表里：

  ```python
  >>> from package1 import subpackage1
  package1
  package1.subpackage1
  >>> dir()               # 导入了子包`subpackage1`
  ['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__', 'subpackage1']
  >>> dir(subpackage1)    # 子包`subpackage1`仅有特殊属性
  ['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__']             # `from A import B`与`import A.B`的区别仅在于,前者直接添加子包/模块
                          # 到当前作用域的属性列表中
  ```

  ```python
  >>> from package1.subpackage1 import module1
  package1
  package1.subpackage1
  module1 initialized.
  package1.subpackage1.module1
  >>> dir()               # 导入了模块`module1`
  ['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__', 'module1']
  >>> dir(module1)        # 模块`module1`有下列属性
  ['Class1', 'GLOBAL', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '_func3', 'func1', 'func2']
  ```

  ```python
  # 若在`package1`的`__init__.py`中添加一行 `INIT = 100`
  >>> from package1 import INIT
  package1
  >>> dir()               # 导入了名称`INIT`.与从模块导入名称相同.
  ['INIT', '__annotations__', '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__']
  ```

  ```python
  >>> from package1.subpackage1.module1 import func2
  package1
  package1.subpackage1
  module1 initialized.
  package1.subpackage1.module1
  >>> dir()               # 属性列表增加了`func2`
  ['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__', 'func2']
  >>> func2()
  100                     # `from ... import ...`与`import ...`的区别仅在于,前者直接添加模块的部分名称
                          # 到当前作用域的属性列表中
  ```

+ `from ... import *`：如果包的 `__init__.py` 定义了属性 `__all__` 为子包/模块名列表，则导入这些子包/模块（详见初始化包），否则只导入 `__init__.py` 中定义的名称，以及之前 `import` 语句显式加载的包的子包/模块；或从模块导入模块内定义的所有（不以`_`开头的）名称：

  ```python
  >>> from package1 import *
  package1
  >>> dir()               # 未导入任何名称
  ['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__']
  ```

  ```python
  # 若在`package1`的`__init__.py`中添加一行 `INIT = 100`
  >>> from package1 import *
  package1
  >>> dir()               # 导入`__init__.py`中定义的名称
  ['INIT', '__annotations__', '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__']
  ```

  ```python
  # 若在`package1`的`__init__.py`中再添加一行 `__all__ = ['subpackage1']`
  >>> from package1 import *
  package1
  package1.subpackage1
  >>> dir()               # 导入`__all__`中包含的子包,相当于`from package1 import subpackage1`
                          # 不导入`__init__.py`中定义的名称
  ['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__', 'subpackage1']
  ```

  ```python
  >>> from package1.subpackage1.module1 import *
  package1
  package1.subpackage1
  module1 initialized.
  package1.subpackage1.module1
  >>> dir()
  ['Class1', 'GLOBAL', '__annotations__', '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__', 'func1', 'func2']        # '_func3'作为保护属性未被导入
  ```

  一般情况下，不建议使用这个功能，因为这种方式导入的一批未知的名称可能会与当前模块的其它名称相互覆盖，并且让读者难以得知到底导入了哪些名称。

+ `import ... as ...`：使用 `as` 之后的名称代表被导入的包/子包/模块：

  ```python
  >>> import package1.subpackage1.module1 as m1
  package1
  package1.subpackage1
  module1 initialized.
  package1.subpackage1.module1
  >>> m1.func1()
  This is module1.func1
  ```

+ `from ... import ... as ...`：使用 `as` 之后的名称代表被导入的子包/模块/名称。

  ```python
  >>> from package1.subpackage1.module1 import func1 as f1
  package1
  package1.subpackage1
  module1 initialized.
  package1.subpackage1.module1
  >>> f1()
  This is module1.func1
  ```



### 初始化包

`__init__.py` 包含的可执行语句以及类和函数的定义用于初始化包，在且仅在 `import` 语句第一次遇到包名时执行。将包 `package1` 的 `__init__.py` 修改如下：

```python
print(__name__)

INIT = 100

def init():
    print(INIT)
```

导入该包：

```python
>>> import package1
package1
>>> dir(package1)         # 导入`__init__.py`中定义的名称
['INIT', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'init']
>>> import package1       # 再次导入时不再调用`print()`函数
>>> 
```



`__init__.py` 包含的 `import` 语句用于将子包/模块/名称作为包实例的属性。将包 `package1` 的 `__init__.py` 修改如下：

```python
print(__name__)

from . import subpackage1                   # 相对导入,点表示当前路径(.../package1)
# from package1 import subpackage1          # 绝对导入
from .subpackage2 import module3            # 相对导入
# from package1.subpackage2 import module3  # 绝对导入

# import package1.subpackage1               # 不好的示例,会导致`package1`作为自身的属性
```

导入该包：

```python
>>> import package1
package1
package1.subpackage1
package1.subpackage2
module3 initialized.
>>> dir(package1)
['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'module3', 'subpackage1', 'subpackage2']   # 注意此时`subpackage2`也被导入
>>> dir(package1.subpackage2)                          # 可能是因为复用了`package1`
['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'module3']
```



如果 `__init__.py` 定义了列表 `__all__`，则执行 `from ... import *` 时将导入列表中包含的子包/模块名；如果没有定义此列表，则应视作包的作者不建议执行导入全部的操作。将包 `package1` 的 `__init__.py` 修改如下：

```python
print(__name__)

__all__ = ['subpackage1']
```

从该包导入全部：

```python
>>> from package1 import *        # 相当于 `from package1 import subpackage1`
package1
package1.subpackage1
>>> dir()
['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__', 'subpackage1']
```

在没有定义此列表的情况下执行 `from ... import *` 时将导入 `__init__.py` 中定义的名称，以及之前 `import` 语句显式加载的包的子包/模块。将包 `package1` 的 `__init__.py` 修改如下：

```python
print(__name__)

INIT = 100

def init():
    print(INIT)

from . import subpackage1
from .subpackage2 import module3
```

从该包导入全部：

```python
>>> from package1 import *
package1
package1.subpackage1
package1.subpackage2
module3 initialized.
>>> dir()                   # 导入`__init__.py`中定义的名称以及显式导入的子包/模块
['INIT', '__annotations__', '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__', 'init', 'module3', 'subpackage1', 'subpackage2']
```



### 包是特殊的模块

包是一种组织模块的方法，但实际上包也可以视作是一种特殊的模块。导入的包和模块在 Python 解释器中都是 `module` 类型，唯一的区别在于包具有特殊属性 `__path__`：

```python
>>> import numpy
>>> type(numpy)
<class 'module'>
>>> numpy.__path__
['/Users/xyx/.pyenv/versions/3.8.7/lib/python3.8/site-packages/numpy']
```

包的 `__path__` 属性会在导入其子包期间被使用。在导入机制内部，它的功能与 `sys.path` 基本相同，即在导入期间提供一个模块搜索位置列表。`__path__` 必须是由字符串组成的可迭代对象，但它也可以为空。

> 包的 `__init__.py` 文件可以设置或更改包的 `__path__` 属性，而且这是在 **PEP 420** 之前实现命名空间包的典型方式。 随着 **PEP 420** 的引入，命名空间包不再需要提供仅包含 `__path__` 操控代码的 `__init__.py` 文件，导入机制会自动为命名空间包正确地设置 `__path__`。



导入包 `bar` 时，Python 解释器从 `sys.path` 的目录列表里查找子目录 `bar`，与[模块的搜索路径](#模块搜索路径)相同。



### 模块之间的相互导入

模块导入同一个包下的另一个模块或其名称有绝对导入和相对导入两种方法，下面给出相应的示例：

```python
package1
├───__init__.py
├───subpackage1
│   ├───__init__.py
│   ├───module1.py
│   └───module2.py
└───subpackage2
    ├───__init__.py
    ├───module3.py
    └───module4.py
```

```python
# module1.py
from . import module2                            # 相对导入同一个子包下的模块
from package1.subpackage1 import module2         # 绝对导入
import module2                                   # error

from .module2 import func4                       # 相对导入同一个子包下的模块的名称
from package1.subpackage1.module2 import func4   # 绝对导入
from module2 import func4                        # error

from ..subpackage2 import module4                # 相对导入另一个子包下的模块
from package1.subpackage2 import module4         # 绝对导入

from ..subpackage2.module4 import Class2         # 相对导入另一个子包下的模块的名称
from package1.subpackage2.module4 import Class2  # 绝对导入
```

一般建议使用绝对导入，因为相对导入的路径不直观。





# pip

```shell
$ pip install package            # 安装包(的最新版本)
$ pip install package=1.0.4      # 安装包的指定版本

$ pip install --upgrade package  # 升级包到最新版本

$ pip install -r requirements.txt   # 从requirements文件中安装

$ pip install package -i <url>   # 指定pypi镜像地址,默认为https://pypi.org/simple
                                 # 豆瓣 https://pypi.douban.com/simple/
                                 # 阿里云 https://mirrors.aliyun.com/pypi/simple/
                                 # 清华 https://pypi.tuna.tsinghua.edu.cn/simple
                                 
$ pip install -e .                 # 以"可编辑"模式从VCS安装项目;安装当前目录下的项目
$ pip install -e path/to/project   # 安装指定目录下的项目
```



`requirements.txt`是

