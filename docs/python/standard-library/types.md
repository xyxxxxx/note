# types——动态类型创建和内置类型名称

`types` 模块为不能直接访问的内置类型定义了名称。

| 名称                                                   | 内置类型                                         |
| ------------------------------------------------------ | ------------------------------------------------ |
| `types.FunctionType`, `types.LambdaType`               | 用户自定义函数和 `lambda` 表达式创建的函数的类型 |
| `types.GeneratorType`                                  | 生成器类型                                       |
|                                                        |                                                  |
| `types.MethodType`                                     | 用户自定义实例方法的类型                         |
| `types.BuiltinFunctionType`, `types.BuiltinMethodType` | 内置函数和内置类型方法的类型                     |
|                                                        |                                                  |

最常见的用法是进行实例和子类检测：

```python
>>> def f(): pass
... 
>>> isinstance(f, types.FunctionType)            # 自定义函数
True
>>>
>>> isinstance(len, types.BuiltinFunctionType)   # 内置函数
True
```

除此之外，还可以用来动态创建内置类型：

```python
# 动态创建自定义实例方法并绑定到实例
>>> import types
>>> class Student(object):
    def set_name(self, name):
        self.name = name
... 
>>> bart = Student()
>>> bob = Student()
>>> bart.set_name('Bart')
>>> bob.set_name('Bob')
>>> def get_name(self):
    return self.name
... 
>>> bart.get_name = types.MethodType(get_name, bart)     # 将函数`get_name`动态绑定为实例`bart`的方法
>>> bart.get_name()                                      # 注意若类设定了`__slots__`则无法绑定
'Bart'
>>> dir(bart)
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'get_name', 'name', 'set_name']
>>> bob.get_name()                                       # 其它实例无法调用
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'Student' object has no attribute 'get_name'
>>> def set_name(self, name):
    self.name = name
    print('Succeeded.')
... 
>>> bart.set_name = types.MethodType(set_name, bart)     # 覆盖实例方法
>>> bart.set_name('Bartt')                               # 注意若实例方法设为只读则无法覆盖
Succeeded.
```

## typing——类型提示支持

> 注意：Python 运行时不强制执行函数和变量类型注解，但这些注解可用于类型检查器、IDE、静态检查器等第三方工具。

此模块为运行时提供了 PEP 484、PEP 526、PEP 544、PEP 586、PEP 589 和 PEP 591 规定的类型提示。最基本的支持由 `Any`，`Union`，`Tuple`，`Callable`，`TypeVar` 和 `Generic` 类型组成。有关完整的规范，请参阅 PEP 484。有关类型提示的简单介绍，请参阅 PEP 483。

函数接受并返回一个字符串，注解如下:

```python
def greeting(name: str) -> str:
    return 'Hello ' + name
```

在函数 `greeting` 中，参数 `name` 预期是 `str` 类型，并且返回 `str` 类型。

## 类型别名

可以为类型定义别名。在下面的例子中，`Vector` 和 `List[float]` 将被视为可互换的同义词:

```python
from typing import List
Vector = List[float]

def scale(scalar: float, vector: Vector) -> Vector:
    return [scalar * num for num in vector]

# typechecks; a list of floats qualifies as a Vector.
new_vector = scale(2.0, [1.0, -4.2, 5.4])
```

类型别名可用于简化复杂的类型签名。例如:

```python
from typing import Dict, Tuple, Sequence

ConnectionOptions = Dict[str, str]
Address = Tuple[str, int]
Server = Tuple[Address, ConnectionOptions]

def broadcast_message(message: str, servers: Sequence[Server]) -> None:
    ...
```

## 泛型

## 模块内容

### 泛型具象容器

#### ChainMap

`collections.ChainMap` 的泛型版本。

#### Counter

`collections.Counter` 的泛型版本。

#### DefaultDict

`collections.defaultdict` 的泛型版本。

#### Deque

`collections.deque` 的泛型版本。

#### Dict

`dict` 的泛型版本，适用于注解返回类型。注解参数时，最好使用 `Mapping` 等抽象容器类型。

```python
def count_words(text: str) -> Dict[str, int]:
    ...
```

#### FrozenSet

`builtins.frozenset` 的泛型版本。

#### List

`list` 的泛型版本，适用于注解返回类型。注解参数时，最好使用 `Sequence` 和 `Iterable` 等抽象容器类型。

```python
T = TypeVar('T', int, float)

def vec2(x: T, y: T) -> List[T]:
    return [x, y]

def keep_positives(vector: Sequence[T]) -> List[T]:
    return [item for item in vector if item > 0]
```

#### OrderedDict

`collections.OrderedDict` 的泛型版本。

#### Set

`builtins.set` 的泛型版本，适用于注解返回类型。注解参数时，最好使用 `AbstractSet` 等抽象容器类型。

### 抽象容器

#### AbstractSet

`collections.abc.Set` 的泛型版本。

#### Collection

`collections.abc.Collection` 的泛型版本。

#### Container

`collections.abc.Container` 的泛型版本。

#### Mapping

`collections.abc.Mapping` 的泛型版本。

#### Sequence

`collections.abc.Sequence` 的泛型版本。

### 特殊类型原语

#### Any

不受限的特殊类型，与所有类型兼容。

#### NoReturn

标记函数没有返回值的特殊类型，例如：

```python
from typing import NoReturn

def stop() -> NoReturn:
    raise RuntimeError('no way')
```

#### Union

联合类型，`Union[X, Y]` 表示非 X 即 Y。联合类型具有以下特征：

* 参数必须是其中某种类型

* 联合类型的嵌套会被展开，例如：

  ```python
  Union[Union[int, str], float] == Union[int, str, float]
  ```

* 仅有一种类型的联合类型就是该类型自身，例如：

  ```python
  Union[int] == int
  ```

* 重复的类型会被忽略，例如：

  ```python
  Union[int, str, int] == Union[int, str]
  ```
  
* 联合类型不能创建子类，也不能实例化

#### Optional

可选类型，`Optional[X]` 等价于 `Union[X, None]` 。

#### Tuple

元组类型，例如 `Tuple[X, Y]` 标注了一个二元组类型，其第一个元素的类型为 `X`，第二个元素的类型为 `Y`。

为表达一个同类型元素的变长元组，使用省略号字面量，如 `Tuple[int, ...]`。

单独的 `Tuple` 等价于 `Tuple[Any, ...]`，进而等价于 `tuple`。

#### Callable

可调用类型，例如 `Callable[[int], str]` 是一个函数，其接受一个 `int` 参数，返回一个 `str`。
