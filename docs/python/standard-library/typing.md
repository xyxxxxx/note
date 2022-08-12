# typing——类型提示支持

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
