# enum——对枚举的支持

## 模块内容

### Enum

创建枚举常量的基类。

### IntEnum

创建 `int` 类枚举常量的基类。

### IntFlag

创建可与位运算符搭配使用，又不失去 `IntFlag` 成员资格的枚举常量的基类。

### Flag

创建可与位运算符搭配使用，又不失去 `Flag` 成员资格的枚举常量的基类。

### unique()

确保一个名称只绑定一个值的 Enum 类装饰器。

### auto

以合适的值代替 Enum 成员的实例。初始值默认从 1 开始。

## 定义枚举类

继承 `Enum` 类以定义枚举类，例如：

```python
>>> from enum import Enum
>>> class Color(Enum):        # `Color`是枚举类
    RED = 1                   # `Color.RED`是枚举类的成员,其中`RED`是名称,`1`是值
    GREEN = 2
    BLUE = 3
>>> print(Color.RED)          # 成员的打印结果
Color.RED
>>> Color.RED
<Color.RED: 1>
>>> type(Color.RED)           # 成员的类型
<enum 'Color'>
>>> isinstance(Color.RED, Color)
True
>>> print(Color.RED.name)     # 成员的名称
RED
>>> print(Color.RED.value)    # 成员的值
1
>>> list(Color)               # 迭代枚举类
[<Color.RED: 1>, <Color.GREEN: 2>, <Color.BLUE: 3>]
```

> 枚举类中定义的所有类属性将成为该枚举类的成员。
>
> 枚举类表示的是常量，因此建议成员名称使用大写字母；以单下划线开头和结尾的名称由枚举保留而不可使用。
>
> 尽管枚举类同样由 `class` 语法定义，但它并不是常规的 Python 类，详见 [How are Enums different?](https://docs.python.org/zh-cn/3/library/enum.html#how-are-enums-different)。

除了 `Color.RED`，成员还支持如下访问方式：

```python
>>> Color['RED']
<Color.RED: 1>
>>> Color(1)
<Color.RED: 1>
```

成员的值一般设定为整数、字符串等。若成员取何值并不重要，则可以使用 `auto()` 自动为成员分配值：

```python
>>> from enum import Enum, auto
>>> class Color(Enum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()
... 
>>> list(Color)
[<Color.RED: 1>, <Color.GREEN: 2>, <Color.BLUE: 3>]
```

`auto()` 的行为可以由重载 `_generate_next_value_()` 方法来定义：

```python
>>> class Color(Enum):
    def _generate_next_value_(name, start, count, last_values):  # 必须定义在任何成员之前
        return name
    RED = auto()
    GREEN = auto()
    BLUE = auto()
... 
>>> list(Color)
[<Color.RED: 'RED'>, <Color.GREEN: 'GREEN'>, <Color.BLUE: 'BLUE'>]
```

成员的值可哈希，因此成员可用于字典和集合：

```python
>>> apples = {}
>>> apples[Color.RED] = 'red delicious'
```

## 重复的成员值

两个成员的名称不能相同，但值可以相同，此时后定义的成员的名称将作为先定义的成员的别名：

```python
>>> class Shape(Enum):
    SQUARE = 2
    SQUARE = 3
...
TypeError: Attempted to reuse key: 'SQUARE'
>>> 
>>> class Shape(Enum):
    SQUARE = 2
    DIAMOND = 1
    CIRCLE = 3
    ALIAS_FOR_SQUARE = 2     # 作为`SQUARE`的别名
... 
>>> Shape.SQUARE
<Shape.SQUARE: 2>
>>> Shape.ALIAS_FOR_SQUARE
<Shape.SQUARE: 2>
>>> Shape(2)
<Shape.SQUARE: 2>
```

迭代枚举类时不会给出别名；枚举类的特殊属性 `__members__` 是从名称到成员的只读有序映射，其包含别名在内的所有名称：

```python
>>> list(Shape)
[<Shape.SQUARE: 2>, <Shape.DIAMOND: 1>, <Shape.CIRCLE: 3>]
>>> list(Shape.__members__.items())
[('SQUARE', <Shape.SQUARE: 2>), ('DIAMOND', <Shape.DIAMOND: 1>), ('CIRCLE', <Shape.CIRCLE: 3>), ('ALIAS_FOR_SQUARE', <Shape.SQUARE: 2>)]
>>> [name for name, member in Shape.__members__.items() if member.name != name]  # 找出所有别名
['ALIAS_FOR_SQUARE']
```

如果想要禁用别名，则可以使用装饰器 `unique`：

```python
>>> from enum import Enum, unique
>>> @unique
class Shape(Enum):
    SQUARE = 2
    DIAMOND = 1
    CIRCLE = 3
    ALIAS_FOR_SQUARE = 2
...
ValueError: duplicate values found in <enum 'Shape'>: ALIAS_FOR_SQUARE -> SQUARE
```

## 比较

成员之间按照标识值进行比较：

```python
>>> Color.RED is Color.RED
True
>>> Color.RED is Color.GREEN
False
>>> Color.RED == Color.RED
True
>>> Color.RED == Color.GREEN
False
```

成员之间的排序比较不被支持：

```python
>>> Color.RED < Color.BLUE
...
TypeError: '<' not supported between instances of 'Color' and 'Color'
```

成员与其它类型的实例的比较将总是不相等：

```python
>>> Color.RED == 1
False
```

## 枚举类的方法

枚举类是特殊的 Python 类，同样可以具有普通方法和特殊方法，例如：

```python
>>> class Mood(Enum):
    FUNKY = 1
    HAPPY = 3
    def describe(self):       # self here is the member
        return self.name, self.value
    def __str__(self):
        return 'my custom str! {0}'.format(self.value)
    @classmethod
    def favorite_mood(cls):   # cls here is the enumeration
        return cls.HAPPY
... 
>>> Mood.favorite_mood()
<Mood.HAPPY: 3>
>>> Mood.HAPPY.describe()
('HAPPY', 3)
>>> print(Mood.FUNKY)
my custom str! 1
```

## 继承枚举类

一个新的枚举类必须继承自一个既有的枚举类，并且父类不可定义有任何成员，因此禁止下列定义：

```python
>>> class MoreColor(Color):
    PINK = 17
......
TypeError: MoreColor: cannot extend enumeration 'Color'
```

但是允许下列定义：

```python
>>> class Foo(Enum):
    def some_behavior(self):
        pass
... 
>>> class Bar(Foo):
    HAPPY = 1
    SAD = 2
... 
```

## 功能性API

`Enum` 类属于可调用对象，它提供了以下功能性 API：

```python
>>> Animal = Enum('Animal', 'ANT BEE CAT DOG')
>>> Animal
<enum 'Animal'>
>>> Animal.ANT
<Animal.ANT: 1>
>>> Animal.ANT.value
1
>>> list(Animal)
[<Animal.ANT: 1>, <Animal.BEE: 2>, <Animal.CAT: 3>, <Animal.DOG: 4>]
```

`Enum` 的第一个参数是枚举的名称；第二个参数是枚举成员名称的来源，它可以是一个用空格分隔的名称字符串、名称序列、键值对二元组的序列或者名称到值的映射，最后两种选项使得可以为枚举任意赋值，而其他选项会自动以从 1 开始递增的整数赋值（使用 `start` 形参可指定不同的起始值）。返回值是一个继承自 `Enum` 的新类，换句话说，上述对 `Animal` 的赋值就等价于:

```python
>>> class Animal(Enum):
    ANT = 1
    BEE = 2
    CAT = 3
    DOG = 4
```

默认以 `1` 而以 `0` 作为起始数值的原因在于 `0` 的布尔值为 `False`，但所有枚举成员都应被求值为 `True`。

## IntEnum

`IntEnum` 是 `Enum` 的一个变种，同时也是 `int` 的一个子类。`IntEnum` 的成员可以与整数进行比较，不同 `IntEnum` 子类的成员也可以互相比较：

```python
>>> from enum import IntEnum
>>> class Shape(IntEnum):
    CIRCLE = 1
    SQUARE = 2
... 
>>> class Request(IntEnum):
    POST = 1
    GET = 2
... 
>>> Shape == 1
False
>>> Shape.CIRCLE == 1
True
>>> Shape.CIRCLE == Request.POST
True
```

`IntEnum` 成员的值在其它方面的行为都类似于整数：

```python
>>> ['a', 'b', 'c'][Shape.CIRCLE]
'b'
```

## IntFlag

`IntFlag` 变种同样基于 `int`，与 `IntEnum` 的不同之处在于 `IntFlag` 成员可以使用按位运算符进行组合并且结果仍然为 `IntFlag` 成员：

```python
>>> from enum import IntFlag
>>> class Perm(IntFlag):
    R = 4
    W = 2
    X = 1
... 
>>> Perm.R | Perm.W
<Perm.R|W: 6>
>>> type(Perm.R | Perm.W)
<enum 'Perm'>
>>> Perm.R in (Perm.R | Perm.W)
True
>>> Perm.R + Perm.W
6
>>> Perm.R | 8               # 与整数进行组合
<Perm.8|R: 12>
>>> type(Perm.R | 8)         # 仍为`IntFlag`成员
<enum 'Perm'>
```

`IntFlag` 和 `IntEnum` 的另一个重要区别在于如果值为 0，则其布尔值为 `False`：

```python
>>> Perm.R & Perm.X
<Perm.0: 0>
>>> bool(Perm.R & Perm.X)
False
```

## Flag

`Flag` 变种与 `IntFlag` 类似，成员可以使用按位运算符进行组合，但不同之处在于成员不可与其它 `Flag` 子类的成员或整数进行组合或比较。虽然可以直接指定值，但推荐使用 `auto` 选择适当的值。

```python
>>> from enum import Flag, auto
>>> class Color(Flag):
    BLACK = 0                    # 定义值为0的flag
    RED = auto()                 # 1
    BLUE = auto()                # 2
    GREEN = auto()               # 4
    WHITE = RED | BLUE | GREEN   # 定义作为多个flag的组合的flag
... 
>>> Color.RED & Color.GREEN
<Color.0: 0>
>>> bool(Color.RED & Color.GREEN)
False
>>> Color.BLACK
<Color.BLACK: 0>
>>> bool(Color.BLACK)
False
>>> Color.WHITE
<Color.WHITE: 7>

```
