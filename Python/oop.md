[toc]



OOP把对象作为程序的基本单元，一个对象包含了数据和操作数据的函数

面向对象的三大特点：**数据封装**，**继承**，**多态**



# 对象

```python
>>> a = 3
```

来看最简单的赋值语句，Python 首先在内存中新建整数对象`3`，然后将变量 `a` 指向这个对象，于是完成了变量的赋值。这也是为什么 Python 中不需要声明变量类型，因为它本来就**没有类型**，它更类似于**名称**或者**标记**。也可以将变量指向的对象类型看作变量暂时的类型，称其为整型变量或者浮点型变量等。

由于变量实质上只是一个标记，我们使用变量的目的不在于使用它的名称，而在于**使用它指向的对象**。

在给 `a` 赋值以后，我们就完成了变量 `a` 的声明，同时这个变量指向了一个对象 `3`，从现在开始 `a` 既是变量又是对象。那么如何区分 `a` 是作为变量还是对象呢？可以简单地理解为，当 `a` 作为左值时是变量，作为右值时是对象。

变量不需要声明类型是动态语言所具有的特征。这样做的好处在于，我们对变量的操作更加灵活。Python的一个赋值语句就涵盖了大量的操作与信息，这也是Python能比其他语言简短但运行速度不足的典型原因。

```python
>>> b = a
```

使用变量给变量赋值，实质上就是变量 `b` 指向变量 `a` 指向的对象。在这里由于整数对象 `3` 是不可变对象，所以看起来和 `b = 3` 并没有什么区别。但如果变量 `a` 指向可变对象：

```python
>>> a = []
>>> b = a
>>> b.append(1)
>>> a
[1]
```

就可以看到变量 `b` 和 `a` 指向了同一个对象。

Python 变量在许多方面就像 C 指针，例如：多个变量可以绑定到同一个对象；传递对象（变量为变量赋值）的代价很小，因为实现只传递一个指针；如果函数修改了作为参数传递的对象，调用者可以看到更改等等。





包含对象的对象（列表）



一切皆为对象





# 作用域

在Python中，使用一个变量时并不需要预先声明它，但是在真正使用它之前，它必须被绑定到某个内存对象（被定义、赋值）；这种变量名的绑定将在当前作用域中引入新的变量，同时屏蔽外层作用域中的同名变量。

Python有4种作用域，从内而外分别是：

- **L（Local）局部作用域**：函数或方法内部，定义的变量为局部变量。每当函数被调用时都会创建一个新的局部作用域。
- **E（Enclosing）嵌套作用域**：嵌套函数的上一级函数的局部作用域。主要是为了实现Python的闭包而增加的实现。
- **G（Global）全局作用域**：一个模块就是一个全局作用域，模块顶层声明的变量为全局变量。从外部来看，模块的全局变量就是一个模块对象的属性。
- **B（Built-in）内置作用域**

当在函数中使用未确定的变量名时，Python会按照优先级依次搜索4个作用域，以此来确定该变量名的意义。首先搜索局部作用域(L)，之后是上一层嵌套函数的嵌套作用域(E)，之后是全局作用域(G)，最后是内置作用域(B)。按这个查找原则，在第一处找到的地方停止。如果没有找到，则引发 `NameError` 错误。

```python
int = 0             # G
def outer():
    int = 1         # E
    def inner():
        int = 2     # L
        print(int)
    inner()
outer()
```

```
2
```

```python
int = 0             # G
def outer():
    int = 1         # E
    def inner():
        print(int)
    inner()
outer()
```

```
1
```

```python
int = 0             # G
def outer():
    def inner():
        print(int)
    inner()
outer()
```

```
0
```

```python
def outer():
    def inner():
        print(int)  # B
    inner()
outer()
```

```
<class 'int'>
```



Python 中只有模块（module），类（class）以及函数（def、lambda）才会引入新的作用域，其它的代码块（如 if/elif/else、try/except、for/while等）是不会引入新的作用域的，也就是说这些语句内定义的变量，外部也可以访问，例如：

```python
>>> k = 0
>>> for i in range(5):  # 循环体没有自己的局部作用域
...   j = i + 1         # i, j不是局部变量
...   k += j
...
>>> print(i)            # 循环结束后仍然存在
4
>>> print(j)
5
>>> print(k)
15
```



如果想要在局部作用域中修改全局变量，可以使用 `global` 关键字声明：

```python
i = 0             # G
def outer():
    i = 1         # E
    def inner():
        global i
        i = 2     # G
    inner()
outer()
print(i)
```

```
2
```



如果想要在局部作用域中修改嵌套作用域的变量，可以使用 `nonlocal` 关键字声明：

```python
i = 0             # G
def outer():
    i = 1         # E
    def inner():
        nonlocal i
        i = 2     # E
    inner()
    print(i)
outer()
```

```
2
```





# 类

## 定义类

定义一个简单的学生类并实例化：

```python
#           类名    父类
>>> class Student(object):			         # 类名首字母大写,遵循驼峰命名法;object类是Python所有类的基类,默认继承object
    num = 0                              # 类属性,下面定义的函数也是类属性
    def __init__(self, name, score):     # 构造函数,第一个参数恒为self,表示创建的实例自身
        self.name = name                 # 定义实例属性
        self.score = score
        Student.num += 1                 # 修改类属性
    def get_score(self):		             # 实例方法,第一个参数恒为self
        print('%s: %s' % (self.name, self.score))
...
>>> bart = Student('Bart Simpson', 59)    # 实例化(创建对象),调用`__init__()`方法
```

可以使用内置的 `type()` 函数动态创建类：

```python
>>> X = type('X', (), dict(a=1, f=abs))
>>> # 相当于
>>> class X:
...     a = 1
...     f = abs
```

详见标准库-内置函数-type。



## 属性

### 实例属性和类属性

```python
>>> class Student(object):
    num = 0
    def __init__(self, name, score):
        self.name = name
        self.score = score
        Student.num += 1
...
>>> bart = Student('Bart Simpson', 59)
>>> 
>>> bart.score         # 实例属性
59
>>> bart.score = 60    # 修改实例属性
>>> bart.score
60
>>> bart.age = 10      # Python是动态语言,允许动态绑定任意属性
>>> bart.age
10
>>> del bart.age       # 删除实例属性
>>> bart.age
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'Student' object has no attribute 'age'
>>> 
>>> Student.num        # 类属性
1
>>> Student.num = 2    # 修改类属性
>>> Student.num
2
>>> bart.num           # 实例可以访问类属性
2
>>> bart.num = 0       # 同名的实例属性会屏蔽类属性
>>> bart.num
0
>>> Student.age = 10
>>> Student.age        # 动态绑定属性
10
>>> del Student.age
>>> Student.age
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: type object 'Student' has no attribute 'age'
```

也可以使用内置的属性操作函数完成相同的操作：

```python
>>> getattr(bart, 'score')
59
>>> setattr(bart, 'score', 60)
>>> bart.score
60
>>> setattr(bart, 'age', 10)
>>> getattr(bart, 'age')
10
>>> delattr(bart, 'age')
>>> getattr(bart, 'age')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'Student' object has no attribute 'age'
```



### 可见性

类和实例的属性的可见性（即 C, Java 中的`private, protected, public`）由其名称决定：

+ `attr`：外部（指其它模块）可以访问
+ `__attr`：外部不可访问
+ `_attr`：外部可以访问，但一般不建议这么做
+ `__attr__ `：特殊属性，外部可以访问

```python
>>> class Student(object):
    def __init__(self, name, score, rank):  # 特殊属性,外部可以访问
        self.name = name                    # 外部可以访问
        self._score = score                 # 外部可以访问,但不推荐
        self.__rank = rank                  # 外部不可访问
    def get_score(self):
        print('%s: %s %s' % (self.name, self._score, self.__rank))   # 可以从内部访问内部属性
>>> bart = Student('Bart Simpson', 59, 77)
>>> bart.name                               # 外部可以访问
'Bart Simpson'
>>> bart._score                             # 外部可以访问,但不推荐
59
>>> bart.__rank                             # 外部不可访问,引发`AttributeError`
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'Student' object has no attribute '__rank'
>>> bart.get_score()                        # 调用方法访问内部属性
Bart Simpson: 59 77
```



### 限制属性

+ 类属性 `__slots__` 限定了允许存在的实例属性，详见[\__slots__](#\__slots__)。
+ 内置装饰器 `@property` 将方法变成属性调用，详见标准库-内置函数-property。

```python
class Student(object):
    __slots__ = ('name','_score')      # 限定可以存在的属性
    def __init__(self, name, score):
        self.name = name
        self._score = score

    @property                          # 定义属性和getter
    def score(self):
        return self._score             # 属性_score用score调用

    @score.setter                      # 定义setter
    def score(self, value):
        if value < 19000000:           # 设置检查条件
            raise ValueError('Invalid score.')
        self._score = value
```

```python
class Student(object):
    """class Student"""
    def __init__(self, name, score):
        self.name = name
        self._score = score
    @property                          # 定义属性和getter
    def score(self):
        """property score"""
        return self._score             # 属性_score用score调用
```







### 实例的特殊属性







#### \__class__

对象所属的类。



#### \__dict__

一个字典对象，用于存储实例的（可写）属性。

```python
>>> class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score
... 
>>> bart = Student('Bart Simpson', 59)
>>> bart.__dict__
{'name': 'Bart Simpson', 'score': 59}
>>> bart.age = 10
>>> bart.__dict__
{'name': 'Bart Simpson', 'score': 59, 'age': 10}    # 保存实例的属性
```



#### \__name__

类、函数、方法、描述器或生成器实例的名称。



#### \__slots__

类属性，可赋值为字符串、字符串序列或其它可迭代对象。`__slots__` 会为已声明的变量保留空间，并阻止自动为每个实例创建 `__dict__` 和 `__weakref__`。

+ 当继承自一个未定义 `__slots__` 的类时，实例的 `__dict__` 和 `__weakref__` 属性将总是可以访问。
+ 由于没有 `__dict__` 属性，实例不能给未在 `__slots__` 定义中列出的新属性赋值，尝试给一个未列出的属性赋值将引发 `AttributeError`。若要为实例动态绑定属性，就要将 `'__dict__'` 加入到 `__slots__` 声明的字符串序列中。
+ 由于没有 `__weakref__` 属性，定义了 `__slots__` 的类不支持对其实例的弱引用。若需要弱引用支持，就要将 `'__weakref__'` 加入到 `__slots__` 声明的字符串序列中。
+ `__slots__` 声明的作用不只限于定义它的类。在父类中声明的 `__slots__` 在其子类中同样可用。不过子类不会阻止创建 `__dict__` 和 `__weakref__`，除非它们也定义了 `__slots__` (其中应仅包含*额外*的变量声明)。
+ 相比使用 `__dict__` 此方式可以显著地节省空间，并且显著地提升属性查找速度。

```python
>>> class Student(object):
    __slots__ = ('name', 'score')
    def __init__(self, name, score):
        self.name = name
        self.score = score
... 
>>> bart = Student('Bart Simpson', 59)
>>> bart.__dict__
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'Student' object has no attribute '__dict__'   # 阻止创建 `__dict__`
>>> bart.age = 10
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'Student' object has no attribute 'age'        # 不能动态绑定属性
```

```python
>>> class Student(object):
    __slots__ = ('name', 'score', '__dict__')                  # 声明 `__dict__`
    def __init__(self, name, score):
        self.name = name
        self.score = score
... 
>>> bart = Student('Bart Simpson', 59)
>>> bart.__dict__
{}
>>> bart.age = 10                                              # 可以动态绑定属性
>>> bart.__dict__
{'age': 10}                                                    # 保存 `__slots__` 声明的变量以外的属性
```







```python
>>> dir('abc')
['__add__', '__class__', '__contains__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getnewargs__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__mod__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__rmod__', '__rmul__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'capitalize', 'casefold', 'center', 'count', 'encode', 'endswith', 'expandtabs', 'find', 'format', 'format_map', 'index', 'isalnum', 'isalpha', 'isascii', 'isdecimal', 'isdigit', 'isidentifier', 'islower', 'isnumeric', 'isprintable', 'isspace', 'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'maketrans', 'partition', 'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit', 'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase', 'title', 'translate', 'upper', 'zfill']
```







## 函数

类中定义的函数也是类属性（即函数对象）。



### 方法

**方法(method)**是从属于对象的函数，第一个参数为对象自身。调用方法 `object.method(*args)` 实际上就相当于 `Class.method(object, *args)`。

了解方法的运作原理有助于加深对方法的理解。当一个实例的非数据属性被引用时，将搜索实例所属的类，如果被引用的属性名称表示一个有效的类属性中的函数对象，会通过打包实例对象和查找到的函数对象到一个抽象对象的方式来创建方法对象。 当附带参数列表调用方法对象时，将基于实例对象和参数列表构建一个新的参数列表，并使用这个新参数列表调用相应的函数对象。



方法定义并非一定要包含在类定义之内，将一个函数对象赋值给一个变量也是可以的，例如：

```python
def get_score_impl(self):
    print('%s: %s' % (self.name, self._score))

class Student(object):
    def __init__(self, name, score):
        self.name = name
        self._score = score
    get_score = get_score_impl            # 函数对象赋值给类属性
```

但是不建议这么做，因为这样只会让读程序的人感到迷惑。



### 类方法和静态方法

**类方法(class method)**将类自己作为第一个实参，就像一个实例方法将实例自己作为第一个实参。类方法使用 `@classmethod` 装饰器声明，例如：

```python
>>> class Student(object):
    num = 0                       # 类属性
    def __init__(self, name, score):
        self.name = name
        self.score = score
        Student.num += 1
    def get_score(self):
        print('%s: %s' % (self.name, self.score))
    @classmethod
    def get_num(cls):             # 类方法,第一个参数恒为cls,表示创建的实例自身
        print(cls.num)
>>> bart = Student('Bart Simpson', 59)
>>> Student.get_num()             # 类调用类方法
1
>>> bart.get_num()                # 实例调用类方法
1
```



在开发中，我们常常需要定义一些方法，这些方法跟类有关，但在实现时并不需要引用类或者实例，例如设置环境变量，修改另一个类的属性等，这个时候我们可以使用**静态方法(static method)**。静态方法使用 `@staticmethod` 装饰器声明，例如：

```python
>>> class Student(object):
    num = 0                       # 类属性
    def __init__(self, name, score):
        self.name = name
        self.score = score
        Student.num += 1
    def get_score(self):
        print('%s: %s' % (self.name, self.score))
    @staticmethod
    def description():            # 静态方法,不需要类或实例作为参数
        print('in education')
... 
>>> bart = Student('Bart Simpson', 59)
>>> Student.description()         # 类调用静态方法
in education
>>> bart.description()            # 实例调用类方法
in education
```



### 方法的实现





### 动态绑定方法

参见标准库-types。







# 继承和多态

```python
class Animal(object):     # 类默认继承自object类
    def run(self):
        print('Animal is running...')

class Dog(Animal):
    def eat(self):
        print('Eating meat...')   

class Cat(Animal):
    def run(self):
        print('Cat is running...')
        super().run()     # 调用父类的方法
        
dog = Dog()
dog.run()	# 继承父类的方法
dog.eat()

cat = Cat()
cat.run()	# 覆盖父类的方法,即多态
```

```python
class Animal:
    def __init__(self, name, age):
        self.name = name
        self.age = age

class Dog(Animal):
    def __init__(self, name, age, price):
        super().__init__(name, age)   # 必须先调用父类的构造函数
        self.price = price
```



## 多重继承

```python
class Animal(object):
    pass
class Mammal(Animal):
    pass
class Bird(Animal):
    pass

class RunnableMixIn(object):
    def run(self):
        print('Running...')
class FlyableMixIn(object):
    def fly(self):
        print('Flying...')
        
class Dog(Mammal, RunnableMixIn):	# 继承主线Mammal,附加RunnableMixIn
    pass

print(Dog.__mro__)                  # 查看方法解析顺序MRO,也包含了继承信息
# (<class '__main__.Dog'>, <class '__main__.Mammal'>, <class '__main__.Animal'>, <class '__main__.RunnableMixIn'>, <class 'object'>)
```

一般不建议使用多重继承，但下面是多重继承的一种使用情形：

```python
class Dog:
    def noise(self):
        return 'Bark'

    def chase(self):
        return 'Chasing!'
    
class Bike:
    def noise(self):
        return 'On Your Left'

    def pedal(self):
        return 'Pedaling!'
    
class Loud:
    def noise(self):
        return super().noise().upper()
    
class LoudDog(Loud, Dog):
    pass

class LoudBike(Loud, Bike):
    pass

d = LoudDog()
print(d.noise())         # BARK
print(LoudDog.__mro__)   # (<class '__main__.LoudDog'>, <class '__main__.Loud'>, <class '__main__.Dog'>, <class 'object'>)
'''
为什么?首先由定义class LoudDog(Loud, Dog),LoudDog的mro中Loud在Dog之前,
因此调用了Loud的super().noise().upper().然后super()指向mro的下一个类即
Dog,再调用其noise()方法.
'''
```



## 抽象类

抽象类用于定义不具体实现的父类，方法由子类实现

```python
class Animal(object):
    def run(self):
        raise NotImplementedError()
    def eat(self):
        raise NotImplementedError()
        
class Dog(Animal):
    def run(self):
        print('Dog is running...')
    def eat(self):
        print('Dog is eating...')
```







# 高级类

## 类的特殊属性

一个类可以通过定义具有特殊名称的方法来实现由特殊语法所引发的特定操作，这是 Python 实现操作符重载的方式，允许每个类自行定义基于操作符的特定行为。例如，如果一个类定义了名为 `__getitem__()` 的方法，并且 `x` 为该类的一个实例，则 `x[i]` 基本就等同于 `type(x).__getitem__(x, i)`。

当没有定义特殊方法，或将特殊方法设为 `None` 时，对应的操作将不可用。例如，将一个类的 `__iter__()` 设为 `None`，则该类就是不可迭代的，因此对其实例调用 `iter()` 将引发一个 `TypeError`。



### 基本定制

#### \__new__()

调用以创建一个 *cls* 类的新实例。`__new__()` 是一个静态方法，它会将所请求实例所属的类作为第一个参数，其余的参数会被传递给对象构造器表达式。`__new__()` 的返回值应为新对象实例 (通常是 *cls* 的实例)。

如果 `__new__()` 未返回一个 *cls* 的实例，则新实例的 `__init__()` 方法就不会被执行。

```python
# 通过__new__()实现单例模式
class NewInt(object):
    _singleton = None
    def __new__(cls, *args, **kwargs):
        if not cls._singleton:
            cls._singleton = object.__new__(cls, *args, **kwargs)
        return cls._singleton

new1 = NewInt()
new2 = NewInt()
print(new1)
print(new2)
```

```
<__main__.NewInt object at 0x000002BB02FF6080>
<__main__.NewInt object at 0x000002BB02FF6080>
```





#### \__init__()

在实例 (通过 `__new__()`) 被创建之后，返回调用者之前调用，用于初始化实例的属性。其参数与传递给对象构造器表达式的参数相同。如果一个派生类的基类也有 `__init__()` 方法，就必须显式地调用它以确保实例基类部分的正确初始化，例如 `super().__init__([args...])`.

因为对象是由 `__new__()` 和 `__init__()` 协作构造完成的 (由 `__new__()` 创建，并由 `__init__()` 定制)，所以 `__init__()` 返回的值只能是 `None`，否则会在运行时引发 `TypeError`。



#### \__del__()

在实例将被销毁时调用。 这还被称为终结器或析构器（不适当）。 如果一个派生类的基类也有 `__del__()` 方法，就必须显式地调用它以确保实例基类部分的正确清除。



#### \__repr__()

被内置函数 `repr()` 调用以返回对象的“标准”字符串表示，返回一个字符串。

此方法通常被用于调试，因此应确保其表示的内容包含对于开发者有用且全面的信息。

```python
>>> class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score
    def get_score(self):
        print('%s: %s' % (self.name, self.score))
    def __repr__(self):
        return 'Student {:s} with score {:d}.'.format(self.name, self.score)
... 
>>> bart = Student('Bart Simpson', 59)
>>> bart
Student Bart Simpson with score 59.
```



#### \__str__()

被内置函数 `str()`, `format()` 或 `print()` 调用以返回对象的格式良好的字符串表示，返回一个字符串。

```python
>>> class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score
    def get_score(self):
        print('%s: %s' % (self.name, self.score))
    def __str__(self):
        return 'Student {:s} with score {:d}.'.format(self.name, self.score)
... 
>>> bart = Student('Bart Simpson', 59)
>>> print(bart)
Student Bart Simpson with score 59.
```



#### \__format__()

被内置函数 `format()`, 格式化字符串的求值或 `str.format()` 调用以返回对象的格式化字符串表示，返回一个字符串。

```python
>>> class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score
    def get_score(self):
        print('%s: %s' % (self.name, self.score))
    def __format__(self, format_spec):    # 返回字符串
        if format_spec == 'd':
            return str(self.score)
        elif format_spec == 's':
            return self.name
        else:
            msg =  "Unknown format code '{:s}' for object of type '{:s}'".format(
                format_spec, self.__class__.__name__)
            raise ValueError(msg)
... 
>>> bart = Student('Bart Simpson', 59)
>>> '{:d}'.format(bart)
'59'
>>> '{:b}'.format(bart)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 1, in __format__
ValueError: Unknown format code 'b' for object of type 'Student'
```



#### \_\_lt\_\_(), \_\_le\_\_(), \_\_eq\_\_(), \_\_ne\_\_(), \_\_gt\_\_(), \_\_ge\_\_()

+ `x<y` 调用 `x.__lt__(y)`
+ `x<=y` 调用 `x.__le__(y)`
+ `x==y` 调用 `x.__eq__(y)`
+ `x!=y` 调用 `x.__ne__(y)`
+ `x>y` 调用 `x.__gt__(y)`
+ `x>=y` 调用 `x.__ge__(y)`

按照惯例，比较结果通常作为判断条件，返回一个布尔值。

在默认情况下，*object* 通过使用 `is` 来实现 `__eq__()`。对于 `__ne__()`，默认会委托给 `__eq__()` 并对结果取反。比较运算符之间没有其他隐含关系或默认实现，例如 `(x<y or x==y)` 为真并不意味着 `x<=y` 为真。

```python
>>> class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score
    def get_score(self):
        print('%s: %s' % (self.name, self.score))
    def __lt__(self, other):
        return self.score < other.score
... 
>>> bart = Student('Bart Simpson', 59)
>>> bob = Student('Bob Simpson', 61)
>>> bart < bob
True
```



#### \__bool__()

在对象进行真值检测或被内置函数 `bool()` 调用时调用，返回一个布尔值。如果未定义此方法，则会查找并调用 `__len__()` 并在其返回非零值时视对象的逻辑值为真。如果一个类既未定义 `__len__()` 也未定义 `__bool__()` 则视其所有实例的逻辑值为真。



### 自定义属性访问

#### \__getattr__()

当默认属性访问因引发 `AttributeError` 而失败时被调用（可能是调用 `__getattribute__()` 时由于 *name* 不是一个实例属性或 `self` 的类关系树中的属性而引发了 `AttributeError`；或者是对 *name* 特性属性调用 `__get__()` 时引发了 `AttributeError`）。此方法应当返回（找到的）属性值或是引发一个 `AttributeError` 异常。



#### \__getattribute__()

此方法会无条件地被调用以实现对类实例属性的访问。如果类还定义了 `__getattr__()`，则后者不会被调用，除非 `__getattribute__()` 显式地调用它或是引发了 `AttributeError`。此方法应当返回（找到的）属性值或是引发一个 `AttributeError` 异常。为了避免此方法中的无限递归，其实现应该总是调用具有相同名称的基类方法来访问它所需要的任何属性，例如 `object.__getattribute__(self, name)`。



#### \__setattr__()

```python
object.__setattr__(self, name, value)
```

此方法在一个属性被尝试赋值时被调用。这个调用会取代正常机制（即将值保存到实例字典）。*name* 为属性名称， *value* 为要赋给属性的值。

```python
>>> class Student(object):   # 正常赋值机制
    def __init__(self, name, score):
        self.name = name
        self.score = score
... 
>>> bart = Student('Bart Simpson', 59)
>>> bart.age = 10            # 将值保存到实例字典
>>> bart.__dict__
{'name': 'Bart Simpson', 'score': 59, 'age': 10}
>>> 
>>> class Student(object):   # 自定义赋值机制
    def __init__(self, name, score):
        self.name = name
        self.score = score
    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)
        print('To assign: {} = {}'.format(key, value))
... 
>>> bart = Student('Bart Simpson', 59)
To assign: name = Bart Simpson
To assign: score = 59
>>> bart.age = 10
To assign: age = 10
>>> bart.__dict__
{'name': 'Bart Simpson', 'score': 59, 'age': 10}
```



#### \__delattr__()

类似于 `__setattr__()` 但其作用为删除而非赋值。



#### \__dir__()

被内置函数 `dir()` 调用，返回值必须为一个序列。`dir()` 会把返回的序列转换为列表并对其排序。



### 模拟可调用对象

#### \__call__()

在对象作为一个函数被调用时调用。如果定义了此方法，则 `x(arg1, arg2, ...)` 就大致可以被改写为 `type(x).__call__(x, arg1, ...)`。



### 模拟容器类型

#### \__len__()

被内置函数 `len()` 调用以返回对象的长度，返回一个非负整数。



#### \_\_getitem\_\_(), \_\_setitem\_\_(), \_\_delitem\_\_()

被语法 `object[key]` 调用以返回/设置/删除对象对应键的值。

对于序列类型，接受的键应为整数和切片对象，负数索引（如果类想要模拟序列类型）的特殊解读在 `__getitem__()` 方法中定义。如果 *key* 的类型不正确则应引发 `TypeError` 异常；如果为序列索引集范围以外的值（在进行任何负数索引的特殊解读之后）则应引发 `IndexError` 异常。

对于映射类型，如果 *key* 找不到（不在容器中）则应引发 `KeyError` 异常。

```python
>>> class Student(object):
    def __init__(self, name, score1, score2, score3):
        self.name = name
        self.score1 = score1
        self.score2 = score2
        self.score3 = score3
    def __getitem__(self, key):
        if isinstance(key, slice):
            print(slice)
            return [self[i] for i in slice]
        elif isinstance(key, int):
            if key < 0 :               # Handle negative indices
                key += 3
            if key < 0 or key >= 3:
                raise IndexError("The index {} is out of range.".format(key))
            return self.get_data(key)  # Get the data from elsewhere
        else:
            raise TypeError("Invalid argument type.")
    def get_data(self, key):
        if key == 0:
            return self.score1
        if key == 1:
            return self.score2
        if key == 2:
            return self.score3
        return
>>> bart = Student('Bart Simpson', 59, 60, 61)
>>> bart[0]
59      
```



#### \__iter__()

在需要为容器对象创建迭代器时被调用，返回一个迭代器对象，它能够逐个迭代容器中的所有对象。

详见



#### \__reverse__()

被内置函数 `reverse()` 调用以实现逆向迭代，返回一个迭代器对象，它能够逆序逐个迭代容器中的所有对象。

详见



#### \__contains__()

```python
object.__contains__(self, item)
```

调用此方法以实现成员检测运算符。如果 *item* 是 *self* 的成员则应返回真，否则返回假。对于映射类型，此检测应基于映射的键而不是值或者键值对。

对于未定义 `__contains__()` 的对象，成员检测将首先尝试通过 `__iter__()` 进行迭代，然后再使用 `__getitem__()`的旧式序列迭代协议。



### 模拟数学类型

#### \_\_add\_\_(), \_\_sub\_\_(), \_\_mul\_\_(), \_\_matmul\_\_(), \_\_truediv\_\_(), \_\_floordiv\_\_(), \_\_mod\_\_(), \_\_divmod\_\_(), \_\_pow\_\_(), \_\_lshift\_\_(), \_\_rshift\_\_(), \_\_and\_\_(), \_\_xor\_\_(), \_\_or\_\_(),

基本数学运算方法：

+ `x+y` 调用 `x.__add__(y)`
+ `x-y` 调用 `x.__sub__(y)`
+ `x*y` 调用 `x.__mul__(y)`
+ `x@y` 调用 `x.__matmul__(y)`
+ `x/y` 调用 `x.__truediv__(y)`
+ `x//y` 调用 `x.__floordiv__(y)`
+ `x%y` 调用 `x.__mod__(y)`
+ `divmod(x, y)` 调用 `x.__divmod__(y)`
+ `pow(x, y)`, `x**y` 调用 `x.__pow__(y)`
+ `x<<y` 调用 `x.__lshift__(y)`
+ `x>>y` 调用 `x.__rshift__(y)`
+ `x&y` 调用 `x.__and__(y)`
+ `x^y` 调用 `x.__xor__(y)`
+ `x|y` 调用 `x.__or__(y)`







### 





















```python
#__call__() 实例() 返回值



#__getattr__()为不存在的属性设定返回值
class Chain(object):
    def __init__(self, path=''):
        self._path = path
    def __getattr__(self, path):	#参数作为str传入
        return Chain('%s/%s' % (self._path, path))

Chain().status.user.timeline.list

#__getitem__()取类的实例
class Fib(object):
    def __getitem__(self, n):	#n为序数
        a, b = 1, 1
        for x in range(n):
            a, b = b, a + b
        return a

f=Fib()    
f[10]		#调用方法
#也可以传入slice等类型，但需要更多工作    
    
    
#__init__()定义构造函数


#__iter__()定义迭代器
class Fib(object):
    def __init__(self):
        self.a, self.b = 0, 1 # 初始化两个计数器a，b
    def __iter__(self):
        return self # 实例本身就是迭代对象，故返回自己
    def __next__(self):
        self.a, self.b = self.b, self.a + self.b # 计算下一个值
        if self.a > 100000: # 退出循环的条件
            raise StopIteration()
        return self.a # 返回下一个值

for n in Fib():	#先后进入init,iter,next,next,...
     print(n)	
    
    
#__len__()




#__slots__()定义类允许绑定的属性
class Student(object):
    __slots__ = ('name', 'age') #tuple

    
#__str__()定义打印实例返回值, __repr__()定义实例直接返回值
class Date(object):
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    # Used with `str()`,显示简单信息
    # print(i)或 print(str(i))打印该信息
    def __str__(self):
        return f'{self.year}-{self.month}-{self.day}'

    # Used with `repr()`,显示详细信息
    # print(repr(i))打印该信息
    def __repr__(self):
        return f'Date({self.year},{self.month},{self.day})'


```

实际上数学运算和数组运算都是调用了特殊方法，因此实现这些方法即能使用符号运算：

```python
a + b       a.__add__(b)
a - b       a.__sub__(b)
a * b       a.__mul__(b)
a / b       a.__truediv__(b)
a // b      a.__floordiv__(b)
a % b       a.__mod__(b)
a << b      a.__lshift__(b)
a >> b      a.__rshift__(b)
a & b       a.__and__(b)
a | b       a.__or__(b)
a ^ b       a.__xor__(b)
a ** b      a.__pow__(b)
-a          a.__neg__()
~a          a.__invert__()
abs(a)      a.__abs__()

len(x)      x.__len__()
x[a]        x.__getitem__(a)
x[a] = v    x.__setitem__(a,v)
del x[a]    x.__delitem__(a)

class Sequence:
    def __len__(self):
        pass
    def __getitem__(self,a):
        pass
    def __setitem__(self,a,v):
        pass
    def __delitem__(self,a):
        pass
```



## 枚举类

```python
#Enum()定义枚举类
from enum import Enum

Month = Enum('Month', ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))		#各实例默认从1开始赋值

#自定义枚举类
from enum import Enum, unique
@unique					#检查没有重复值
class Weekday(Enum):	#继承Enum
    Sun = 0 			#Sun的value被设定为0
    Mon = 1
    Tue = 2
    Wed = 3
    Thu = 4
    Fri = 5
    Sat = 6
    
day1=Weekday.Mon
day1=Weekday(1)
print(day1)
print(day1.value)
```



## 元类















# 获取对象信息

```python
# type()判断对象类型
type(123)		# int
type('123')		# str

type('abc')==type('123')	# True
type('abc')==str			# True
type('abc')==type(123)		# False

type(abs)==types.BuiltinFunctionType		# 判断函数类型
type(lambda x: x)==types.LambdaType
```

```python
# isinstance()判断对象类型

# 继承关系:object -> Animal -> Dog -> Husky
h = Husky()
isinstance(h,Husky)		# True,继承父类的数据类型
isinstance(h,Dog)		# True
isinstance(h,Animal)	# True

isinstance([1, 2, 3], (list, tuple))	# True,或型判断
```

```python
# 操作对象属性
class Student(object):			
    def __init__(self, name, score):	
        self.__name = name		# 定义为private变量,外部不可直接访问
        self.__score = score

bart=Student()        
dir(bart)					# 返回对象所有属性和方法
hasattr(bart,'name')		# 判断有无某属性
setattr(bart,'age',10)		# 创建属性
getattr(bart,'sex',404)		# 获取属性,不存在时返回404

```

```python
# 判断两个变量是否指向同一对象
a = [1,2,3]
b = a
c = [1,2,3]
a is b       # True
id(a)        # 140295002377160
id(b)        # 140295002377160
a is c       # False
```





# 面向对象的实现

Python中的类和实例都是由 `dict` 实现：

```python
class Animal:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def run(self):
        print('Animal is running...')
        
a = Animal('white',1)
print(a.__self__)
print(Animal.__dict__)

# {'name': 'white', 'age': 1}
# {'__module__': '__main__', '__init__': <function Animal.__init__ at 0x7f3b1db09620>, 'run': <function Animal.run at 0x7f3b1db098c8>, '__dict__': <attribute '__dict__' of 'Animal' objects>, '__weakref__': <attribute '__weakref__' of 'Animal' objects>, '__doc__': None}

```

