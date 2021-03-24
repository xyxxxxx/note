[toc]



OOP把对象作为程序的基本单元，一个对象包含了数据和操作数据的函数

面向对象的三大特点：**数据封装**，**继承**，**多态**

# 类与对象

## 类

```python
class Student(object):			# 类名通常首字母大写,继承object类
    def __init__(self, name, score): # 构造函数,第一个参数恒为self,表示创建的实例自身,之后的参数表示实例的属性
                                # 所有的属性均在构造函数中定义
        self.name = name
        self.score = score
    def print_score(self):		# 方法的第一个参数也恒为self		
        print('%s: %s' % (self.name, self.score))

bart = Student()				    # 创建对象
bart.name = 'Bart Simpson'	    # 对象属性赋值
bart.score = 59
bart = Student('Bart Simpson',59)	# 创建并初始化对象

print_score(bart)			    # 调用内部函数
bart.print_score()

```

**动态创建类**

```python
def fn(self, name='world'): #先定义函数
     print('Hello, %s.' % name)
Hello = type('Hello', (object,), dict(hello=fn)) 
# 创建Hello class,依次传入3个参数:
# class的名称；
# 继承的父类集合
# class的方法名称与函数绑定
```



## 定制类（类的特殊方法）

```python
#__call__()定义 实例() 返回值



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


#__member__()返回类的所有实例



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



## 元类metaclass





# 数据封装

类的属性和方法的可见性（即 Java 中的`private,protected,public`）由其名称决定：

+ `attr`：外部可以访问
+ `__attr`：外部不可访问
+ `_attr`：外部可以访问，但一般不建议这么做
+ `__attr__ `：特殊变量，外部可以访问

```python
class Student(object):			
    def __init__(self, name, score):	
        self.__name = name		# 定义为private变量,外部不可直接访问
        self._score = score
    def get_name(self):			
        return self.__name
    def get_score(self):		# getter
        return self.__score
    def set_score(self, score): # setter
        self.__score = score 
    def print_score(self):				
        print('%s: %s' % (self.name, self.score))
        
s = Student('Alice',90)
print(s.__name)             # error
print(getattr(s,'__name'))  # error
print(s.get_name())         # Alice
print(s._score)             # 90
print(getattr(s,'_score'))  # 90
print(s.get_score())        # 90
```





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

```python
class Animal(object):               # 抽象类用于定义不具体实现的父类，方法由子类实现
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



# 方法

## 实例方法、类方法和静态方法





类方法将类自己作为第一个实参，就像一个实例方法将实例自己作为第一个实参。请用以下习惯来声明类方法:

```python
class C:
    @classmethod
    def f(cls, arg1, arg2, ...): ...
```





# 属性

## 实例属性和类属性

```python
class Student(object):
    age=7		    # 类属性
    def __init__(self, name):
        self.name = name # 实例属性

s = Student('Bob')
s.score = 90		# py是动态语言,允许绑定任意属性
print(Student.age)	
print(s.age)		# 每个实例皆可访问类属性
s.age = 8
print(s.age)		# 实例属性覆盖类属性

# 属性操作函数
# getattr(obj, 'name')          # Same as obj.name
# setattr(obj, 'name', value)   # Same as obj.name = value
# delattr(obj, 'name')          # Same as del obj.name
# hasattr(obj, 'name')          # Tests if attribute exists
setattr(s,'name','Boc')
print(getattr(s,'name'))
```



## 动态绑定属性

```python
class Student(object):
    pass

s = Student()
s.name = 'Michael' 		# 动态给实例绑定一个属性

def set_age(self, age): # 定义方法
	self.age = age

from types import MethodType
s.set_age = MethodType(set_age, s) # 给实例绑定方法
s.set_age(25) 					   # 调用实例方法

Student.set_age = set_age	   # 给class绑定方法

```



## 属性限制条件

```python
class Student(object):
    __slots__ = ('name','_birth')     # 限定可以存在的属性
    def __init__(self, name, birth):
        self.name = name
        self._birth = birth

    @property  # 定义属性和getter
    def birth(self):
        return self._birth            # 属性_birth用birth调用

    @birth.setter  # 定义setter
    def birth(self, value):
        if value < 19000000:
            raise ValueError('Invalid birthday')  # 报错
        self._birth = value

s = Student('Alice', 20001111)
print(s.birth)
s.birth = 1
print(s.birth)
s.sex = 'F'                           # AttributeError

```





# 面向对象的实现

Python中的类和实例都是由`dict`实现：

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

