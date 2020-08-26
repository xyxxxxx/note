OOP把对象作为程序的基本单元，一个对象包含了数据和操作数据的函数

面向对象的三大特点：**数据封装**，**继承**，**多态**

# class & instance

## 类

```python
class Student(object):			#类名通常首字母大写,继承object类
    def __init__(self, name, score):	#构造函数,第一个参数恒为self,表示创建的实例自身,之后的参数表示实例的属性
        self.name = name
        self.score = score
    def print_score(self):				
        print('%s: %s' % (self.name, self.score))

bart=Student()				#创建对象
bart.name='Bart Simpson'	#对象属性赋值
bart.score=59
bart=Student('Bart Simpson',59)	#创建并初始化对象

print_score(bart)			#调用内部函数
bart.print_score()

```

**动态创建类**

```python
def fn(self, name='world'): #先定义函数
     print('Hello, %s.' % name)
Hello = type('Hello', (object,), dict(hello=fn)) 
#创建Hello class,依次传入3个参数:
#class的名称；
#继承的父类集合
#class的方法名称与函数绑定
```



## 定制类

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
class Student(object):
    def __str__(self):
        return 'Student object (name: %s)' % self.name
	__repr__=__str__



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

```python
class Student(object):			
    def __init__(self, name, score):	
        self.__name = name		#定义为private变量,外部不可直接访问
        self.__score = score
    def get_name(self):			
        return self.__name
    def get_score(self):		#访问private变量的函数
        return self.__score
    def set_score(self, score): #修改private变量的函数
        self.__score = score 
    def print_score(self):				
        print('%s: %s' % (self.name, self.score))
```





# 继承和多态

```python
class Animal(object):
    def run(self):
        print('Animal is running...')

class Dog(Animal):
    def eat(self):
        print('Eating meat...')   

class Cat(Animal):
    def run(self):
        print('Cat is running...')        
        
dog=Dog()
dog.run()	#继承父类的函数
dog.eat()

cat=Cat()
cat.run()	#覆盖父类的函数,即多态
```





# 多重继承

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
        
class Dog(Mammal, RunnableMixIn):	#继承主线Mammal,附加RunnableMixIn
    pass        
```





# 获取对象信息

```python
#type()判断对象类型
type(123)		#int
type('123')		#str

type('abc')==type('123')	#True
type('abc')==str			#True
type('abc')==type(123)		#False

type(abs)==types.BuiltinFunctionType		#判断函数类型
type(lambda x: x)==types.LambdaType
```

```python
#isinstance()判断对象类型

#继承关系:object -> Animal -> Dog -> Husky
h = Husky()
isinstance(h,Husky)		#True,继承父类的数据类型
isinstance(h,Dog)		#True
isinstance(h,Animal)	#True

isinstance([1, 2, 3], (list, tuple))	#True,或型判断
```

```python
#操作对象属性
class Student(object):			
    def __init__(self, name, score):	
        self.__name = name		#定义为private变量,外部不可直接访问
        self.__score = score

bart=Student()        
dir(bart)					#返回对象所有属性和方法
hasattr(bart,'name')		#判断有无某属性
setattr(bart,'age',10)		#创建属性
getattr(bart,'sex',404)		#获取属性,不存在时返回404

```





# 属性

## 实例属性和类属性

```python
class Student(object):
    age=7		#类属性
    def __init__(self, name):
        self.name = name

s = Student('Bob')
s.score = 90		#py是动态语言,允许绑定任意属性
print(Student.age)	
print(s.age)		#每个实例皆可访问类属性
s.age=8
print(s.age)		#实例属性覆盖类属性
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
    @property		#定义属性和getter
    def birth(self):
        return self._birth
    @birth.setter	#定义setter
    def birth(self, value):
        if value<19000000
        	raise ValueError('Invalid birthday') #报错
        self._birth = value
    @property
    def age(self):
        return 2015 - self._birth
```



