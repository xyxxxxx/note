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
