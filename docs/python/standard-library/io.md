# io——处理流的核心工具

`io` 模块提供了 Python 用于处理各种 I/O 类别的主要工具。三种主要的 I/O 类别分别为：文本 I/O、二进制 I/O 和原始 I/O。这些是泛型类别，每一种类别可以使用多种后端存储。一个属于这些类别中的任何一个的具体对象被称为文件对象（file object）。其他常用的术语还包括流（stream）和类文件对象（file-like object）。

独立于其类别，每个具体的流对象也具有各种功能：它可以是只读、只写或读写。它还可以允许任意随机访问（向前或向后寻找任何位置），或仅允许顺序访问（例如在套接字或管道的情况下）。

所有流都对于提供给它们的数据类型很敏感。例如将 `str` 对象提供给二进制流的 `write()` 方法会引发 `TypeError`，将 `bytes` 对象提供给文本流的 `write()` 方法也是如此。

**文本 I/O**

文本 I/O 接受和产生 `str` 对象。这意味着，只要后台存储是原生由字节组成（例如在文件的情况下），数据的编码和解码都是透明的，并且可以选择转换特定于平台的换行符。

最简单的创建文本流的方法是调用 `open()`，可以选择指定一种编码：

```python
f = open("myfile.txt", "r", encoding="utf-8")
```

内存中的文本流也可以作为 `StringIO` 对象使用：

```python
f = io.StringIO("some initial text data")
```

[`TextIOBase`](#textiobase) 部分详细描述了文本流的 API。

**二进制 I/O**

二进制 I/O（也称为缓冲 I/O）接受类字节对象，产生 `bytes` 对象。不执行编码、解码或换行转换。这种类别的流可以用于所有类型的非文本数据，并且还可以在需要手动控制文本数据的处理时使用。

最简单的创建二进制流的方法是调用 `open()`，并在模式字符串中指定 `"b"`：

```python
f = open("myfile.jpg", "rb")
```

内存中的二进制流也可以作为 `BytesIO` 对象使用：

```python
f = io.BytesIO(b"some initial binary data: \x00\x01")
```

[`BufferedIOBase`](#bufferediobase) 部分详细描述了二进制流（缓冲流）的 API。

其他库模块可以提供额外的方式来创建文本或二进制流。例如参见 `socket.socket.makefile()`。

**原始 I/O**

原始 I/O（也称为非缓冲 I/O）通常用作二进制和文本流的低级构建块。用户代码直接操作原始流的用法非常罕见。尽管如此，你也可以通过在禁用缓冲的情况下以二进制模式打开文件来创建原始流：

```python
f = open("myfile.jpg", "rb", buffering=0)
```

[RawIOBase](#rawiobase) 部分详细描述了原始流的 API。

## 高阶模块接口

## 类的层次结构

I/O 流被安排为按类的层次结构实现。 首先是 抽象基类 (ABC)，用于指定流的各种类别，然后是提供标准流实现的具体类。

!!! note "注意"
    抽象基类还提供某些方法的默认实现，以帮助实现具体的流类。例如 BufferedIOBase 提供了 readinto() 和 readline() 的未优化实现。

I/O 层次结构的顶部是抽象基类 IOBase 。它定义了流的基本接口。但是请注意，对流的读取和写入之间没有分离。如果实现不支持指定的操作，则会引发 UnsupportedOperation 。

抽象基类 RawIOBase 是 IOBase 的子类。它负责将字节读取和写入流中。 RawIOBase 的子类 FileIO 提供计算机文件系统中文件的接口。

抽象基类 BufferedIOBase 处理原始字节流（ RawIOBase ）上的缓冲。其子类 BufferedWriter 、 BufferedReader 和 BufferedRWPair 缓冲流是可读、可写以及可读写的。 BufferedRandom 为随机访问流提供缓冲接口。 BufferedIOBase 的另一个子类 BytesIO 是内存中字节流。

抽象基类 TextIOBase 是 IOBase 的另一个子类，它处理字节表示文本的流，并处理字符串之间的编码和解码。其一个子类 TextIOWrapper 是原始缓冲流（ BufferedIOBase ）的缓冲文本接口。另一个子类 StringIO 用于文本的内存流。

参数名不是规范的一部分，只有 open() 的参数才用作关键字参数。

### I/O 基类

#### IOBase

#### RawIOBase

#### BufferedIOBase

### 原始文件 I/O

#### FileIO

### 缓冲流

#### BytesIO

#### BufferedReader

#### BufferedWriter

### 文本 I/O

#### TextIOBase

#### TextIOWrapper

#### StringIO
