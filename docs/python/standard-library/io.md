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

内存中文本流也可以作为 `StringIO` 对象使用：

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

内存中二进制流也可以作为 `BytesIO` 对象使用：

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

I/O 流的实现被组织为类的层次结构。首先是抽象基类，用于指定流的各种类别，然后是提供标准流实现的具体类。

!!! note "注意"
    抽象基类还提供某些方法的默认实现，以帮助实现具体类。例如 `BufferedIOBase` 提供了 `readinto()` 和 `readline()` 的未优化实现。

I/O 层次结构的顶端是抽象基类 `IOBase`。它定义了流的基本接口。但是请注意，对于流的读取和写入之间没有分离。如果实现不支持指定的操作，则会引发 `UnsupportedOperation`。

抽象基类 `RawIOBase` 是 `IOBase` 的子类，它处理流的字节读写。`RawIOBase` 的子类 `FileIO` 提供计算机文件系统中文件的接口。

抽象基类 `BufferedIOBase` 处理原始字节流（`RawIOBase`）上的缓冲。其子类 `BufferedWriter`、`BufferedReader` 和 `BufferedRWPair` 缓冲流是可读、可写以及可读写的。`BufferedRandom` 为随机访问流提供缓冲接口。`BufferedIOBase` 的另一个子类 `BytesIO` 是内存中字节流。

抽象基类 `TextIOBase` 是 `IOBase` 的另一个子类，它处理字节表示文本的流，并处理字符串的编码和解码。它的一个子类 `TextIOWrapper` 是原始缓冲流（`BufferedIOBase`）的缓冲文本接口，另一个子类 `StringIO` 是内存中文本流。

参数名不是规范的一部分，并且只有 `open()` 的参数才被用作关键字参数。

| 抽象基类         | 继承     | 抽象方法                                | Mixin 方法和属性                                                                                                                                                           |
| ---------------- | -------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `IOBase`         |          | `fileno`、`seek` 和 `truncate`          | `close`、`closed`、`__enter__`、`__exit__`、`flush`、`isatty`、`__iter__`、`__next__`、`readable`、`readline`、`readlines`、`seekable`、`tell`、`writable` 和 `writelines` |
| `RawIOBase`      | `IOBase` | `readinto` 和 `write`                   | 继承 `IOBase` 方法、`read` 和 `readall`                                                                                                                                    |
| `BufferedIOBase` | `IOBase` | `detach`、`read`、`read1` 和 `write`    | 继承 `IOBase` 方法、`readinto` 和 `readinto1`                                                                                                                              |
| `TextIOBase`     | `IOBase` | `detach`、`read`、`readline` 和 `write` | 继承 `IOBase` 方法、`encoding`、`errors` 和 `newlines`                                                                                                                     |

### I/O 基类

#### IOBase

所有 I/O 类的抽象基类，作用于字节流。没有公共构造函数。

此类为许多方法提供了空的抽象实现，派生类可以选择性地重载。默认实现代表一个无法读取、写入或查找（seek）的文件。

尽管 `IOBase` 没有声明 `read()` 或 `write()`，因为它们的签名会有所不同，但是实现和客户端应该将这些方法视为接口的一部分。此外，当调用实现不支持的操作时，可以引发一个 `ValueError`（或 `UnsupportedOperation`）。

从文件读取或向文件写入的二进制数据的基本类型为 `bytes`。其他类字节对象也可以作为方法参数。文本 I/O 类使用 `str` 数据。

请注意，在关闭的流上调用任何方法（甚至查询）都是未定义的。在这种情况下，实现可以引发 `ValueError`。

`IOBase`（及其子类）支持迭代器协议，这意味着可以迭代一个 `IOBase` 对象以产生流中的行。取决于流是二进制流（产生字节）还是文本流（产生字符串），行的定义略有不同。请参见后文的 `readline()`。

`IOBase` 也是一个上下文管理器，因此支持 `with` 语句。在下面这个示例中，*file* 将在 `with` 语句块执行完成之后被关闭——即使发生了异常：

```python
with open('spam.txt', 'w') as file:
    file.write('Spam and eggs!')
```

`IOBase` 提供以下数据属性和方法：

##### close()

清空（flush）并关闭此流。如果文件已经关闭，则此方法无效。文件关闭后，对于文件的任何操作（例如读取或写入）都会引发 `ValueError`。

为方便起见，允许多次调用此方法，但只有第一次调用有效。

##### closed

如果流已关闭，则返回 True。

##### fileno()

返回流的底层文件描述符（一个整数）（如果流存在）。如果 IO 对象不使用文件描述符，则会引发一个 `OSError`。

##### flush()

清空流的写入缓冲区（如果适用）。这对只读和非阻塞流不起作用。

##### isatty()

如果流是交互式的（即连接到终端/tty设备），则返回 True 。

##### readable()

如果可以读取流，则返回 True 。否则为 False ，且 read() 将引发 OSError 错误。

##### readline()

```python
readline(size=-1)
```

从流中读取并返回一行。如果指定了 *size*，将最多读取 *size* 个字节。

对于二进制文件，行结束符总是 `b'\n'`；对于文本文件，可以通过向 `open()` 函数传入 *newline* 参数来指定要识别的行结束符。

##### readlines()

```python
readlines(hint=-1)
```

从流中读取并返回包含多行的列表。可以指定 *hint* 来控制要读取的行数：如果已经读取的所有行的（以字节/字符数表示的）总大小超出了 *hint*，则将不会继续读取下一行。

请注意可以使用 `for line in file: ...` 直接对文件对象进行迭代，而不必调用 `file.readlines()`。

##### seek()

```python
seek(offset, whence=SEEK_SET)
```

将流的位置修改到给定字节的 *offset*（偏移）。*offset* 相对于由 *whence* 指定的位置进行解析。*whence* 的值有：

* `SEEK_SET` 或 `0`：流的开头（默认值）；*offset* 应为零或正值
* `SEEK_CUR` 或 `1`：当前流位置；*offset* 可以为负值
* `SEEK_END` 或 `2`：流的末尾；*offset* 通常为负值

返回新的绝对位置。

##### seekable()

如果流支持随机访问则返回 True。若为 False，则 `seek()`、`tell()` 和 `truncate()` 将引发 `OSError`。

##### tell()

返回流的当前位置。

##### truncate()

```python
truncate(size=None)
```

将流的大小调整为给定的 *size* 个字节（如果未指定 *size* 则调整至当前位置）。流的当前位置不变。这个调整操作可扩展或减小当前文件大小。在扩展的情况下，新文件区域的内容取决于具体平台（在大多数系统上，额外的字节会填充为零）。返回新的文件大小。

##### writable()

如果流支持写入则返回 True。若为 False，则 `write()` 和 `truncate()` 将引发 `OSError`。

##### writelines()

将包含多行的列表写入到流。不会添加行分隔符，因此通常提供的每一行的末尾都带有行分隔符。

##### \__del__()

为对象销毁进行准备。`IOBase` 提供了此方法的默认实现，该实现会调用实例的 `close()` 方法。

#### RawIOBase

原始二进制 I/O 的基类。它继承自 `IOBase`。没有公共构造函数。

原始二进制 I/O 通常提供对底层 OS 设备或 API 的低层级访问，而不会尝试将其封装到高级的原语中（这是留给缓冲 I/O 和文本 I/O 的，将在后文中描述）。

除了 `IOBase` 的属性和方法之外，`RawIOBase` 还提供了下列方法:

##### read()

```python
read(size=-1)
```

从对象读取最多 *size* 个字节并将其返回。为方便起见，如果 *size* 未指定或为 -1，则返回所有字节直到 EOF。在其他情况下，只会执行一次系统调用。如果操作系统调用返回少于 size 个字节则此方法也可能返回少于 size 个字节。

如果返回 0 个字节而 *size* 不为零 0，这表明到达文件末尾。如果对象处于非阻塞模式并且没有更多字节可用，则返回 None。

默认实现会转向 `readall()` 和 `readinto()`。

##### readall()

从流中读取并返回所有字节直到 EOF，如有必要将对于流执行多次调用。

##### readinto()

```python
readinto(b)
```

将字节数据读入预先分配的可写的类字节对象 *b*，并返回读取的字节数。例如，*b* 可以是一个 `bytearray`。如果对象处于非阻塞模式并且没有更多字节可用，则返回 None。

##### write()

```python
write(b)
```

将给定的类字节对象 *b* 写入到底层的原始流，并返回所写入的字节数。这可以少于 *b* 的字节长度，具体取决于底层原始流的设定，特别是当它处于非阻塞模式。如果原始流设为非阻塞并且不能实际向其写入单个字节则返回 None。

调用函数可以在此方法返回后释放或改变 *b*，因此实现应仅在方法调用期间访问 *b*。

#### BufferedIOBase

支持某种缓冲的二进制流的基类。它继承自 `IOBase`。没有公共构造函数。

与 `RawIOBase` 的主要区别在于 `read()`、`readinto()` 和 `write()` 方法将（分别）尝试按照请求读取足量的输入或是耗尽所有给定的输出，其代价是可能会执行多于一次的系统调用。

除此之外，这些方法还可能引发 `BlockingIOError`，如果底层的原始流处于非阻塞模式并且无法接受或给出足够的数据；不同于 `RawIOBase` 的对应方法，它们永远不会返回 None。

并且，`read()` 方法也没有转向 `readinto()` 的默认实现。

典型的 `BufferedIOBase` 实现不应继承自 `RawIOBase` 实现，而应包装一个实现，就像 `BufferedWriter` 和 `BufferedReader` 一样。

除了 `IOBase` 的属性和方法之外，`BufferedIOBase` 还提供了下列方法和属性：

##### raw

`BufferedIOBase` 处理的底层原始流（一个 `RawIOBase` 实例）。它不是 `BufferedIOBase` API 的组成部分并且不存在于某些实现中。

##### detach()

从缓冲区分离出底层原始流并将其返回。

在原始流被分离之后，缓冲区将处于不可用的状态。

某些缓冲区（例如 `BytesIO`）的实现并没有原始流的概念。它们调用此方法将引发 `UnsupportedOperation`。

##### read()

```python
read(size=-1)
```

读取并返回最多 *size* 个字节。如果此参数被省略、为 None 或为负值，则读取并返回所有数据直到 EOF。如果流已经到达 EOF 则返回一个空的 `bytes` 对象。

如果此参数为正值，并且底层原始流不可交互，则可能发起多个原始读取以满足字节计数（除非先遇到 EOF）。但对于可交互原始流，则将最多发起一个原始读取，并且短的结果并不意味着即将到达 EOF。

`BlockingIOError` 会在底层原始流不处于阻塞模式，并且当前没有可用数据时被引发。

##### read1()

```python
read1([size])
```

通过最多一次对底层原始流的 `read()`（或 `readinto()`）方法的调用读取并返回最多 *size* 个字节。这在你在 `BufferedIOBase` 对象之上实现你自己的缓冲区的情况下很有用。

如果 *size* 为 -1（默认值），则返回任意数量的字节（多于零个字节，除非已到达 EOF）。

##### readinto()

```python
readinto(b)
```

将字节数据读入预先分配的可写的类字节对象 *b* 并返回读取的字节数。例如，*b* 可以是一个 `bytearray`。

类似于 `read()`，可能对底层原始流发起多次读取，除非底层原始流为交互式。

`BlockingIOError` 会在底层原始流不处于阻塞模式，并且当前没有可用数据时被引发。

##### readinto1()

```python
readinto1(b)
```

将字节数据读入预先分配的可写的类字节对象 *b* 并返回读取的字节数，最多使用一次对底层原始流 `read()`(或 `readinto()`) 方法的调用。

`BlockingIOError` 会在底层原始流不处于阻塞模式，并且当前没有可用数据时被引发。

##### write()

```python
write(b)
```

写入类字节对象 *b*，并返回写入的字节数（总是等于 *b* 的字节长度，因为如果写入失败会引发 `OSError`）。根据具体实现的不同，这些字节可能被实际写入底层流，或是出于运行效率和冗余等考虑而暂存于缓冲区。

当处于非阻塞模式时，如果需要将数据写入原始流但它无法在不阻塞的情况下接受所有数据则将引发 `BlockingIOError`。

调用函数可以在此方法返回后释放或改变 *b*，因此实现应仅在方法调用期间访问 *b*。

### 原始文件 I/O

#### FileIO

```python
class io.FileIO(name, mode='r', closefd=True, opener=None)
```

`FileIO` 代表操作系统层级的包含字节数据的文件。它实现了 `RawIOBase` 接口（因而也实现了 `IOBase` 接口）。

*name* 可以是以下两项之一：

* 代表将被打开的文件的路径的字符串或 `bytes` 对象。在这种情况下 *closefd* 必须为 True（默认值），否则将会引发异常。
* 代表一个现有的操作系统层级的文件描述符的整数，作为结果的 `FileIO` 对象将可以访问该文件。当 `FileIO` 对象被关闭时此文件描述符也将被关闭，除非 *closefd* 设为 False。

*mode* 可以是 `'r'`、`'w'`、`'x'` 或 `'a'`，分别表示读取（默认模式）、写入、（独占）新建或添加。如果以写入或添加模式打开的文件不存在则将自动新建；当以写入模式打开时文件将先清空。以新建模式打开时如果文件已存在则将引发 `FileExistsError`。以新建模式打开文件也意味着要写入，因此该模式的行为与 `'w'` 类似。在模式中附带 `'+'` 将允许同时读取和写入。

此类的 `read()`（当附带正值参数调用时）、`readinto()` 和 `write()` 方法将只执行一次系统调用。

可以通过传入一个可调用对象作为 *opener* 来使用自定义文件打开器。然后通过调用 *opener* 并传入 *(name, flags)* 来获取文件对象的底层文件描述符。*opener* 必须返回一个打开的文件描述符（传入 `os.open` 作为 *opener* 在功能上将与传入 None 类似）。

新创建的文件是不可继承的。

有关 *opener* 参数的示例，请参见内置函数 `open()`。

##### mode

构造函数中给定的模式。

##### name

文件名。当构造函数中没有给定名称时，这是文件的文件描述符。

### 缓冲流

相比原始 I/O，缓冲 I/O 流提供了针对 I/O 设备的更高层级的接口。

#### BytesIO

```python
class io.BytesIO([initial_bytes])
```

一个使用内存中字节缓冲区的流实现。它继承自 `BufferedIOBase`。当调用 `close()` 方法时将丢弃缓冲区。

可选参数 *initial_bytes* 是一个包含初始数据的类字节对象。

除了 `BufferedIOBase` 和 `IOBase` 的属性和方法之外，`BytesIO` 还提供或重载了下列方法:

##### getbuffer()

返回一个对应于缓冲区内容的可读写视图，而不必拷贝其数据。此外，改变视图将透明地更新缓冲区内容:

```python
>>> b = io.BytesIO(b"abcdef")
>>> view = b.getbuffer()
>>> view[2:4] = b"56"
>>> b.getvalue()
b'ab56ef'
```

!!! note "注意"
    只要视图保持存在，`BytesIO` 对象就无法被改变大小或关闭。

##### getvalue()

返回包含缓冲区全部内容的 `bytes`。

##### read1()

在 `BytesIO` 中，这与 `read()` 相同。

##### readinto1()

在 `BytesIO` 中，这与 `readinto()` 相同。

#### BufferedReader

一个对可读的序列型 `RawIOBase` 对象提供更高层级访问的缓冲区。它继承自 `BufferedIOBase`。当从此对象读取数据时，可能会从底层原始流请求更大量的数据，并存放到内部缓冲区中。接下来可以在后续读取时直接返回缓冲数据。

构造函数根据给定的可读的原始流 *raw* 和 *buffer_size* 创建一个 `BufferedReader`。如果省略 *buffer_size* 则使用默认值 `DEFAULT_BUFFER_SIZE`。

除了 `BufferedIOBase` 和 `IOBase` 的属性和方法之外，`BufferedReader` 还提供或重载了下列方法:

##### peek()

```python
peek([size])
```

从流返回字节数据而不前移位置。完成此调用将最多读取一次原始流。返回的字节数量可能少于或多于请求的数量。

##### read()

```python
read([size])
```

读取并返回 *size* 个字节。如果 *size* 未给定或为负值，则读取至 EOF，或直到读取调用在非阻塞模式下阻塞。

##### read1()

```python
read1([size])
```

在原始流上通过单次调用读取并返回最多 *size* 个字节。如果至少缓冲了一个字节，则只返回缓冲的字节；否则执行一次原始流读取调用。

#### BufferedWriter

```python
class io.BufferedWriter(raw, buffer_size=DEFAULT_BUFFER_SIZE)¶
```

一个对可写的序列型 `RawIOBase` 对象提供更高层级访问的缓冲区。它继承自 `BufferedIOBase`。当写入到此对象时，数据通常会被放入内部缓冲区。缓冲区将在某些条件下被写入到底层的 `RawIOBase` 对象，包括:

* 当缓冲区相对于所有挂起的数据太小时
* 当 `flush()` 被调用时
* 当 `seek()`（为 `BufferedRandom` 对象）被请求时
* 当 `BufferedWriter` 对象被关闭或销毁时

构造函数根据给定的可写的原始流 *raw* 和 *buffer_size* 创建一个 `BufferedWriter`。如果省略 *buffer_size* 则使用默认值 `DEFAULT_BUFFER_SIZE`。

除了 `BufferedIOBase` 和 `IOBase` 的属性和方法之外，`BufferedWriter` 还提供或重载了下列方法:

##### flush()

将缓冲区中保存的字节数据强制写入原始流。如果原始流发生阻塞则应引发 `BlockingIOError`。

##### write()

```python
write(b)
```

写入类字节对象 *b*，并返回写入的字节数。当处于非阻塞模式时，如果需要写入缓冲区但原始流发生阻塞则将引发 `BlockingIOError`。

#### BufferedRandom

```python
class io.BufferedRandom(raw, buffer_size=DEFAULT_BUFFER_SIZE)
```

随机访问流的带缓冲的接口。它继承自 `BufferedReader` 和 `BufferedWriter`。

构造函数会为在第一个参数中给定的可查找的原始流创建一个 reader 和 writer。如果省略 *buffer_size* 则使用默认值 `DEFAULT_BUFFER_SIZE`。

`BufferedRandom` 能够做到 `BufferedReader` 或 `BufferedWriter` 所能够做到的任何事。此外，还会确保实现 `seek()` 和 `tell()`。

### 文本 I/O

#### TextIOBase

文本流的基类。此类提供了基于字符和行的流 I/O 的接口。它继承自 `IOBase`。没有公共构造函数。

除了 `IOBase` 的属性和方法之外，`TextIOBase` 还提供或重载了下列方法和属性：

##### encoding

用于将流的字节串解码为字符串以及将字符串编码为字节串的编码名称。

##### errors

解码器或编码器的错误设置。

##### newlines

一个字符串、字符串元组或 None，指示到目前为止已经转写的新行。根据具体实现和初始构造函数旗标的不同，此属性或许不可用。

##### buffer

`TextIOBase` 处理的底层二进制缓冲区（一个 `BufferedIOBase` 实例）。它不是 `TextIOBase` API 的组成部分并且不存在于某些实现中。

##### detach()

从 `TextIOBase` 分离出底层二进制缓冲区并将其返回。

在底层缓冲区被分离之后，`TextIOBase` 将处于不可用的状态。

某些 `TextIOBase`（例如 `StringIO`）的实现并没有底层缓冲区的概念，它们调用此方法将引发 `UnsupportedOperation`。

##### read()

```python
read(size=-1)
```

从流中读取最多 *size* 个字符并以单个字符串的形式返回。如果 *size* 为负值或 None，则读取至 EOF。

##### readline()

```python
readline(size=-1)
```

读取至换行符或 EOF 并返回单个字符串。如果流已经到达 EOF，则将返回一个空字符串。

如果指定了 *size*，将最多读取 *size* 个字符。

##### seek()

```python
seek(offset, whence=SEEK_SET)
```

将流的位置更改为给定的偏移位置 *offset*。具体行为取决于 *whence* 参数：

* `SEEK_SET` 或 `0`：从流的开头查找；*offset* 必须为 `TextIOBase.tell()` 返回的数值或为零。任何其他 *offset* 值都将导致未定义的行为。
* `SEEK_CUR` 或 `1`：查找到当前位置；*offset* 必须为零，表示无操作（所有其他值均不受支持）。
* `SEEK_END` 或 `2`：查找到流的末尾；*offset* 必须为零（所有其他值均不受支持）。

以不透明数字的形式返回新的绝对位置。

##### tell()

以不透明数字的形式返回流的当前位置。该数字并不通常代表底层二进制存储中的字节数。

##### write()

```python
write(s)
```

将字符串 *s* 写入到流并返回写入的字符数。

#### TextIOWrapper

```python
io.TextIOWrapper(buffer, encoding=None, errors=None, newline=None, line_buffering=False, write_through=False)
```

一个基于 `BufferedIOBase` 二进制流的缓冲文本流。它继承自 `TextIOBase`。

*encoding* 给出用于流编解码的编码名称。它默认为 `locale.getpreferredencoding(False)`。

*errors* 是一个可选的字符串，其指明编解码错误的处理方式，具有下列值：

* `'strict'`：在出现编码错误时引发 `ValueError`（默认值 None 具有相同的效果）。
* `'ignore'`：忽略错误（注意忽略编码错误可能导致数据丢失）。
* `'replace'`：在出现格式错误的数据时插入一个替换标记（例如 `'?'`）。
* `'backslashreplace'`：将格式错误的数据替换为一个反斜杠转义序列。

在写入时，还可以使用 `'xmlcharrefreplace'`（替换为适当的 XML 字符引用）或 `'namereplace'`（替换为 `\N{...}` 转义序列）。任何其他通过 `codecs.register_error()` 注册的错误处理名称也是有效的。

*newline* 控制行结束符的处理方式。它可以是 None、`''`、`'\n'`、`'\r'` 和 `'\r\n'`。其工作原理如下:

* 当从流中读取输入时，如果 *newline* 为 None，则会启用[通用换行](https://docs.python.org/zh-cn/3.8/glossary.html#term-universal-newlines)模式。输入中的行结束符可以是 `'\n'`、`'\r'` 或 `'\r\n'`，在被返回给调用函数之前它们会被统一转写为 `'\n'`。如果 *newline* 为 `''`，也会启用通用换行模式，但行结束符会不被转写地返回给调用函数。如果 *newline* 是任何其他合法值，则输入行将仅由给定的字符串结束，并且行结束符会不被转写地返回给调用者。
* 将输出写入流时，如果 *newline* 为 None，则写入的任何 `'\n'` 字符都被转写为系统默认行分隔符 `os.linesep`。如果 *newline* 是 `''` 或 `'\n'`，则不进行转写。如果 *newline* 是任何其他合法值，则写入的任何 `'\n'` 字符将被转写为给定的字符串。

如果 *line_buffering* 为 True，则当一个写入调用包含换行符或回车时将会应用 `flush()`。

如果 *write_through* 为 True，对 `write()` 的调用会确保不被缓冲：对 `TextIOWrapper` 对象写入的任何数据会立即交给其下层的 *buffer* 来处理。

#### StringIO

用于文本 I/O 的内存中流。当调用 `close()` 方法时将会丢弃文本缓冲区。

缓冲区的初始值可以通过提供 *initial_value* 参数来设置。如果启用了行结束符转写，换行会以与 `write()` 相同的方式被编码。流的位置被设在缓冲区的开头。

*newline* 参数的规则与 `TextIOWrapper` 的一致。默认规则是仅将 `\n` 字符视为行结束符并且不执行换行符转写。如果 *newline* 设为 None，在所有平台上换行符都被写入为 `\n`，但当读取时仍然会执行通用换行编码。

```python
import io

output = io.StringIO()
output.write('First line.\n')
print('Second line.', file=output)

# Retrieve file contents -- this will be
# 'First line.\nSecond line.\n'
contents = output.getvalue()

# Close object and discard memory buffer --
# .getvalue() will now raise an exception.
output.close()
```

除了 `TextIOBase` 及其父类的属性和方法之外，`StringIO` 还提供或重载了下列方法:

##### getvalue()

返回一个包含缓冲区全部内容的字符串。换行会以与 `read()` 相同的方式被编码，但是流的位置不会被改变。

## 性能

### 二进制 I/O

通过仅读取和写入大块数据的方法（即使用户请求单个字节），缓冲 I/O 隐藏了在调用和执行操作系统无缓冲 I/O 例程过程中的任何低效性。增益取决于操作系统以及执行的 I/O 类型。例如，在某些现代操作系统上（例如 Linux），无缓冲磁盘 I/O 可以与缓冲 I/O 一样快。但最重要的是，无论平台和支持设备如何，缓冲 I/O 都能提供可预测的性能。因此，对于二进制数据，应首选缓冲 I/O 而不是未缓冲的 I/O 。

### 文本 I/O

二进制存储（如文件）上的文本 I/O 比同一存储上的二进制 I/O 慢得多，因为它需要使用字符编解码器在 Unicode 和二进制数据之间进行转换。这在处理巨量文本数据（如大型日志文件）时会变得非常明显。此外，由于使用了重构算法，`TextIOWrapper.tell()` 和 `TextIOWrapper.seek()` 都相当慢。

尽管如此，`StringIO` 是原生的内存中 Unicode 容器，其速度与 `BytesIO` 接近。

### 多线程

`FileIO` 对象是线程安全的，只要它们封装的操作系统调用（比如 Unix 下的 `read(2)`）也是线程安全的。

二进制缓冲对象（`BufferedReader`、`BufferedWriter`、`BufferedRandom` 和 `BufferedRWPair`）使用锁来保护其内部结构；因此，可以安全地一次从多个线程中调用它们。

`TextIOWrapper` 对象不是线程安全的。

### 可重入性

二进制缓冲对象（`BufferedReader`、`BufferedWriter`、`BufferedRandom` 和 `BufferedRWPair`）不是可重入的。虽然在正常情况下不会发生可重入调用，但仍可能会在 `signal` 处理程序执行 I/O 时产生。如果线程尝试重入已经访问的缓冲对象，则会引发 `RuntimeError`。注意这并不禁止其他线程进入缓冲对象。

上面的内容隐含地扩展到文本文件，因为 `open()` 函数会将缓冲对象封装到 `TextIOWrapper` 中。这包括标准流，因此也会影响内置函数 `print()`。
