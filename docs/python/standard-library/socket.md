# socket——底层网络接口

## 常量

### AF_UNIX, AF_INET, AF_INET6

这些常量表示地址（和协议）簇，用于 `socket()` 的第一个参数。如果 `AF_UNIX` 常量未定义，即表示不支持该协议。不同系统可能会有更多其他常量可用。

# SOCK_STREAM, SOCK_DGRAM, SOCK_RAW, SOCK_RDM, SOCK_SEQPACKET

这些常量表示套接字类型，用于 `socket()` 的第二个参数。不同系统可能会有更多其他常量可用（一般只有 `SOCK_STREAM` 和 `SOCK_DGRAM` 可用）。

## 函数

### 创建套接字

#### socket()

```python
socket.socket(family=AF_INET, type=SOCK_STREAM, proto=0, fileno=None)
```

使用给定的地址簇、套接字类型和协议号创建一个新的套接字。地址簇应为 `AF_INET`（默认）、`AF_INET6`、`AF_UNIX`、`AF_CAN`、`AF_PACKET` 或 `AF_RDS` 其中之一。套接字类型应为 `SOCK_STREAM`（默认）、`SOCK_DGRAM`、`SOCK_RAW` 或其他 `SOCK_` 常量之一。协议号通常为零，可以省略，或者在地址簇为 `AF_CAN` 的情况下，协议号应为 `CAN_RAW`、`CAN_BCM` 或 `CAN_ISOTP` 之一。

#### socketpair()

构建一对已连接的套接字对象，使用给定的地址簇、套接字类型和协议号。地址簇、套接字类型和协议号与上述 `socket()` 函数相同。默认地址簇为 `AF_UNIX`（需要当前平台支持，不支持则默认为 `AF_INET`）。

#### create_connection()

```python
socket.create_connection(address[, timeout[, source_address]])
```

连接到一个 TCP 服务，该服务正在侦听 Internet *address* （用二元组 `(host, port)` 表示）。连接后返回套接字对象。这是比 `socket.connect()` 更高级的函数：如果 *host* 是非数字主机名，它将尝试从 `AF_INET` 和 `AF_INET6` 解析它，然后依次尝试连接到所有可能的地址，直到连接成功。这使得编写兼容 IPv4 和 IPv6 的客户端变得容易。

传入可选参数 *timeout* 可以在套接字实例上设置超时（在尝试连接前）。如果未提供 *timeout*，则使用由 getdefaulttimeout() 返回的全局默认超时设置。

如果提供了 *source_address*，它必须是二元组 `(host, port)`，以便套接字在连接之前绑定为其源地址。如果 host 或 port 分别为 '' 或 0，则使用操作系统默认行为。

#### create_server()

```python
socket.create_server(address, *, family=AF_INET, backlog=None, reuse_port=False, dualstack_ipv6=False)
```

便捷函数，创建绑定到 *address*（二元组 `(host, port)`）的 TCP 套接字，返回套接字对象。

*family* 应设置为 `AF_INET` 或 `AF_INET6`。*backlog* 是传递给 `socket.listen()` 的队列大小，当它为 0 则表示默认的合理值。*reuse_port* 表示是否设置 `SO_REUSEPORT` 套接字选项。

### 其他功能

#### getaddrinfo()


#### gethostbyname()

将主机名转换为 IPv4 地址格式。IPv4 地址以字符串格式返回，如 `'100.50.200.5'`。如果主机名本身是 IPv4 地址，则原样返回。更完整的接口请参考 `gethostbyname_ex()`。`gethostbyname()` 不支持 IPv6 名称解析，应使用 `getaddrinfo()` 来支持 IPv4/v6 双协议栈。

#### gethostname()

返回一个字符串，包含当前正在运行 Python 解释器的机器的主机名。

## 套接字对象

### accept()

接受一个连接。此套接字必须绑定到一个地址上并且监听连接。返回值是一个 `(conn, address)` 对，其中 `conn` 是一个新的套接字对象，用于在此连接上收发数据，`address` 是连接另一端的套接字所绑定的地址。

### bind()

```python
socket.bind(address)
```

将套接字绑定到 *address*。套接字必须尚未绑定。

### close()

将套接字标记为关闭。当 `makefile()` 创建的所有文件对象都关闭时，底层系统资源（如文件描述符）也将关闭。一旦上述情况发生，之后对套接字对象的所有操作都会失败，对端将接收不到任何数据（清空队列数据后）。

垃圾回收时，套接字会自动关闭，但建议显式地 `close()` 它们，或在它们周围使用 `with` 语句。

### connect()

```python
socket.connect(address)
```

连接到 *address* 处的远程套接字。

如果连接被信号中断，则本方法将等待，直到连接完成。如果信号处理程序未抛出异常，且套接字阻塞中或已超时，则在超时后抛出 `socket.timeout`。对于非阻塞套接字，如果连接被信号中断，则本方法将抛出 `InterruptedError` 异常（或信号处理程序抛出的异常）。

### family

套接字的协议簇。

### fileno()

返回套接字的文件描述符（一个小整数），失败时返回 `-1`。经常配合 `select.select()` 使用。

### getpeername()

返回套接字连接到的远程地址。举例而言，这可以用于查找远程 IPv4/v6 套接字的端口号。部分系统不支持此函数。

### getsockname()

返回套接字本身的地址。举例而言，这可以用于查找 IPv4/v6 套接字的端口号。

### recv()

```python
socket.recv(bufsize[, flags])
```

从套接字接收数据。返回一个字节对象，表示接收到的数据。*bufsize* 指定一次接收的最大数据量。可选参数 *flags* 的含义请参阅 Unix 手册页 recv(2)，它默认为 0。

### recv_into()

```python
socket.recv_into(buffer[, nbytes[, flags]])
```

从套接字接收至多 *nbytes* 个字节，将其写入缓冲区而不是创建新的字节串。如果 *nbytes* 未指定（或指定为 0），则接收至所给缓冲区的最大可用大小。返回接收到的字节数。可选参数 *flags* 的含义请参阅 Unix 手册页 recv(2)，它默认为 0。

### send()

```python
socket.send(bytes[, flags])
```

发送数据给套接字。本套接字必须已连接到远程套接字。返回已发送的字节数。可选参数 *flags* 的含义请参阅 Unix 手册页 recv(2)，它默认为 0。

应用程序要负责检查所有数据是否已发送，如果仅传输了部分数据，程序需要自行尝试传输其余数据。

### sendall()

```python
socket.sendall(bytes[, flags])
```

发送数据给套接字。本套接字必须已连接到远程套接字。与 `send()` 不同，本方法持续从 *bytes* 发送数据，直到所有数据都已发送或发生错误为止。成功后会返回 `None`；出错后会引发一个异常，此时并没有办法确定成功发送了多少数据。可选参数 flags 的含义与上述 recv() 中的相同。

### setblocking()

设置套接字为阻塞或非阻塞模式。本函数可以视作 `settimeout()` 函数的简写：

* `sock.setblocking(True)` 相当于 `sock.settimeout(None)`
* `sock.setblocking(False)` 相当于 `sock.settimeout(0.0)`

### settimeout()

```python
socket.settimeout(value)
```

为阻塞套接字的操作设置超时。*value* 参数可以是非负浮点数，以秒为单位，也可以是 None。如果赋为一个非零值，那么如果在操作完成前超过了超时时间 *value*，后续的套接字操作将抛出 `timeout` 异常。如果赋为 0，则套接字将处于非阻塞模式。如果指定为 None，则套接字将处于阻塞模式。

### proto

套接字的协议。

### type

套接字的类型。

## 关于套接字超时的说明

一个套接字对象可以处于以下三种模式之一：阻塞、非阻塞或超时。套接字默认以阻塞模式创建，但是可以调用 `setdefaulttimeout()` 来更改。

* 在阻塞模式中，操作将阻塞，直到操作完成或系统返回错误（如连接超时）。
* 在非阻塞模式中，如果操作无法立即完成，则操作将失败（不幸的是，不同系统返回的错误不同）；`select` 模块中的函数可用于了解套接字何时以及是否可以读取或写入。
* 在超时模式下，如果无法在指定的超时内完成操作（抛出 `timeout` 异常），或如果系统返回错误，则操作将失败。
