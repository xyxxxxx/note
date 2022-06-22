# socket——底层网络接口

## 常量

### AF_UNIX

### AF_INET

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

### bind()

### close()

### connect()
