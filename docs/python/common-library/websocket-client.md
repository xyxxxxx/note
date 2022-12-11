# websocket-client (websocket)

websocket-client 是一个 Python 的 WebSocket 客户端，它提供了访问 WebSocket 底层 API 的方法。websocket-client 实现了 WebSocket 协议的 [hybi-13](https://tools.ietf.org/html/draft-ietf-hybi-thewebsocketprotocol-13) 版本，其当前不支持 RFC 7692 中的 permessage-deflate 扩展。

## 示例

### 创建你的第一个 WebSocket 连接

适合短时间的连接。

```python
import websocket

ws = websocket.WebSocket()
ws.connect("ws://echo.websocket.events")
ws.send("Hello, Server")
print(ws.recv())
ws.close()
```

适合长时间的连接。

```python
import websocket

def on_message(wsapp, message):
    print(message)

wsapp = websocket.WebSocketApp("wss://testnet-explorer.binance.org/ws/block", on_message=on_message)
wsapp.run_forever()
```

### 调试选项

使用 `websocket.enableTrace(True)`。

```python
import websocket

websocket.enableTrace(True)
ws = websocket.WebSocket()
ws.connect("ws://echo.websocket.events/", origin="testing_websockets.com")
ws.send("Hello, Server")
print(ws.recv())
ws.close()
```

### 连接选项

使用特定的选项来自定义你的连接，包括：

* Host 头
* Cookie 头
* Origin 头
* 自定义头
* WebSocket 子协议
* SSL 或主机名验证
* 超时时间

## API

### create_connection()

### WebSocket

低级 WebSocket 接口。

此类基于 WebSocket 协议 [draft-hixie-thewebsocketprotocol-76](http://tools.ietf.org/html/draft-hixie-thewebsocketprotocol-76)。

我们可以连接到 WebSocket 服务器并发送、接收数据。下面的示例是一个回声客户端：

```python
>>> import websocket
>>> ws = websocket.WebSocket()
>>> ws.connect("ws://echo.websocket.events")
>>> ws.recv()
'echo.websocket.events sponsored by Lob.com'
>>> ws.send("Hello, Server")
19
>>> ws.recv()
'Hello, Server'
>>> ws.close()
```

```python
classwebsocket._core.WebSocket(get_mask_key=None, sockopt=None, sslopt=None, fire_cont_frame=False, enable_multithread=True, skip_utf8_validation=False, **_)[source]
# get_mask_key
# sockopt
# sslopt                作为SSL socket选项的可选字典对象
# fire_cont_frame
# enable_multithread
# skip_utf8_validation  跳过UTF8验证
```

#### close()

关闭 WebSocket 对象。

```python
close(status=1000, reason=b'', timeout=3)
# status  发送的状态码
# reason  关闭的原因,使用UTF-8编码
# timeout 接收关闭帧的超时时间.若为None,则将永远等待直到接收一个关闭帧
```

#### connect()

连接到 URL。URL 为 WebSocket URL 格式，即 `ws://host:port/resource`。你可以使用多个选项自定义连接。

```python
connect(url, **options)
# header      作为自定义 HTTP 头的列表或字典
# cookie      Cookie值
# origin      自定义原URL
# connection  自定义连接头值,默认值"Upgrade"设定在`_handshake.py`文件中
# timeout     Socket超时时间
# ...
```

#### getheaders()

获取握手响应头。

#### getstatus()

获取握手状态。

#### getsubprotocol()

获取子协议。

#### headers

获取握手响应头。

#### ping()

发送 ping 数据。

#### pong()

发送 pong 数据。

#### recv()

从服务器接收字符串数据（字节数组）。

#### recv_frame()

以帧的形式从服务器接收数据。

#### send()


