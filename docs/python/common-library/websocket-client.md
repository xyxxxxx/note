# websocket-client (websocket)

websocket-client 是一个 Python 的 WebSocket 客户端，它提供了访问 WebSocket 底层 API 的方法。websocket-client 实现了 WebSocket 协议的 [hybi-13](https://tools.ietf.org/html/draft-ietf-hybi-thewebsocketprotocol-13) 版本，其当前不支持 RFC 7692 中的 permessage-deflate 扩展。

## 示例

### 创建一个 WebSocket

适合短时间的连接。

```python
import websocket

ws = websocket.WebSocket()
ws.connect("ws://echo.websocket.events")
ws.send("Hello, Server")
print(ws.recv())
ws.close()
```

### 创建一个 WebSocketApp

适合长时间的连接。

```python
import websocket

def on_message(wsapp, message):
    print(message)

wsapp = websocket.WebSocketApp("wss://testnet-explorer.binance.org/ws/block", on_message=on_message)
wsapp.run_forever()
```
