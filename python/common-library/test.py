import websocket

websocket.enableTrace(True)
ws = websocket.WebSocket()
ws.connect("ws://echo.websocket.events/", origin="testing_websockets.com")
ws.send("Hello, Server")
print(ws.recv())
ws.close()
