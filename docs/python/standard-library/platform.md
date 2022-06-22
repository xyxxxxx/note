# platform——获取底层平台的标识数据

## machine()

返回机器类型。

```python
>>> platform.machine()
'x86_64'
```

## node()

返回计算机的网络名称。

```python
>>> platform.node()
'Yuxuans-MacBook-Pro.local'
```

## platform()

返回一个标识底层平台的字符串，其中带有尽可能多的有用信息。

```python
>>> platform.platform()
'macOS-11.2.3-x86_64-i386-64bit'
```

## python_version()

```python
>>> platform.python_version()
'3.8.7'
```

## release()

返回系统的发布版本。

```python
>>> platform.release()
'7'                           # Windows version
>>> platform.release()
'10'                          # Windows version
>>> platform.release()
'20.3.0'                      # Darwin version, refer to https://en.wikipedia.org/wiki/MacOS_Big_Sur
```

## system()

返回系统平台/OS 的名称。

```python
>>> platform.system()
'Windows'                     # Windows
>>> platform.system()
'Darwin'                      # macOS
>>> platform.system()
'Linux'                       # Linux
```

## Mac OS平台

### mac_ver()

获取 Mac OS 版本信息并将其返回为元组 `(release,versioninfo,machine)`，其中 *versioninfo* 是一个元组 `(version,dev_stage,non_release_version)`。

```python
>>> platform.mac_ver()
('11.2.3', ('', '', ''), 'x86_64')    # macOS Big Sur Version 11.2.3
```
