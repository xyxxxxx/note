# filelock

支持 `with` 语句的文件锁。

## Timeout

若未能在 `timeout` 秒之内获得，则引发此异常。

## FileLock

## UnixFileLock

在 Unix 系统上使用 `fcntl.flock()` 以硬锁定文件。

## WindowsFileLock
