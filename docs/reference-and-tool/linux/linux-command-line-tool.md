# Linux 命令行工具

## curl

curl 是一个常用的命令行工具，用于数据传输，支持多种协议，功能十分强大。

```shell
$ curl www.example.com         # 自动选择协议,默认为`GET`方法
<!doctype html>
<html>
<head>
    <title>Example Domain</title>    
...
</body>
</html>
```

### -b

向服务器发送Cookie。

```shell
$ curl -b 'foo=bar' https://www.google.com                # 生成请求头`Cookie: foo=bar`
$ curl -b 'foo1=bar1; foo2=bar2' https://www.google.com   # 发送多个Cookie
$ curl -b cookies.txt https://www.google.com              # 读取本地Cookie文件并发送
```

### -c

将服务器返回的Cookie写入文件。

```shell
$ curl -c cookies.txt https://www.google.com          # 将响应头的Cookie设置写入`cookie.txt`文件
```

### -d

发送 `POST` 请求的数据体。

```shell
# 请求自动转为POST方法,请求头设置 `Content-Type : application/x-www-form-urlencoded`
$ curl -d 'login=emma＆password=123' [-X POST] https://www.google.com/login
$ curl -d 'login=emma' -d 'password=123' [-X POST] https://www.google.com/login   # 同上
$ curl -d '@data.txt' https://www.google.com/login        # 读取本地`data.txt`文件并作为数据体发送
```

### -F

向服务器上传二进制文件。

```shell
# 请求头设置 `Content-Type: multipart/form-data`
$ curl -F 'file=@photo.png' https://www.google.com/profile                   # 上传`photo.png`文件
$ curl -F 'file=@photo.png;filename=me.png' https://www.google.com/profile   # 指定服务器接收到的文件名 
```

### -G

构造URL的查询字符串。

```shell
$ curl -G -d 'q=kitties' -d 'count=20' https://www.google.com/search
# 相当于发送GET请求,实际请求的URL为`https://www.google.com/search?q=kitties&count=20`
# 若去掉`-G`选项,则变为发送`POST`请求的数据体
```

### -H

增加请求头。

```shell
$ curl -H 'accept-language: en-US' https://www.google.com
```

### -i

打印响应头和响应内容。

```shell
$ curl -i https://www.example.com
HTTP/2 200
accept-ranges: bytes
age: 451289
cache-control: max-age=604800
content-type: text/html; charset=UTF-8
date: Fri, 26 Mar 2021 05:48:02 GMT
etag: "3147526947"
expires: Fri, 02 Apr 2021 05:48:02 GMT
last-modified: Thu, 17 Oct 2019 07:18:26 GMT
server: ECS (oxr/830C)
vary: Accept-Encoding
x-cache: HIT
content-length: 1256

<!doctype html>
<html>
<head>
    <title>Example Domain</title>
...
</body>
</html>
```

### -I, --head

向服务器发出`HEAD`请求，打印响应头。

```shell
$ curl -I https://www.example.com
HTTP/2 200 
content-encoding: gzip
accept-ranges: bytes
age: 563101
cache-control: max-age=604800
content-type: text/html; charset=UTF-8
date: Fri, 26 Mar 2021 05:59:42 GMT
etag: "3147526947"
expires: Fri, 02 Apr 2021 05:59:42 GMT
last-modified: Thu, 17 Oct 2019 07:18:26 GMT
server: ECS (oxr/8325)
x-cache: HIT
content-length: 648
```

### -L

使请求跟随服务器的重定向。默认不跟随重定向。

```shell
$ curl -L https://ff.sdo.com
```

### --limit-rate

限制请求和响应的带宽，用于模拟网速慢的环境。

```shell
$ curl --limit-rate 200k https://www.google.com    # 限速200kB/s
```

### -o, -O

将服务器的响应保存成文件，等同于 `wget` 命令。

```bash
$ curl -o example.html https://www.example.com     # 保存到`example.html`文件中
$ curl -O https://www.example.com/foo/bar.html     # 保存到`bar.html`文件中

$ curl -O http://www.example.com/foo/bar_[0-23].gz   # 批量下载`bar_0.gz`到`bar_23.gz`
$ curl
```

### -T

上传本地文件。

### -u

设置服务器认证的用户名和密码。

```shell
$ curl -u 'bob:12345' https://google.com/login     # 设置用户名为`bob`,密码为`12345`
                                                   # 将被转换为请求头`Authorization: Basic Ym9iOjEyMzQ1`
```

### -v

输出通信的整个过程，用于调试。

```shell
$ curl -v https://www.example.com
* Uses proxy env variable NO_PROXY == '127.0.0.0/8,localhost,192.168.0.0/16,100.64.0.0/16,10.147.0.0/16,.tensorstack.net,.tsz.io,tsz.io'
*   Trying 93.184.216.34...
* TCP_NODELAY set
* Connected to www.example.com (93.184.216.34) port 443 (#0)
* ALPN, offering h2
* ALPN, offering http/1.1
* successfully set certificate verify locations:
*   CAfile: /etc/ssl/cert.pem
  CApath: none
* TLSv1.2 (OUT), TLS handshake, Client hello (1):
* TLSv1.2 (IN), TLS handshake, Server hello (2):
* TLSv1.2 (IN), TLS handshake, Certificate (11):
* TLSv1.2 (IN), TLS handshake, Server key exchange (12):
* TLSv1.2 (IN), TLS handshake, Server finished (14):
* TLSv1.2 (OUT), TLS handshake, Client key exchange (16):
* TLSv1.2 (OUT), TLS change cipher, Change cipher spec (1):
* TLSv1.2 (OUT), TLS handshake, Finished (20):
* TLSv1.2 (IN), TLS change cipher, Change cipher spec (1):
* TLSv1.2 (IN), TLS handshake, Finished (20):
* SSL connection using TLSv1.2 / ECDHE-RSA-AES128-GCM-SHA256
* ALPN, server accepted to use h2
* Server certificate:
*  subject: C=US; ST=California; L=Los Angeles; O=Internet Corporation for Assigned Names and Numbers; CN=www.example.org
*  start date: Nov 24 00:00:00 2020 GMT
*  expire date: Dec 25 23:59:59 2021 GMT
*  subjectAltName: host "www.example.com" matched cert's "www.example.com"
*  issuer: C=US; O=DigiCert Inc; CN=DigiCert TLS RSA SHA256 2020 CA1
*  SSL certificate verify ok.
* Using HTTP2, server supports multi-use
* Connection state changed (HTTP/2 confirmed)
* Copying HTTP/2 data in stream buffer to connection buffer after upgrade: len=0
* Using Stream ID: 1 (easy handle 0x7fb64a00d600)
> GET / HTTP/2
> Host: www.example.com
> User-Agent: curl/7.64.1
> Accept: */*
> 
* Connection state changed (MAX_CONCURRENT_STREAMS == 100)!
< HTTP/2 200 
< age: 595764
< cache-control: max-age=604800
< content-type: text/html; charset=UTF-8
< date: Fri, 26 Mar 2021 06:01:37 GMT
< etag: "3147526947+ident"
< expires: Fri, 02 Apr 2021 06:01:37 GMT
< last-modified: Thu, 17 Oct 2019 07:18:26 GMT
< server: ECS (oxr/830D)
< vary: Accept-Encoding
< x-cache: HIT
< content-length: 1256
< 
<!doctype html>
<html>
<head>
    <title>Example Domain</title>
</body>
</html>
* Connection #0 to host www.example.com left intact
* Closing connection 0
```

### -x

指定HTTP请求的代理。

```shell
$ curl -x socks5://james:cats@myproxy.com:8080 https://www.example.com
```

### -X

指定HTTP请求的方法。

```shell
$ curl -X POST https://www.example.com
```

### 常见应用

## scp

## rsync

rsync 是一个高效的远程和本地文件同步工具。rsync 使用的算法能够通过仅移动文件的变动部分最小化需要复制的数据量。

```shell
# 将dir1目录下的所有文件全部递归地同步到dir2目录下
$ rsync -a dir1/ dir2

# 将dir1目录及其之下的所有文件全部递归地同步到dir2目录下
$ rsync -a dir1 dir2

# 将当前目录下的所有文件全部递归地同步到远程主机的指定目录下(通过SSH连接,下同)
$ rsync -a ./ username@host:/path/to/dir

# 将远程主机的指定目录下的所有文件全部递归地同步到当前目录下
$ rsync -a username@host:/path/to/dir/ .

# 将远程主机的dir1目录下的所有文件全部递归地同步到另一个远程主机的dir2目录下
$ rsync -a username@host1:/path/to/dir1/ username@host2:/path/to/dir2
```

### -n

## wget

wget 是一个常用的命令行工具，用于下载文件。wget 非常稳定，其对带宽很窄、网络不稳定等情况均有很强的适应性。

```shell
$ wget <url>                  # 下载url位置的文件
$ wget -O <filename> <url>    # 下载文件并命名,默认名称为url最后一个`/`之后的字符串
$ wget --limit-rate=1M <url>  # 限速下载
$ wget -c <url>               # 断点续传,即继续下载中断的文件
$ wget -b <url>               # 后台下载
$ wget -i <urllistfile>       # 批量下载:从文本文件中读取所有需要下载的url,每个url占一行
$ wget -o download.log <url>  # 将下载信息保存到日志文件,而不显示在终端
```

### 常见应用

```shell
# 镜像网站
$ wget
```

