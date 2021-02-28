##  info

```shell
$ whatis <command>      # 命令info
$ info <command>		# 命令详细info
$ man <command>		    # 命令manual

$ which <command>		# 查找命令,返回可执行程序的绝对路径
$ which mv
/bin/mv
```



##  files management

```shell
# 移动文件或目录
$ mv <file> <dir>
$ mv <file> <file>	    # 重命名
$ mv <dir> <dir>		# 移动或重命名

# 删除
$ rm <file>
$ rm -r <dir>			# 删除目录及其目录下的所有文件
#    -f: 直接删除只读文件

# 复制
$ cp <file> <dir+file>	# 复制文件
$ cp -r <dir> <dir>	    # 复制目录

# 读
$ more <file>           # 打印文件内容
$ more -5 <file>	    # 单次打印5行, space 翻页, q 退出
$ more +5 <file>		# 从第5行开始打印
$ tail -5 <file>	    # 打印最后5行
$ tail +5 <file>		# 从第5行开始打印
$ tail -f <file>		# 打印更新内容

# 写
$ cat <file>		         # 打印文件内容
$ cat <file1> > <file2>	     # 将打印内容写入文件
$ cat <file1> >> <file2>     # 将打印内容添加到文件末尾
#     -n: 打印行号
#     -s: 连续两行以上的空白行替换为一行空白行
$ :> <file>			         # 清除文件内容

# 创建
$ touch <file>           # 新建空文件

# 权限, u=user,g=group,o=other group,a=ugo
$ chmod u=rwx,g=rx,o=x <file>    # rwxr-x--x
$ chmod 777 <file>               # rwxrwxrwx
$ chmod a+r <file>               # ugo + r
$ chmod ug+w,o-w <file>          # ug + w, o - w

# 比较
$ diff <file1> <file2>	     # 比较文件
#      -y: 并排格式输出

```



##  files operation

```shell
# 排序
$ sort <file>			# 将文件各行按ASCII序排序
#      -r: 排逆序
#      -n: 按数值大小排序

# 链接
$ ln -s <file1> <file2>	 # 软链接(符号链接),相当于Windows的快捷方式.lnk
$ ln <file1> <file2>	 # 硬链接,相当于文件有多个名称

# 压缩
$ tar -cvf <file.tar> <dir>	            # 备份
$ tar -zcvf <file.tar.gz> <dir>	        # 备份并压缩
$ tar -xvf <file.tar> [-C <dir>]        # 还原              
$ tar -zxvf <file.tar.gz> [-C <dir>]    # 解压缩并还原
#     -c: 新建备份文件
#     -C: 指定目的目录
#     -f: 指定备份文件(名)
#     -v: 显示执行过程
#     -x: 从备份文件还原文件
#     -z: 通过gzip指令压缩或解压缩文件

$ xz -z <file.tar>                      # 备份
$ xz -d <file.tar.xz>                   # 备份并压缩
#    -k: keep origin file
```



##  dir

```shell
# 查看目录
$ pwd					# current dir

# 切换目录
$ cd					# home
$ cd /			        # root
$ cd ~			        # home
$ cd -				    # last dir
$ cd <dir>			    # specified dir
$ cd ..			        # parent dir 

# 新建与删除
$ mkdir <dir>			# 创建目录
$ mkdir -p <dir1>/<dir2>  # 即使dir1不存在,依然创建
$ rmdir <dir>			# 删除目录
$ rm -r <dir>			# 删除目录及其目录下的所有文件

# 展示目录下文件
$ ls                    # list every file
$ ls -lrt               # list with info
$ ls -lrt s*            # list matching certain name
$ ls -a                 # list hidden file

# 文件系统下查找文件
$ find . -name "*.c"    # by name in current dir
$ find . -mtime -20     # revised in 20d
$ find . -ctime +20     # created before 20d
$ find . -type f        # type f:normal
$ find . -size +100c    # size > 100 Bytes, c,k,M
$ find / -type f -size 0 -exec ls -l {} \;	# find every size 0 normal file in \, show path
```



##  text

```shell
# 查找文本
$ egrep <str>/<re> <file>      # search in file by str/regular expression
$ egrep -r <str>/<re> <dir>    # search in dir by str/regular expression
```



##  logic

```shell
#  &&: command1 succeed, then command2
$ cp sql.txt sql.bak.txt && cat sql.bak.txt

#  ||: command1 failed, then command2
$ ls /proc && echo success || echo failed

#  (;) 

#  |: command1 output as command2 input
$ ls -l /etc | more
$ echo "Hello World" | cat > hello.txt
$ ps -ef | grep <str>

```



##  Shell

```shell
$ echo <str>           # 打印字符串 str

$ clear                # 清空命令行
$ exit                 # 退出命令行

# xargs 是给命令传递参数的一个过滤器,相当于把前面传来的文本作为参数解析
# 一般用法为 command1 |xargs -item command2
$ cat test.txt | xargs	    # 单行输出
$ cat test.txt | xargs -n3	# 多行输出,每行3个参数
$ echo "nameXnameXnameXname" | xargs -dX	# 用X分割并输出

# -I指定占位符,占位符指定参数传入的位置
$ ls *.jpg | xargs -n1 -I {} cp {} /data/images
# 将当前目录下所有.jpg文件备份并压缩
$ find . -type f -name "*.jpg" -print | xargs tar -czvf images.tar.gz
# 按照url-list.txt内容批量下载
$ cat url-list.txt | xargs wget -i
```



##  net

```shell
# 显示hostname
$ hostname
# revise at /etc/sysconfig/network

# ipconfig
$ ifconfig
# revise at /etc/sysconfig/network-scripts/ifcfg-<name>

# 检查端口状态
$ lsof -i:8080

# 显示网络状态
$ netstat -a                    # 显示详细的网络状态
$ netstat -tunlp | grep 8080    # 显示进程8080占用端口状态

# network restart
$ nmcli c reload

# 请求服务器
$ curl https://www.example.com          # 打印响应内容,默认为GET方法
$ curl -X POST https://www.example.com  # 指定请求方法
$ curl -i https://www.example.com       # 打印响应头和响应内容
$ curl -I https://www.example.com       # 打印响应头
$ curl -d 'user=xiaoruan&age=22' https://www.example.com  # 指定参数

# 下载到当前目录
$ wget [url]                  # 下载url的资源
$ wget -O [filename] [url]    # 下载并命名,默认名称为url最后一个/后面的字符串
$ wget --limit-rate=1M [url]  # 限速下载
$ wget -c [url]               # 断点续传
$ wget -b [url]               # 后台下载
$ wget -i [urllistfile]       # 批量下载
```



##  system management

```shell
$ sudo                # run as root
$ sudo -u             # run as user

$ ps -ef			  # 查看所有进程: USER, PID, PPID, START, CMD, ...
$ ps -aux             # 查看所有进程: USER, PID, %CPU, %MEM, START, CMD, ...

$ top                 # 显示实时进程,相当于Windows的任务管理器
$ top -d 2            # 每2s更新
$ top -p <num>        # 显示指定进程

$ kill <num>		  # 杀死进程
$ kill -9 <num>	      # 强制杀死进程

$ date                # 打印当前时间

$ shutdown
$ reboot

```



##  system setting

```shell
# redhat package manager
$ rpm -a				          # show packages
$ rpm -qa | grep <name>	  # search package with name
$ rpm -e --nodeps <name>  # uninstall package

# 环境变量
$ env                     # 显示所有环境变量
$ export                  # 显示所有环境变量

$ echo $ENV               # 使用环境变量的值

$ export VAR=10           # 新增或修改环境变量
$ unset VAR               # 删除环境变量
                          # 新增,修改和删除操作仅在当前终端有效
```



##  gcc

```shell
$ gcc -E hello.c -o hello.i    # Preprocess
$ gcc -S hello.c -o hello.s    # Compile
$ gcc -c hello.c -o hello.o    # Assemble
$ gcc hello.c -o hello.exe     # Link
```

