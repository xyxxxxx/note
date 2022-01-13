# Linux 命令

## 帮助

### info

查看命令的 `info` 格式的帮助文档。

```shell
$ info ls
10.1 ‘ls’: List directory contents
==================================

The ‘ls’ program lists information about files (of any type, including
directories).  Options and file arguments can be intermixed arbitrarily,
as usual.

...
```

### man

查看命令的使用说明。

```shell
$ man ls
NAME
       ls - list directory contents

SYNOPSIS
       ls [OPTION]... [FILE]...

DESCRIPTION
       List information about the FILEs (the current directory by default).  Sort entries
       alphabetically if none of -cftuvSUX nor --sort is specified.

       Mandatory arguments to long options are mandatory for short options too.

       -a, --all
              do not ignore entries starting with .
              
       ...
```

### type

查看命令是否是 Shell 的内置命令。

```shell
$ type echo                     # 内置命令
echo is a shell builtin
$ type python                   # 外部程序
python is /Users/xyx/.pyenv/shims/python

$ type -a echo                  # 既是内置命令,也有对应的外部程序
echo is a shell builtin
echo is /usr/local/opt/coreutils/libexec/gnubin/echo
echo is /bin/echo
$ type -a python                # 所有对应的外部程序,按照PATH环境变量中各个路径的顺序
python is /Users/xyx/.pyenv/shims/python
python is /usr/bin/python
```

### whatis

在一些特定的包含系统命令的简短描述的数据库文件中查找关键字，并把结果发送到标准输出。whatis 数据库文件是由 `/usr/sbin/makewhatis` 命令建立的。

```shell
$ whatis ls
ls(1)                    - list directory contents
```

### which

在环境变量 `PATH` 指定的路径中查找命令的位置，并返回第一个搜索结果。

```shell
$ which cp
/usr/local/opt/coreutils/libexec/gnubin/cp
$ which python
/Users/xyx/.pyenv/shims/python
```

## 文件

### cat

连接文件并打印到标准输出设备上。

```shell
$ cat file1              # 打印文件内容
$ cat file1 > file2      # 将文件内容写入另一文件
$ cat file1 >> file2     # 将文件内容添加到另一文件末尾
$ cat /dev/null > file1  # 清空文件内容,等同于`:> file1`
#     -n                    打印行号
#     -s                    连续两行以上的空白行替换为一行空白行
```

### chmod

修改文件或目录的权限。

```shell
# u=user, g=group, o=other group, a=ugo
$ chmod u=rwx,g=rx,o=x file1     # rwxr-x--x
$ chmod 777 file1                # rwxrwxrwx
$ chmod a+r file1                # ugo增加权限r
$ chmod +rx file1                # ugo增加权限rx
$ chmod ug+w, o-w file1          # ug增加权限w,o减少权限w
```

### cp

复制文件或目录。

```shell
$ cp file1 file2        # 复制文件到指定文件
$ cp -r dir1 dir2       # 复制目录及其下所有文件到指定目录
#    -f                   直接覆盖同名文件而不询问
#    -i                   覆盖同名文件前询问用户
#    -p                   保留原文件的修改时间和访问权限
```

### diff

逐行比较文本文件的差异。

```shell
$ diff file1 file2
#      -B                 忽略空白行
#      -c                 显示全文并标出不同之处
#      -q                 仅显示是否有差异,不显示详细信息
#      -w                 忽略whitespace
#      -W                 指定栏宽
#      -y                 并排格式输出
```

### ln

为文件创建链接。

```shell
$ ln -s file1 file2     # 软链接(符号链接),相当于Windows的快捷方式.lnk
$ ln file1 file2        # 硬链接,相当于文件有多个名称
```

### mktemp

创建临时文件或目录。

```shell
$ mktemp                # 创建临时文件
#        -d             # 创建临时目录
#        -p [path]      # 在指定路径下创建临时文件
```

```shell
$ TMPFILE=$(mktemp)     # 创建临时文件
$ echo "Our temp file is $TMPFILE"
$ rm -f "$TMPFILE"
```

```sh
#!/bin/bash

trap 'rm -f "$TMPFILE"' EXIT    # 脚本退出时删除临时文件

TMPFILE=$(mktemp) || exit 1
ls /etc > $TMPFILE
if grep -qi "kernel" $TMPFILE; then
  echo 'find'
fi
```

### more

分页显示文本文件内容，按 ++space++ 到下一页，++b++ 键到上一页，++q++ 键退出。

```shell
$ more file1
$ more -5 file1         # 单次显示5行
$ more +5 file1         # 从第5行开始显示
```

### mv

移动文件或目录，也可用于文件和目录的重命名。

```shell
$ mv file1 dir1         # 移动文件到目录下
$ mv file1 file2        # 移动文件并重命名
$ mv dir1 dir2          # 移动目录并重命名
#    -f                   直接覆盖同名文件而不询问
#    -i                   覆盖同名文件前询问用户
#    -n                   不覆盖同名文件
```

### rename

批量重命名文件。

```shell
$ rename -x *.bak       # 去掉当前目录下所有BAK文件的后缀名    
$ rename -s abc def *   # 将当前目录下所有文件的文件名中的'abc'替换为'def'
$ rename -s abc '' *    # 去掉当前目录下所有文件的文件名中的'abc'
```

在没有安装 `rename` 的情形下：

```shell
# 脚本
```

### rm

删除文件或目录。

```shell
$ rm file1
$ rm -r dir1            # 删除目录及其下所有文件
#    -f                   直接删除只读文件
```

### sort

对文本文件内容进行排序。

```shell
$ sort file1            # 将文件各行按ASCII的字典序进行排序
#      -d                 忽略英文字母、数字、whitespace之外的所有字符
#      -f                 将小写字母视作大写字母
#      -i                 忽略040至176之间的ASCII字符之外的所有字符
#      -r                 排逆序
#      -n                 按数值大小排序
```

### tail

查看文本文件内容。

```shell
$ tail -n 5 file1       # 显示最后5行
$ tail -n +5 file1      # 显示从第5行到文件末尾
$ tail -f file1         # 循环读取并打印更新的内容,常用于查看正在写入内容的日志文件
```

### tar

备份文件。

```shell
$ tar -cvf dir1.tar dir1                # 备份目录dir1
$ tar -zcvf dir1.tar.gz dir1            # 备份目录dir1并压缩
$ tar -xvf dir1.tar [-C dir1]           # 还原目录dir1
$ tar -zxvf dir1.tar.gz [-C dir1]       # 解压缩并还原目录dir1
#     -c                                  新建备份文件
#     -C                                  指定目标目录
#     -f                                  指定备份文件
#     -v                                  显示执行过程
#     -x                                  从备份文件还原文件
#     -z                                  通过gzip指令压缩或解压缩文件
```

### touch

新建空文件，或修改文件或目录的时间属性。

```shell
$ touch file1             # 新建空文件,或修改文件的时间属性为当前系统时间
```

### unzip

解压缩文件。

```shell
$ unzip dir1.zip -d dir1  # 将压缩文件解压到目录dir1下
#       -c                  显示解压过程
#       -l                  显示压缩文件内包含的文件
```

### zip

压缩文件。

```shell
$ zip -r dir1.zip dir1   # 将目录dir1打包为压缩文件
```

## 目录

### cd

切换当前工作目录。

```shell
$ cd                     # home
$ cd ~                   # home
$ cd /                   # 根目录
$ cd ..                  # 上级目录
$ cd -                   # 上一个目录
$ cd dir1                # 指定目录
```

### find

在目录下查找文件。

```shell
$ find . -name "*.c"     # 在当前目录下查找名称匹配`*.c`的文件
$ find . -mtime -20      #               20天内修改过的文件
$ find . -ctime +20      #               20天前创建的文件
$ find . -type f         #               类型为f(normal)的文件
$ find . -size +100c     #               大小大于100字节的文件(c,k,M分别表示B,kB,MB)
$ find . -size 0 -exec ls -l {} \;     # 空文件,并对这些文件执行指定命令
```

### ls

显示目录下所有文件和子目录。

```shell
$ ls
$ ls -lrt                # 显示文件和目录的详细信息,按从旧到新的顺序排列
$ ls s*                  # 显示文件名匹配`s*`的文件
$ ls -a                  # 显示`.`开头的隐藏文件
#    --block-size          在-l的基础上,以指定单位显示文件大小
#    -h                    在-ls的基础上,使用人类可读的文件大小格式
#    -l                    显示文件权限、所有者、大小、创建时间等详细信息
#    -r                    逆序显示
#    -S                    按文件大小从大到小排列
#    -t                    按文件创建时间的先后顺序排列
#    -X                    按后缀名的首字母顺序排列
#                          默认按文件名的首字母顺序排列
```

```shell
$ ls -lrth
-rw-r--r-- 1 xyx staff  96M Feb  7  2021  googlechrome.dmg
-rw-r--r-- 1 xyx staff  11M Feb  7  2021  Typora.dmg
-rw-r--r-- 1 xyx staff  98M Feb  7  2021  VSCode-darwin.zip
-rw-r--r-- 1 xyx staff 124M Feb  7  2021 'Firefox 85.0.1.dmg'
-rw-r--r-- 1 xyx staff 1.3K Mar 25  2021  docs.svg
-rw-r--r-- 1 xyx staff 3.5K Mar 25  2021  favorites.svg
-rw-r--r-- 1 xyx staff 3.0M Jul 14 14:51  Golang101-v1.16.b.pdf-SongTi.pdf
```

### mkdir

新建目录。

```shell
$ mkdir dir1
$ mkdir -p dir1/dir2     # 创建不存在的多级目录
```

### pwd

显示当前工作目录。

```shell
$ pwd
```

### rmdir

删除空目录。删除非空目录请使用命令 `rm`。

```shell
$ rmdir dir1
#       -p       若空目录被删除后其上级目录成为空目录,则一并删除
```

##  文本

### echo

打印文本。

```shell
$ echo abc123            # 打印一般字符串
abc123
$ echo "abc123"          # 打印一般字符串
abc123
$ echo $HOME             # 打印环境变量的值
/Users/xyx
$ echo \"How are you\?\" \"I\'m fine.\"     # 使用转义符号打印特殊符号
"How are you?" "I'm fine."
$ echo abc123 > file1    # 将打印内容写入文件
```

```shell
$ echo -n a; echo b      # 取消末尾自动添加的换行符
ab
$ echo -e "Hello\nWorld"    # 解释引号内的转义字符
Hello
World
```

```shell
$ echo `date`            # 打印当前日期时间
Thu Jan 6 10:24:53 CST 2022
```

### egrep

查找文本文件中符合条件的内容。与 `grep` 的区别在于解读字符串的方法：`egrep` 使用扩展正则表达式语法进行解读，而 `grep` 则使用基本正则表达式语法进行解读，前者比后者的表达更规范。

```shell
# 查找文本
$ egrep [str/re] file1      # 在文件中使用字符串/正则表达式搜索
$ egrep -r [str/re] dir1    # 在目录中使用字符串/正则表达式搜索
```

### grep

查找文本文件中符合条件的内容。

```shell
$ grep [str] file1          # 查找并打印文本文件中包含该字符串的所有行
#      -e                     使用正则表达式匹配字符串
#      -i                     忽略字符的大小写
#      -r                     递归地查找目录下的所有文件
#      -v                     查找不包括该字符串的所有行
```

### printf

打印字符串。

```shell
$ printf "abc123\n"
abc123
$ printf "%-10s %-1s %-2d %-5.1f %-4.1f\n" Alice F 10 140.2 31.5
Alice      F 10 140.2 31.5
```

### wc

统计文本的字数。

```shell
$ echo -e "This is line 1\nThis is line 2\nAnd this is line 3\nOver" >> test
$ wc test                 # test文件的统计信息
 4 14 54 test             # 行数4,单词数14,字节数54
$ wc -l test              # 行数
4 test
$ wc -w test              # 单词数
14 test
$ wc -c test              # 字节数
54 test
```

```shell
$ ls
file1  file2  file3  file4
$ ls | wc -l              # 将ls的打印结果作为wc的输入,以统计文件/目录数
4
```

## Shell

### clear

清空命令行。

```shell
$ clear
```

### env

显示环境变量。

```shell
$ env
```

### exit

退出命令行。

```shell
$ exit
$ exit -1             # 以指定状态值退出
```

### export

设置或显示环境变量。

```shell
$ export -p           # 显示Shell提供的环境变量
$ export MYENV=1      # 定义环境变量,仅在当前Shell中有效
$ export -n MYENV     # 删除环境变量,仅在当前Shell中有效
```

### history

查看最近执行过的命令。

```shell
$ history
...
498 echo Goodbye
499 ls ~
500 cd
```

### set

设置或显示变量。

```shell
$ set                 # 显示变量和Shell函数
$ set MYENV=1         # 定义变量,仅在当前Shell中有效
```

```shell
$ set -e              # 任意命令以非零值退出时,脚本终止执行
$ set -o pipefail     # 任意子命令失败则视作整个管道命令失败
```

```shell
$ set -f      # 关闭模式扩展(globbing)
```

```shell
$ set -n      # 不执行命令,只检查语法是否正确
```

```shell
$ echo $a     # 未定义的变量
              # 打印空字符串
$ set -u      # 当执行任意命令遇到未定义的变量时,抛出错误信息
$ echo $a
bash: a: unbound variable     # 报错
$ set +u
```

```shell
$ set -x      # 执行任意命令时,先显示该命令及其参数
$ echo abc
+ echo abc
abc
$ ls
+ ls
folder1 folder2 file1 file2
$ set +x
+ set +x
```

### shopt

设置 Bash 的选项。

```shell
$ shopt               # 查看所有选项的开关状态
$ shopt -s option     # 打开某个选项
$ shopt -u option     # 关闭某个选项
```

### tee

读取标准输入并同时写入到标准输出和文件。

```shell
$ echo abc | tee file1
abc
$ ls
file1
$ cat file1
abc
```

### time

执行指定命令并计时。

```shell
$ time ls
folder1 folder2 file1 file2

real	0m0.007s
user	0m0.002s
sys	0m0.002s

$ time sleep 5

real	0m5.021s
user	0m0.001s
sys	0m0.002s
```

### unset

删除变量。

```shell
$ unset MYENV         # 删除变量,仅在当前Shell中有效
# 相当于
$ MYENV=

$ unset -f myfunc     # 删除函数
```

### xargs

`xargs` 是给命令传递参数的一个过滤器，通常和管道一起使用，相当于将前一命令的输出解析为后一命令的参数。

```shell
$ cat test.txt
a b c d e f g
h i j k
$ cat test.txt | xargs
a b c d e f g h i j k
$ cat test.txt | xargs -n3    # 多行输出,每行3个参数
a b c
d e f
g h i
j k
$ echo "nameXnameXnameXname" | xargs -dX    # 以'X'分割字符串并输出
name name name name
```

```shell
# 将当前目录下所有.jpg文件备份并压缩
$ find . -type f -name "*.jpg" -print | xargs tar -czvf images.tar.gz
```

```shell
# -I指定占位符,占位符指定参数传入的位置
# 将当前目录下所有.jpg文件复制到/data/images路径下
$ ls *.jpg | xargs -n1 -I {} cp {} /data/images
```

```shell
$ cat url-list.txt | xargs wget -i      # 批量下载`url-list.txt`中的链接
```

## 磁盘

### df

显示当前文件系统的磁盘使用情况统计。

```shell
$ df
Filesystem     1K-blocks      Used Available Use% Mounted on
udev             4003836         0   4003836   0% /dev
tmpfs             805676      1928    803748   1% /run
/dev/sda2      114337956  10797408  97689412  10% /
tmpfs            4028364         0   4028364   0% /dev/shm
tmpfs               5120         4      5116   1% /run/lock
tmpfs            4028364         0   4028364   0% /sys/fs/cgroup
...
$ df -h         # 使用人类可读的单位格式
Filesystem      Size  Used Avail Use% Mounted on
udev            3.9G     0  3.9G   0% /dev
tmpfs           787M  1.9M  785M   1% /run
/dev/sda2       110G   11G   94G  10% /
tmpfs           3.9G     0  3.9G   0% /dev/shm
tmpfs           5.0M  4.0K  5.0M   1% /run/lock
tmpfs           3.9G     0  3.9G   0% /sys/fs/cgroup
...
```

### du

显示目录或文件的大小。

```shell
$ du                  # 显示所有目录的大小
$ du dir1             # 显示指定路径下的所有目录的大小
#    -a                 显示所有目录和文件的大小
#    -h                 使用人类可读的单位格式
#    --max-depth=n      只深入n层目录
#    -s                 只显示总计大小
```

```
$ du -sh .
497G
```

##  网络

### hostname

显示或设置（仅限超级用户）主机名称。

```shell
$ hostname
192.168.0.100
# 在 /etc/sysconfig/network 位置修改
```

### ifconfig

显示或设置网络设备。

```shell
$ ifconfig                    # 显示网络设备
$ ifconfig eth0 down          # 关闭网络设备
$ ifconfig eth0 up            # 启动网络设备
$ ifconfig eth0 hw ether 00:AA:BB:CC:DD:EE              # 为eth0网卡修改MAC地址
$ ifconfig eth0 192.168.1.56 netmask 255.255.255.0      # 为eth0网卡配置IP地址,并加上子掩码
# 在 /etc/sysconfig/network-scripts/ifcfg-<name> 位置修改
```

### lsof

检查端口状态。

```shell
$ lsof -i:8080                # 检查8080端口的使用状态
```

### nc

设置路由器。

### netstat

显示网络状态。

```shell
$ netstat -an                 # 显示详细网络状态
$ netstat -i                  # 显示网络设备信息
$ netstat -rn                 # 显示本地路由表
$ netstat -s                  # 显示网络统计信息
$ netstat -l                  # 显示监听的服务器的套接字
$ netstat -tunlp | grep 8080  # [Linux]显示进程8080占用端口状态
```

### ping

检测与主机的连接。

```shell
$ ping www.bilibili.com
#      -c [num]    发送信息的次数
#      -i [secs]   发送的间隔秒数
```

## 系统

### date

显示或设置系统的日期时间。

```shell
$ date
Mon Jun 28 10:31:54 CST 2021
```

### declare

声明变量。

```shell
$ declare         # 打印当前环境的所有变量
$ declare -p      # 打印所有变量的信息
$ declare -p foo  # 打印指定变量的信息
$ declare -f      # 打印所有函数的信息
$ declare -f foo  # 打印指定函数的信息
$ declare -F      # 打印所有函数的名称
```

```shell
$ declare -i a    # 声明整数型变量
$ a=2*3
$ echo $a
6

$ declare -x a    # 声明环境变量
# 相当于
$ export a

$ declare -r a=1  # 声明只读变量
$ a=2
bash: read-only variable: a

$ declare -A colors   # 声明关联数组(映射)
$ colors["red"]="#ff0000"
$ colors["green"]="#00ff00"
$ colors["blue"]="#0000ff"
$ echo ${colors["red"]}
#ff0000
$ echo ${colors[@]}
#0000ff #ff0000 #00ff00
$ echo ${#colors[@]}
3
$ echo ${!colors[@]}
blue red green
```

### free

显示内存使用情况。

```shell
$ free     #      显示内存使用情况
#      -b         以B为单位显示
#      -k         以kB为单位显示
#      -m         以MB为单位显示
#      -h         以合适的单位显示
#      -s [secs]  每`secs`秒执行一次命令
```

### groupadd

创建新用户。

```shell
$ groupadd office      # 创建一般用户
#          -r            创建系统群组,其ID应小于500
#          -g 1000       指定群组ID
```

### htop

实时显示资源占用和进程状态。

### id

显示用户的 ID 以及所属群组的 ID。

```shell
$ id          # 显示全部
uid=0(root) gid=0(root) groups=10(wheel)
$ id -u       # 用户ID
501
$ id -un      # 用户名
xyx
$ id -g       # 群组ID
20
$ id -gn      # 群组名
staff
```

### kill

杀死进程。

```shell
$ kill -l             # 列出所有可用信号
HUP INT QUIT ILL TRAP ABRT EMT FPE KILL BUS SEGV SYS PIPE ALRM TERM URG STOP TSTP CONT CHLD TTIN TTOU IO XCPU XFSZ VTALRM PROF WINCH INFO USR1 USR2
$ kill 12345          # 正常终止进程12345
$ kill -9 12345       # 强制杀死进程12345
$ kill -KILL 12345    # 强制杀死进程12345
```

### reboot

重新启动计算机。

```shell
$ reboot
```

### rpm

管理套件。RPM（Redhat Package Manager）原本是 Red Hat Linux 发行版专门用来管理 Linux 各项套件的程序，由于它遵循 GPL 规则且功能强大方便，因而广受欢迎，逐渐受到其他发行版的采用。

```shell
$ rpm -a                  # 显示套件
$ rpm -e --nodeps [pkg]   # 卸载套件
```

### ps

显示当前的进程状态。

```shell
$ ps -ef             # 显示进程详细信息: USER, PID, PPID, START, CMD, ...
$ ps -aux            # [Linux]显示进程详细信息: USER, PID, %CPU, %MEM, START, CMD, ...
$ ps -A              # 显示所有进程
```

### shutdown

关闭计算机。

```shell
$ shutdown now       # 立即关闭计算机
#          -t [secs]   设定`secs`秒后运行关机程序
#          -r          重新启动计算机
```

### sleep

等待一段时间。

```shell
$ sleep 10      # 等待10s
$ sleep 10m
$ sleep 2h
$ sleep 1d
```

### sudo

以系统超级用户（管理员）的身份执行命令。

```shell
$ sudo [command]
#      -u [username]/[uid]    以用户的身份执行命令
```

### top

实时显示进程状态。

```shell
$ top
#     -p <pid>         [Linux]显示指定进程
#     -pid <pid>       [MacOS]显示指定进程
#     -d <secs>        [Linux]每`secs`秒更新
```

### uname

显示系统信息。

```shell
$ uname       # 打印内核名称
#       -a      打印所有信息
#       -n      打印网络主机名
#       -m      打印机器硬件名称
#       -o      打印操作系统名称 
```

### useradd

创建新用户。

```shell
$ useradd xyx         # 创建一般用户
#         -r            创建系统用户
#         -d            指定用户的Home目录
#         -m            创建用户的Home目录,如果其不存在
#         -u 1000       指定用户ID
#         -g 1000       指定用户所属群组的名称或ID
#         -s /bin/bash  指定用户所使用的shell
```

## 组合命令

`;`：顺序执行命令，不论成功或失败。
 
```shell
$ clear; ls
```

`&&`：如果前一命令成功，则执行后一命令（command1 succeed, then command2）。

```shell
$ cp sql.txt sql.bak.txt && cat sql.bak.txt
```

`||`：如果前一命令失败，则执行后一命令（command1 failed, then command2）。

```shell

$ ls /proc && echo success || echo failed
```

`|`：前一命令的输出作为后一命令的输入（command1 output as command2 input）。

```shell
$ ls -l /etc | more
$ echo "Hello World" | cat > hello.txt
$ ps -ef | egrep python
```

## gcc

GCC（GNU Compiler Collection，GNU 编译器套装） 是一套编译器，起初为 C 语言编译器（GNU C Compiler），之后得到扩展，支持处理 C++、Fortran、Pascal、Java、Go 等语言。许多类 Unix 系统（包括 Linux、MacOS）都采用 GCC 作为标准编译器。

GCC 支持多种硬件平台，如 ARM、x86 等，并且还支持跨平台交叉编译（即 x86 平台上编译的程序可以在 ARM 平台上运行）。

GCC 通过 `gcc` 命令使用：

```shell
$ gcc -E hello.c -o hello.i    # 预处理(Preprocess)
$ gcc -S hello.c -o hello.s    # 编译(Compile)
$ gcc -c hello.c -o hello.o    # 汇编(Assemble)
$ gcc hello.c -o hello.exe     # 链接(Link)
```

## 常用命令

展示当前的 Shell：

```shell
$ echo $SHELL
/bin/bash
```

展示当前的类 Unix 系统中安装的所有 Shell：

```shell
$ cat /etc/shells
/bin/sh
/bin/bash
/bin/rbash
/bin/dash
/usr/bin/tmux
```

展示环境变量 `PATH` 的值（每行一条路径）：

```shell
$ echo -e ${PATH//:/'\n'}
/Users/xyx/Library/Python/3.8/bin
/usr/local/opt/findutils/libexec/gnubin
/usr/local/opt/gnu-sed/libexec/gnubin
/usr/local/opt/coreutils/libexec/gnubin
/Users/xyx/.pyenv/shims
/usr/local/bin
/usr/bin
/bin
/usr/sbin
/sbin
/Applications/VMware Fusion.app/Contents/Public
/usr/local/go/bin
```
