## [`argparse`](https://docs.python.org/zh-cn/3/library/argparse.html)——命令行选项、参数和子命令解析器

如果脚本很简单或者临时使用，可以使用`sys.argv`直接读取命令行参数。`sys.argv`返回一个参数列表，其中首个元素是程序名，随后是命令行参数，所有元素都是字符串类型。例如以下脚本：

```python
# test.py

import sys
print "Input argument is %s" %(sys.argv)
```

```shell
$ python3 test.py 1 2 -a 3
Input argument is ['test.py', '1', '2', '-a', '3']
```



`argparse`模块可以让人轻松编写用户友好的命令行接口。程序定义它需要的参数，然后 `argparse` 将弄清如何从 `sys.argv` 解析出那些参数。 `argparse` 模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息。

教程：[Argparse 教程](https://docs.python.org/zh-cn/3/howto/argparse.html)