# click

[click](https://click.palletsprojects.com/en/8.1.x/) 是一个用于创建漂亮的命令行界面的 Python 包，其采用可组合的方法，只需要很少的代码修改。

click 的目标是使得编写命令行工具的过程变得快速而有趣，同时防止因为不能实现想要的 CLI API 导致的失望。

## 命令

```python
@click.command()    # Click命令
@click.option('--count', default=1, help='number of greetings')  # 选项
@click.argument('name')                                          # 参数
def hello(count, name):
    for x in range(count):
        click.echo(f"Hello {name}!")
```

## 命令组

```python
@click.group()      # Click命令组
def cli():
    pass

@cli.command()      # 添加为命令组cli的命令
def initdb():
    click.echo('Initialized the database')

@cli.command()
def dropdb():
    click.echo('Dropped the database')
```

```python
# command.py
@click.command()
def greet():
    click.echo("Hello, World!")

# group.py
from command import greet

@click.group()      # Click命令组
def cli():
    pass

group.add_command(greet)   # 添加命令
```

## 命令选项

选项名称与被装饰函数的参数名称的关系：

```python
@click.command()
@click.option('-s')
def echo(s):
    click.echo(s)

@click.command()
@click.option('-s', '--string-to-echo')
def echo(string_to_echo):
    click.echo(string_to_echo)

@click.command()
@click.option('-s', '--string-to-echo', 'string')
def echo(string):
    click.echo(string)
```

单值选项：

```python
@click.command()
@click.option('--n', default=1, show_default=True)  # 提供默认值，参数类型从默认值推断
def dots(n):                                        # 在帮助信息中展示默认值
    click.echo('.' * n)

@click.command()
@click.option('--n', required=True, type=int)  # 使选项成为必填项，显式指定参数类型
def dots(n):
    click.echo('.' * n)
```

多值选项：

```python
@click.command()
@click.option('--pos', nargs=2, type=float)    # 指定数量和类型（同一类型）
def findme(pos):
    a, b = pos
    click.echo(f"{a} / {b}")
```

```shell
$ findme --pos 2.0 3.0
2.0 / 3.0
```

```python
@click.command()
@click.option('--item', type=(str, int))       # 分别指定类型
def putitem(item):
    name, id = item
    click.echo(f"name={name} id={id}")
```

```shell
$ putitem --item peter 1338
name=peter id=1338
```

多重选项：

```python
@click.command()
@click.option('--message', '-m', multiple=True)
def commit(message):    # 以元组的形式传入
    click.echo('\n'.join(message))
```

```shell
$ commit -m foo -m bar
foo
bar
```

计数选项：

```python
@click.command()
@click.option('-v', '--verbose', count=True)
def log(verbose):
    click.echo(f"Verbosity: {verbose}")
```

```shell
$ log -vvv
Verbosity: 3
```

布尔选项：

```python
@click.command()
@click.option('-s', '--shout', is_flag=True)
def hello(shout):
    greeting = 'Hello'
    if shout:
        greeting = greeting.upper() + '!'
    else:
        greeting = greeting + '.'
    click.echo(greeting)
```

```shell
$ hello
Hello.
$ hello --shout
HELLO!
```

特性切换选项：

```python
@click.command()
@click.option('--transform', 'transform', default='')
@click.option('--upper', 'transform', flag_value='upper')  # 额外的选项为参数赋指定值
@click.option('--lower', 'transform', flag_value='lower')
def hello(transform):
    greeting = 'Hello'
    if transform:
        click.echo(getattr(greeting, transform)())
    else:
        click.echo(greeting)
```

选择选项：

```python
@click.command()
@click.option('--hash-type',
              type=click.Choice(['MD5', 'SHA1'], case_sensitive=False))
def digest(hash_type):
    click.echo(hash_type)
```

```shell
$ digest --hash-type=MD5
MD5

$ digest --hash-type=md5
MD5

$ digest --hash-type=foo
Usage: digest [OPTIONS]
Try 'digest --help' for help.

Error: Invalid value for '--hash-type': 'foo' is not one of 'MD5', 'SHA1'.

$ digest --help
Usage: digest [OPTIONS]

Options:
  --hash-type [MD5|SHA1]
  --help                  Show this message and exit.
```

提示输入：

```python
@click.command()
@click.option('--name', prompt=True)
def hello(name):
    click.echo(f"Hello {name}!")
```

```shell
$ hello --name=John
Hello John!
$ hello
Name: John
Hello John!
```

```python
@click.command()
@click.option('--name', prompt='Please enter your name:')  # 自定义提示信息
def hello(name):
    click.echo(f"Hello {name}!")
```

```shell
$ hello
Please enter your name: John
Hello John!
```

```python
import codecs

@click.command()
@click.option(
    "--password", prompt=True, hide_input=True,            # 输入密码的解决方案
    confirmation_prompt=True
)
# equals to
# @click.password_option()
def encode(password):
    click.echo(f"encoded: {codecs.encode(password, 'rot13')}")
```

```shell
$ encode
Password: 
Repeat for confirmation: 
encoded: frperg
```

```python
import os

@click.command()
@click.option(
    "--username", prompt=True,
    default=lambda: os.environ.get("USER", ""),      # 默认值从环境变量获取
    show_default="current user"                      # 在帮助信息中展示指定内容作为默认值
)
def hello(username):
    click.echo(f"Hello, {username}!")
```

从环境变量获取选项值：

```python
@click.command()
@click.option('--username')
def greet(username):
    click.echo(f'Hello {username}!')

if __name__ == '__main__':
    greet(auto_envvar_prefix='GREETER')    # `GREETER`作为环境变量名的前缀
```

```shell
$ export GREETER_USERNAME=john
$ greet
Hello john!
```

```python
@click.group()
@click.option('--debug/--no-debug')
def cli(debug):
    click.echo(f"Debug mode is {'on' if debug else 'off'}")

@cli.command()
@click.option('--username')
def greet(username):
    click.echo(f"Hello {username}!")

if __name__ == '__main__':
    cli(auto_envvar_prefix='GREETER')
```

```shell
$ export GREETER_DEBUG=false
$ export GREETER_GREET_USERNAME=John
$ cli greet
Debug mode is off
Hello John!
```

```python
@click.command()
@click.option('--username', envvar='USERNAME')    # 直接指定环境变量名
def greet(username):
   click.echo(f"Hello {username}!")

if __name__ == '__main__':
    greet()
```

```shell
$ export USERNAME=john
$ greet
Hello john!
```

```python
@click.command()
@click.option('--path', 'paths', envvar='PATHS', multiple=True,
              type=click.Path())    # 从环境变量获取多个值
def perform(paths):
    for path in paths:
        click.echo(path)

if __name__ == '__main__':
    perform()
```

```shell
$ export PATHS=./foo/bar:./test
$ perform
./foo/bar
./test
```

## 回调

确认操作：

```python
def abort_if_false(ctx, param, value):
    if not value:
        ctx.abort()

@click.command()
@click.option('--yes', is_flag=True, callback=abort_if_false,
              expose_value=False,
              prompt='Are you sure you want to drop the db?')
# equals to
# @click.confirmation_option(prompt='Are you sure you want to drop the db?')
def dropdb():
    click.echo('Dropped all tables!')
```


## 命令参数

建议仅对路径和 URL 使用命令参数，其余参数请使用命令选项。

基本参数：

```python
@click.command()
@click.argument('filename')
def touch(filename):
    """Print FILENAME."""
    click.echo(filename)
```

```shell
$ touch foo.txt
foo.txt
```

文件参数：

```python
@click.command()
@click.argument('input', type=click.File('rb'))
@click.argument('output', type=click.File('wb'))
def copy(input, output):
    """Copy contents of INPUT to OUTPUT."""
    while True:
        chunk = input.read(1024)
        if not chunk:
            break
        output.write(chunk)
```

```shell
$ inout - hello.txt      # `-`指代stdin
hello
^D
$ inout hello.txt -      # `-`指代stdout
hello
```

文件路径参数：

```python
@click.command()
@click.argument('filename', type=click.Path(exists=True))  # 检查路径是否存在
def touch(filename):
    """Print FILENAME if the file exists."""
    click.echo(click.format_filename(filename))
```

```shell
$ touch hello.txt
hello.txt

$ touch dir
dir

$ touch missing.txt
Usage: touch [OPTIONS] FILENAME
Try 'touch --help' for help.

Error: Invalid value for 'FILENAME': Path 'missing.txt' does not exist.
```

类似选项的参数：

```python
@click.command()
@click.argument('files', nargs=-1, type=click.Path())
def touch(files):
    """Print all FILES file names."""
    for filename in files:
        click.echo(filename)
```

```shell
$ touch -- -foo.txt bar.txt  # `--`之后的部分全部视为参数
-foo.txt
bar.txt
```
