# shlex——简单的词法分析

`shlex` 模块可以用于编写类似 Unix shell 的简单词法分析程序。通常可用于编写“迷你语言”（如 Python 应用程序的运行控制文件）或解析带引号的字符串。

## split(), join()

```python
>>> command_line = input()
/bin/vikings -input eggs.txt -output "spam spam.txt" -cmd "echo '$MONEY'"
>>> shlex.split(command_line)
['/bin/vikings', '-input', 'eggs.txt', '-output', 'spam spam.txt', '-cmd', "echo '$MONEY'"]
>>> shlex.join(shlex.split(command_line))
'/bin/vikings -input eggs.txt -output \'spam spam.txt\' -cmd \'echo \'"\'"\'$MONEY\'"\'"\'\''
```
