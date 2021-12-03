# 流程控制

## 条件结构

### `if`

```C
if (条件) {语句组}

if (条件)
	{语句组1}
else if
	{语句组2}
else
	{语句组3}
```



### `switch`

```c
switch (表达式)/* int or char*/
{case constant1:
 	sentence1
	break;
 case constant2:
 	sentence2
	break;
 ......
 default:
 	sentence
}
```



## 循环结构

### `while`

```c
while (condition) // if non-zero, execute sentence
{
    sentence;
}

while (1) // infinite loop
{
	sentence;    
}
    
break;    //跳出循环
continue; //立即进行下一次循环
```



### `do while`

```c
do
	{
    sentence;
	} while(condition)  
```



### `for`

```c
for(初值;条件;增量){ //适用于变量初始化和范围确定
	sentence;    
}

for(;;){          // infinite loop
    sentence;
}

for(i = 0, j = strlen(s)-1; i < j; i++, j--) //利用逗号复合语句
    c = s[i], s[i] = s[j], s[j] = c;

```

