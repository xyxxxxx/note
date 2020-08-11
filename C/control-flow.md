# 流程控制

## 条件结构

```C
if (条件) {语句组}

if (条件)
	{语句组1}
else if
	{语句组2}
else
	{语句组3}


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

```c
while (condition)
	{
    	sentence;
	}
-----
do
	{
    sentence;
	} while(condition)  
-----
for(初值;条件;增量) /*适用于变量范围确定*/
-----
break 跳出循环
continue 立即进行下一次循环
```

# 