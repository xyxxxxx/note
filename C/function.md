# 函数

## 定义函数

```c
int isprime(int n) /*函数的返回值类型，函数名，参数类型和名*/
{
    int i;
    for(i=2;i<=n/2;i++);
    	if(n%i==0)
            return 0;
    	return 1; /*函数返回值*/    
}

int age(int n)/*等差数列递归*/
{	if(n==1) return 10;
 	else	return age(n-1)+2;
}

double px(double x,int n)/*递归*/
{	if(n==1)
    	return x;
 	else
    	return x*(1-px(x,n-1));
}   

void int_turn(int n)
{if(n>=10)
	{printf("%d",n%10);
     int_turn(n/10);    
	}
 else
     printf("%d",n);
}  
```



