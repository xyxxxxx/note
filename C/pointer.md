# 指针

## 指针的类型

```c
int *pa; /*定义指针*/
pa=&a;
//or 
int *pa=&a; 
-----
int a[5]={1,2,3,4,5}; int *pa=a/*指针指向数组*/
a[i]==*(pa+i)==*(a+i);/*pa和a都是数组a的起始地址即a[0],+i表示其后i个地址,*表示取该地址的存储数据*/
-----
int a[5]={1,2,3,4,5}; int *p1, *p2;
p1=&a[0]; p2=&a[4];   //p2-p1==4
-----
char *a="Hello world";/*字符指针变量*/
/*相当于有s[]="Hello world",然后a指向字符串s的起始位置*/
-----
-----
-----    
int a[3][4]={1,2,3,4,5,6,7,8,9,10,11,12}; /*指针指向二维数组*/
/* a,a[0],&a[0][0],*a(==a[0])都表示a的起始地址*/
/* a+i,a[i],&a[i][0],*(a+i)表示a第i+1行的起始地址*/
/* a[i]+j是a[i]的第j个元素地址,即&a[i][j]*/
int *p;
p=a;
-----
int a[3][4]={1,2,3,4,5,6,7,8,9,10,11,12}; /*行指针*/
int (*p)[4]; /*4为长度*/
/* p=a时,其指向一维数组a[0]*/
/* p+i指向数组a[i],因此*(p+i)+j即a[i]+j */
-----
-----
-----    
/*指向函数的指针*/
---
float *search(float (*pointer)[4],int n) /*指针型函数,返回一个地址*/
-----
char name[5][10]={"Alice","Bob","Cindy","Dave","Elizabeth"};
char *p[5]={name[0],name[1],name[2],name[3],name[4]};/*指针数组,由5个指针变量组成*/ 
-----
int *p1,**p2;/*二级指针,p2保存p1的地址*/
```

## 指针的举例说明

```c
int x=1;
char a="A";//
int y[3]={1,2,3}; 
char b[5]="Alice";//数组
int z[3][3]={1,2,3,4,5,6,7,8,9};
char c[5][10]={"Alice","Bob","Cindy","Dave","Elizabeth"};//二维数组
int *px=&x; 
char *pa=&a;//
int *py=y=&y[0];
char *pb=b=&b[0];//数组的指针
int *pz=z=z[0]=&z[0][0]=*z;//*z=z[0]
char *pc=c=c[0]=&c[0][0];//二维数组的指针
int (*pzl)[3]=z;
char (*pcl)[10]=c; //二维数组的行指针
int *pzm[3]={z[0],z[1],z[2]};
char *pcm[5]={c[0],c[1],c[2],c[3],c[4]};//二维数组的指针数组
int **pp=&pzm//二级指针

| address | name    | value |
| 0000    | x       | 1     |
| 0010    | a       | A     |
| 0100    | y[0]    | 1     |
| 0101    | y[1]    | 2     |
| 0102    | y[2]    | 3     |
| 0200    | b[0]    | A     |
| 0201    | b[1]    | l     |
| 0202    | b[2]    | i     |
| 0203    | b[3]    | c     |
| 0204    | b[4]    | e     |
| 0205    | b[5]    | \0    |
| 1000    | z[0][0] | 1     |
| 1001    | z[0][1] | 2     |
| 1002    | z[0][2] | 3     |
| 1003    | z[1][0] | 4     |
| 1004    | z[1][1] | 5     |
| 1005    | z[1][2] | 6     |
| 1006    | z[2][0] | 7     |
| 1007    | z[2][1] | 8     |
| 1008    | z[2][2] | 9     |
| 2000    | c[0][0] | A     |
| 2001    | c[0][1] | l     |
| 2002    | c[0][2] | i     |
| 2003    | c[0][3] | c     |
| 2004    | c[0][4] | e     |
| 2005    | c[0][5] | \0    |
...
| 2009    | c[0][9] | \0    |
| 2010    | c[1][0] | B     |
...
| 2020    | c[2][0] | C     |
...
| 2030    | c[3][0] | D     |
...
| 2040    | c[4][0] | E     |
...
| 2049    | c[0][9] | \0    |
| 8000    | px      | 0000  |
| 8010    | pa      | 0010  |
| 8100    | py      | 0100  |
|         | py+1    | 0101  |    
| 8200    | pb      | 0200  |    
| 9000    | pz      | 1000  |
|         | pz+1    | 1001  |
| 9100    | pc      | 2000  |
|         | pc+1    | 2001  |
| 9200    | pzl     | 1000~1002 |
|         | pzl+1   | 1003~1005 |
|         | *(pzl+1)+1 | 1004   |// *(pzm+1)=z[1]
| 9300    | pcl     | 2000~2009 |
|         | pcl+1   | 2010~2019 |
|         | *(pcl+1)+1 | 2011   |
| 9400    | pzm[0]  | 1000  |    
| 9401    | pzm[1]  | 1003  |        
| 9402    | pzm[2]  | 1006  |            
| 9500    | pcm[0]  | 2000  |    
| 9501    | pcm[1]  | 2010  |        
| 9502    | pcm[2]  | 2020  |    
| 9503    | pcm[3]  | 2030  |    
| 9504    | pcm[4]  | 2040  |
| 10000   | pp      | 9400  |
|         | pp+1    | 9401  |    

    
```

地址+地址的运算是非法的

