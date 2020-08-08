# IO

## 流

```c
//标准输入流 stdin=0 对应文件standard input 通常连接键盘
//标准输出流 stdout=1 对应文件standard output 通常连接显示器
//标准错误流 stderr=2 对应文件standard error 通常连接显示器

//文件流
FILE *fopen(char *filename, char *mode);  //return NULL if failed
int fclose(FILE *fp);                     //return 0 if succeeded, 

FILE * fp;
fp = fopen ("file.txt", "w+");

```

| r    | 已有文件，只读         | r+   | 已有文件，读写       |
| ---- | ---------------------- | ---- | -------------------- |
| w    | 新文件，只写           | w+   | 新文件，读写         |
| a    | 已有或新文件，追加     | a+   | 已有或新文件，读追加 |
| ～b  | 字节文件(而非文本文件) |      |                      |

```c
//fflush将输出流的缓冲区的剩余数据全部写出
//remove删除指定文件
//rename重命名指定文件
//tmpfile返回一个临时文件的流
```



## IO function

**printf, scanf**

```C
//printf("<格式化字符串>", <参量表>);
//printf发送格式化输出到标准输出stdout
//scanf从标准输入stdin读取格式化输入
//scanf("%[a-z]",&str);读取字符集合

#include <stdio.h>
int main()
{
    float f;
    printf("Enter a number: ");
    scanf("%f",&f);            //args必须传入地址/指针
    printf("Value = %f", f);
    return 0;
}
```

```c
//fprintf发送格式化字符到流stream中
//fscanf从流stream读取格式化输入
int fprintf(FILE *stream, const char *format, ...)
int fscanf(FILE *stream, const char *format, ...)
```

|       |                                              |
| ----- | -------------------------------------------- |
| %d    | int                                          |
| %3d   | int with 3 char wide                         |
| %f    | double                                       |
| %6.2f | double with 6 char wide with 2 after decimal |
| %s    | str                                          |
| %c    | char                                         |
| %o    | oct                                          |
| %x    | hex                                          |

|      |                  |
| ---- | ---------------- |
| \n   | 换行             |
| \t   | Tab              |
| \r   | 回车（回到行首） |
| \b   | 退格             |
|      | 空格             |
| \\\  | \                |
|      |                  |
|      |                  |

**getchar, putchar**

```c
//getchar从标准输入stdin获取1个字符
//putchar将1个字符写入标准输出stdout

#include <stdio.h>
int main( )
{
   int c;                       //int类型以接收EOF
 
   printf( "Enter a value :");
   c = getchar( );              //get 1 char
 
   printf( "\nYou entered: ");
   putchar( c );                //put 1 char
   printf( "\n");
   return 0;
}
```

```c
int getc(FILE *fp)               //return EOF if end or err
int putc(int c, FILE *fp)        //return EOF if err
```



**gets, puts**

```c
//gets从标准输入stdin读入一行并存储在str指向的字符串
//puts将一个字符串写入标准输出stdout

#include <stdio.h>
int main( )
{
   char str[100];
 
   printf( "Enter a value :");
   gets( str );                  //read 1 line from stdin to buffer
 
   printf( "\nYou entered: ");
   puts( str );                  //write buffer to stdout
   return 0;
}
```

```c
//fgets从流stream中读入n-1个字符(或者读至行末)并存储到buffer
//fputs将字符串写入流stream中并换行
char *fgets( char *buf, int n, FILE *fp ); //
int fputs(char *s, FILE *fp);
```



**read, write**

```c
//fread从流stream中读取数据到指针指向的数组
//fwrite将指针指向数组的数据写入流stream

size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream)
size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream)
    

#include <stdio.h>
#include <string.h>
 
int main()
{
   FILE *fp;
   char c[] = "This is runoob";
   char buffer[20];
 
   /* 打开文件用于读写 */
   fp = fopen("file.txt", "w+");
 
   /* 写入数据到文件 */
   fwrite(c, strlen(c) + 1, 1, fp);
 
   /* 查找文件的开头 */
   fseek(fp, 0, SEEK_SET);
 
   /* 读取并显示数据 */
   fread(buffer, strlen(c)+1, 1, fp);
   printf("%s\n", buffer);
   fclose(fp);
   
   return(0);
}
```



**positioning**

```c
//fseek使位置指针从设定位置偏移
#include <stdio.h>
int main ()
{
   FILE *fp;

   fp = fopen("file.txt","w+");
   fputs("This is runoob.com", fp);
  
   fseek( fp, 7, SEEK_SET );
   //SEEK_SET 文件开头    SEEK_CUR 指针当前位置	SEEK_END 文件末尾
   fputs(" C Programming Langauge", fp);
   fclose(fp);
   
   return(0);
}
```

```c
//rewind(倒带)使位置指针回到文件头
//ftell返回指针当前位置
```



**error**

```c
//feof检测流是否已设定文件结束标识符EOF，返回0表示未设定
if( feof(fp) ){ 
    break ;
}

//ferror检测流是否已设定错误标识，返回0表示未设定
if( ferror(fp) )
{
    printf("读取文件：file.txt 时发生错误\n");
}
```

