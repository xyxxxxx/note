# 内存管理

```c
void *calloc(int num, int size);  //void * 表示指向未确定类型的指针,赋值时需要强制转型zhuanxing
//动态分配 num 个长度为 size 的连续空间，并将每一个字节都初始化为 0

void *malloc(size_t n);
//动态分配一块n字节大小的未初始化的内存空间，返回指向它的指针

void *realloc(void *address, int newsize);
//重新分配内存，把内存块扩展到 newsize。

void free(void *address);
//释放 address 所指向的内存块,释放的是动态分配的内存空间
```

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
 
int main()
{
   char name[100];              //given length str
   char *description;           //uncertain length str
 
   strcpy(name, "Zara Ali");
 
   description = (char *)malloc( 30 * sizeof(char) ); //allocate memory dynamically
   if( description == NULL )    //allocation failed
   {
      fprintf(stderr, "Error - unable to allocate required memory\n");
   }
   else
   {
      strcpy( description, "Zara ali a DPS student.");
   }
   
   description = (char *) realloc( description, 100 * sizeof(char) );
   if( description == NULL )
   {
      fprintf(stderr, "Error - unable to allocate required memory\n");
   }
   else
   {
      strcat( description, "She is in class 10th");
   }
   
   printf("Name = %s\n", name );
   printf("Description: %s\n", description );
 
   free(description);           //free memory
}
```

