## Format

**rule**

```
<target>: <prerequisites>
[Tab] <commands>
```

**target**

```
//target chain
result.txt: source.txt
	cp source.txt result.txt

source.txt:
	echo "new source" > source.txt
```

```
all: target1 target2 target3
```

**commands**

```
//replace Tab
.RECIPEPREFIX = >
all:
>echo "hello"
```

```
var-lost:
	export foo=bar                        //2 shell
	echo "foo=[$$foo]"

var-kept:
	export foo=bar; echo "foo=[$$foo]"    //1 shell
```



## Grammar

**comment**

```
# comment

@# comment not printed

@echo "hello"
```

**match**

```
//wildcard
*.c
?????.c

//pattern matching
%.o:%.c        // file1.o:file1.c, file2.o:file2.c, ...
```

**custom var**

```
txt = Hello
test:
	@echo $(txt)
```

**environment var**

```
test:
	@echo $$HOME
```

**implicit var**

```
$(CC)          //current compiler
$(MAKE)        //current Make tool
```

**automatic var**

```
$@             //current target
a.txt:
	touch $@

$(@D) $(@F)    //current target dir & name

$<             //first prerequisites
a.txt: b.txt c.txt
	cp $< $@   
	
$(<D) $(<F)	

$?             //prerequisites newer than target

$^             //all prerequisites

$*             // str % matches
```

**structure**

```makefile
ifeq ($(CC),gcc)
	libs=$(libs_for_gcc)
else
	libs=$(normal_libs)
endif
```

```
LIST = one two three
all:
	for i in $(LIST); do echo $(i); done???
	
```

## function

```
srcfiles := $(shell echo src/{00..99}.txt)
```

