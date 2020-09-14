##  info

```shell
$ whatis <command>
$ info <command>		# detailed info
$ man <command>		    # manual

$ which <command>		# dir
$ whereis <command>
```



##  files management

```shell
# move
$ mv <file> <dir>
$ mv <file> <file>	    # rename
$ mv <dir> <dir>		# move or rename

# delete
$ rm <file>
$ rm -r <dir>			# delete dir

# copy
$ cp [] <file> <dir+file>	# copy file
$ cp -r <dir> <dir>	        # copy dir

# read
$ more -num <file>	    # show -num lines once
# space to pgdn, q to quit,
$ more +5 <file>		# read from line 5
$ tail -5 <file>	    # read last 5 lines
$ tail +5 <file>		# read from line 5
$ tail -f <file>		# read updates

# write
$ cat -n <file>		        # read txt with line numbers
$ cat -n <file1> > <file2>	# read & write
$ cat -n <file1> >> <file2>	# read & add

# create
$ touch <file>		# create new empty file

# access
$ chmod a+r <file>          # anyone can read
$ chmod ug+w,o-w <file>     # user,group can write, other group can not
$ chmod u=rwx,g=rx,o=x <file>
$ chmod 777 <file>

# compare
$ diff [] <file1><file2>	# compare files

# clear
$ :> <file>			# clear file contents


```



##  files operation

```shell
# sort file contents by line
$ sort [] <file>			# sort by ASCII
	-r reverse 
	-n by value

# link
$ ln -s <file1> <file2>	# symbolic link, like .lnk in Windows
$ ln <file1> <file2>	# hard link, means multiple names of single file

# compress
$ tar -cvf <file.tar>	<dir>		    # package
$ tar -zcvf <file.tar.gz> <dir>	        # package & compress
$ tar -xvf <file.tar>
$ tar -zxvf <file.tar.gz> [-C <dir>]	# decompress

$ xz -z <file.tar>                      # package
$ xz -d <file.tar.xz>                   # decompress
#      -k: keep origin file
```



##  dir

```shell
$ cd					# home
$ cd /			        # root
$ cd ~			        # home
$ cd -				    # last dir
$ cd <dir>			    # switch dir
$ cd ..			        # parent dir 
$ pwd					# current dir

$ mkdir <dir>			# create dir
$ rmdir <dir>			# delete dir
$ rm -rf <dir>          # delete non-empty dir

$ ls					# list every file
$ ls -lrt				# list with info
$ ls -lrt	s*			# list matching certain name
$ ls -a                 # list hidden

# search file in disk
$ find <dir> []		    # find file 
$ find -name "*.c"	    # by name
$ find -mtime -20		# revised in 20d
$ find -ctime +20		# created before 20d
$ find -type f		    # type f:normal
$ find -size +100c	    # size > 100 Bytes, c,k,M
$ find / -type f -size 0 -exec ls -l {} \;	# find every size 0 normal file in disk, show path
```



##  text

```shell
# search text
$ egrep <str> <file/dir> [-A1 -B1]	# search in file/dir by str
$ egrep <re> <file/dir>	# search in file/dir by regular expression
```



##  logic

```shell
#  && command1 succeed, then command2
$ cp sql.txt sql.bak.txt && cat sql.bak.txt


#  || command1 fail, then command2
$ ls /proc && echo  success || echo failed

#  (;) 

#  | command1 output as command2 input
$ ls -l /etc | more
$ echo "Hello World" | cat > hello.txt
$ ps -ef | grep <str>

```



##  Shell

```shell
$ echo <str>			# output str

$ clear                # clear shell
$ exit                 # exit shell

# xargs
$ cat test.txt | xargs	# output single line
$ cat test.txt | xargs -n3	# output by every line 3 words
$ echo "nameXnameXnameXname" | xargs -dX	# output by splitting by X

```



##  net

```shell
# show hostname
$ hostname
# revise at /etc/sysconfig/network

# ipconfig
$ ifconfig
# revise at /etc/sysconfig/network-scripts/ifcfg-<name>

# network restart
$ nmcli c reload

# request server
$ curl https://www.example.com          # send GET request
$ curl -X POST https://www.example.com  # assign request method
$ curl -i https://www.example.com       # print header of response packet
```



##  system management

```shell
$ sudo                # run as root
$ sudo -u             # run as user

$ ps -ef			  # show all processes
$ ps -aux

$ top                 # show real-time processes, equal task manager on Windows
$ top -d 2            # update every 2s
$ top -p <num>        # show designated process

$ kill <num>		  # kill process
$ kill -9 <num>	      # kill process compulsorily

$ date                # print current time

$ shutdown
$ reboot

```



##  system setting

```shell
# redhat package manager
$ rpm -a				  # show packages
$ rpm -qa | grep <name>	  # search package with name
$ rpm -e --nodeps <name>  # uninstall package

# environment variable
$ env                     # show environment variables
$ export -p               # show environment variables
$ export var=10           # define environment variable, assign
$ echo $ENV               # print environment variable

# shortcut command
$ alias <alias>=<command>

```



##  gcc

```shell
$ gcc -E hello.c -o hello.i    # Preprocess
$ gcc -S hello.c -o hello.s    # Compile
$ gcc -c hello.c -o hello.o    # Assemble
$ gcc hello.c -o hello.exe     # Link
```

