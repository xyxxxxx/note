**版本管理**

```shell
$ cd /c/user/my_project		#切换目录

$ git init					#创建工作区
$ git clone git@github.com:FishpondA/testproj.git	#从远程版本库克隆至本地

$ git add <file>			#添加到暂存区(新文件或修改过的文件)

$ git commit -m "readme added"	#添加到版本库并提交说明
$ git commit --amend        #重新提交

$ git status				#查看是否存在未add或commit的修改

$ git diff <file> 			#查看未提交的文件修改

$ git log					#查看当前的版本历史
$ git log -p -2             #查看每次提交的更新内容,查看最近2次更新

$ git reset --hard HEAD^	#版本回滚
#HEAD表示当前版本,HEAD^表示上版本,HEAD~100表示之前第100个版本

$ git reset --hard b7df		#版本恢复,b7df...表示版本号,可以不写全

$ git reflog				#查看所有的版本历史

$ git checkout -- <file>	#撤销工作区的修改
#若暂存区有内容,则回滚至暂存区内容;否则回滚至版本库的最新版本

$ git reset HEAD <file>		#清除暂存区的内容

$ git rm <file>				#从版本中清除文件,作用同git add <file>

$ git mv <fname1> <fname2>  #文件更名
```



**远程仓库**

```shell
$ git remote -v				#查看远程库信息

$ git remote add origin git@github.com:FishpondA/testproj.git #设定远程版本库
$ git remote show origin    #查看远程版本库
$ git remote rename origin ori    #重命名远程版本库
$ git remote rm origin		#删除远程版本库

$ git push -u origin master		#本地master分支推送至远程,-u表示关联
$ git push origin master		#本地master分支推送至远程

$ git fetch <remote>        #抓取本地没有的数据

$ git pull					#抓取远程分支并合并到当前分支
$ git branch --set-upstream <branch> origin/<branch>#建立本地和远程分支的关联
```



**标签管理**

```shell
$ git tag v1.0	#打标签
$ git tag		#查看所有标签
$ git tag v0.9 <commit>	#对历史commit打标签
$ git tag -a v0.1 -m "version 0.1 released" <commit> #附注标签 -a标签名,-m说明文字

$ git show v0.9 #查看该标签的commit信息

$ git push origin <tagname>	#推送标签
$ git push origin --tags	#推送所有标签

$ git tag -d v0.1	#删除标签
$ git push origin :refs/tags/<tagname>	#删除远程标签
$ git push origin --delete <tagname>
```



**分支管理**

```shell
$ git branch <branch>	#创建分支
$ git switch <branch>	#切换分支
$ git switch -c <branch>	#创建分支并切换
$ git switch -c <branch> origin/<branch>	#在本地创建对应于远程分支的分支

$ git branch			#查看所有分支

$ git merge <branch>	#默认使用--ff
$ git merge --no-ff -m "merge with no-ff" <branch>
#区别见下图

$ git branch -d <branch>	#删除分支

$ git log --graph		#查看分支合并图

$ git stash				#保存当前工作
$ git stash list		#查看所有保存工作

$ git stash apply stash@{0}	#恢复指定保存工作
$ git stash drop stash@{0}	#删除指定保存工作
$ git stash pop 		#弹出最近保存工作

$ git cherry-pick <commit>	#复制特定提交

```

