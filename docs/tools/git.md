[toc]



# 命令

## add

添加指定文件或目录（递归地添加目录下的所有文件）（或其修改）到暂存区。

```shell
$ git add [file1] [file2] ...   # 添加指定文件到暂存区
$ git add [dir]                 # 添加指定目录到暂存区（递归地添加目录下的所有文件）
$ git add .                     # 添加当前目录到暂存区

# oh-my-zsh:
# git add -> ga
```



## branch





## checkout



## cherry-pick



## clone

克隆一个仓库到一个新建的目录下，为仓库中的每个分支创建远程跟踪分支，创建一个初始分支并切换到该分支，其派生自仓库的当前活动分支。

```shell
$ git clone https://github.com/tensorflow/tensorflow.git

$ git branch --remotes             # 查看远程跟踪分支
```



## commit

提交暂存区的文件（或其修改）到本地仓库中。





## config

获取和设定当前仓库和全局配置。



### -e, --edit

打开一个编辑器以修改指定的配置文件。

```shell
$ git config -e              # 修改仓库配置
$ git config -e --system     # 修改系统配置
$ git config -e --global     # 修改全局配置
```



### --global





### -l, --list

获取配置文件中所有变量和它们的值。

```shell
$ git config -l 
credential.helper=osxkeychain
user.name=xyx
user.email=xyx@tensorstack.com
core.repositoryformatversion=0
core.filemode=true
core.bare=false
core.logallrefupdates=true
core.ignorecase=true
core.precomposeunicode=true
```





## diff

比较暂存区快照与当前文件或上一次提交之间的差异。

```shell
$ git diff              # 比较暂存区快照和当前文件之间的差异

$ git diff --staged     # 比较暂存区快照和最后一次提交之间的差异
$ git diff --cached     # 同上

$ git diff --stat       # 显示修改的摘要而非详情
```



## fetch





## init

在当前目录下初始化一个空的Git仓库。

```shell
# 为现有代码库创建新的Git仓库
$ cd /path/to/my/codebase
$ git init                           # 创建/path/to/my/codebase/.git文件
Initialized empty Git repository in /path/to/my/codebase/.git/
$ git add .                          # 将所有文件添加到索引
$ git commit -m "First commit."      # 将原始状态记录为历史记录中的第一次提交
$ git branch
* master                             # 初始分支默认名称为`master`
```



## log



## merge



## mv



## pull



## push



```shell
$ git push origin c46e5:branch1              # 推送指定提交到远程分支
$ git push origin c46e5:refs/heads/branch1   # 推送指定提交到新建的远程分支

```







### -f



## rebase



### --abort



### --continue



### -i







## remote



## reset



## restore



## rm



## show



## status

检查当前的文件状态。

![lifecycle](https://git-scm.com/book/en/v2/images/lifecycle.png)

```shell
$ git status         
On branch master
Your branch is based on 'origin/master', but the upstream is gone.
  (use "git branch --unset-upstream" to fixup)

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        modified:   folder1/file1.md        # 已添加到暂存区的修改，使用`commit`命令以提交
        new file:   folder1/file2.md        # 已添加到暂存区的新文件，使用`commit`命令以提交

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   folder1/file1.md        # 未添加到暂存区的修改。可以看到同一文件可以同时有已暂存和未暂存的修改

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        folder1/file3.md                    # 未追踪的新文件

$ git status -s              # 简化输出
MM folder1/file1.md          # `M`表示修改，第一列对应暂存区，第二列对应工作区
A  folder1/file2.md          # `A`表示已添加到暂存区但未提交的新文件
?? folder1/file3.md          # `??`表示未追踪的新文件
```



## tag







# .gitignore 文件

参考：

+ [Git 官方教程](https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository#_ignoring)
+ [Git 官方文档](https://git-scm.com/docs/gitignore)
+ [.gitignore 模版合集](https://github.com/github/gitignore)





# 使用场景

## 合并指定提交之后的几次提交

```shell
$ git rebase -i [commit]            # 修改指定提交之后的所有提交

pick 4dac89b Update file1           # 下方的提交是新的提交，将新的提交的行首命令修改为`s`
s 7d1dbd3 Update file1
s d3bbb50 Update file1              # 保存并退出

Update file1                        # 下方的提交是新的提交，将不需要的消息添加注释
# Update file1
# Update file1                      # 保存并退出

$ git push -f origin branch1        # 合并之后推送到远程仓库时，需要附加`-f`选项
```



## 修改上次提交的消息

```shell
$ git commit -m "Some messages."                # 提交

$ git commit --amend -m "Modified messages."    # 修改上次提交的消息
```











**新建**

```shell
# 在当前目录新建一个Git代码库
$ git init

# 新建一个目录，将其初始化为Git代码库
$ git init [project-name]

# 下载一个项目和它的整个代码历史
$ git clone [url]

# 下载一个项目的指定分支
$ git clone -b [branch-name] [url]
```



**配置**

```shell
# 显示当前的Git配置
$ git config --list

# 编辑Git配置文件
$ git config -e [--global]

# 设置提交代码时的用户信息
$ git config [--global] user.name "[name]"
$ git config [--global] user.email "[email address]"
```



**增加/删除文件**

```shell
# 添加指定文件到暂存区
$ git add [file1] [file2] ...

# 添加指定目录到暂存区，包括子目录
$ git add [dir]

# 添加当前目录的所有文件到暂存区
$ git add .

# 添加每个变化前，都会要求确认
# 对于同一个文件的多处变化，可以实现分次提交
$ git add -p

# 删除工作区文件，并且将这次删除放入暂存区
$ git rm [file1] [file2] ...

# 停止追踪指定文件，但该文件会保留在工作区
$ git rm --cached [file]

# 改名文件，并且将这个改名放入暂存区
$ git mv [file-original] [file-renamed]
```



**代码提交**

```shell
# 提交暂存区到仓库区
$ git commit -m [message]

# 提交暂存区的指定文件到仓库区
$ git commit [file1] [file2] ... -m [message]

# 提交工作区自上次commit之后的变化，直接到仓库区
$ git commit -a

# 提交时显示所有diff信息
$ git commit -v

# 使用一次新的commit，替代上一次提交
# 如果代码没有任何新变化，则用来改写上一次commit的提交信息
$ git commit --amend -m [message]

# 重做上一次commit，并包括指定文件的新变化
$ git commit --amend [file1] [file2] ...
```



**分支管理**

```shell
# 列出所有本地分支
$ git branch

# 列出所有远程分支
$ git branch -r

# 列出所有本地分支和远程分支
$ git branch -a

# 新建一个分支，但依然停留在当前分支
$ git branch [branch-name]

# 新建一个分支，并切换到该分支
$ git checkout -b [branch]

# 新建一个分支，基于指定commit
$ git branch [branch] [commit]

# 新建一个分支，与指定的远程分支建立追踪关系
$ git branch --track [branch] [remote-branch]

# 切换到指定分支，并更新工作区
$ git checkout [branch-name]

# 切换到上一个分支
$ git checkout -

# 建立追踪关系，在现有分支与指定的远程分支之间
$ git branch --set-upstream [branch] [remote-branch]

# 合并指定分支到当前分支
$ git merge [branch]

# 将当前分支的所有commit剪断并嫁接在指定分支上
$ git rebase [branch]

# 选择一个commit，作为当前分支的新commit
$ git cherry-pick [commit]

# 删除分支
$ git branch -d [branch-name]

# 删除远程分支
$ git push origin --delete [branch-name]
$ git branch -dr [remote/branch]
```



**标签管理**

```shell
# 列出所有tag
$ git tag

# 新建一个tag在当前commit
$ git tag [tag]

# 新建一个tag在指定commit
$ git tag [tag] [commit]

# 删除本地tag
$ git tag -d [tag]

# 删除远程tag
$ git push origin :refs/tags/[tagName]

# 查看tag信息
$ git show [tag]

# 提交指定tag
$ git push [remote] [tag]

# 提交所有tag
$ git push [remote] --tags

# 新建一个分支，指向某个tag
$ git checkout -b [branch] [tag]
```



**查看信息**

```shell
# 显示有变更的文件
$ git status

# 显示当前分支的版本历史
$ git log

# 显示commit历史，以及每次commit发生变更的文件
$ git log --stat

# 搜索提交历史，根据关键词
$ git log -S [keyword]

# 显示某个commit之后的所有变动，每个commit占据一行
$ git log [tag] HEAD --pretty=format:%s

# 显示某个commit之后的所有变动，其"提交说明"必须符合搜索条件
$ git log [tag] HEAD --grep feature

# 显示某个文件的版本历史，包括文件改名
$ git log --follow [file]		
$ git whatchanged [file]

# 显示指定文件相关的每一次diff
$ git log -p [file]

# 显示过去5次提交
$ git log -5 --pretty --oneline

# 查看分支合并图
$ git log --graph

# 显示所有提交过的用户，按提交次数排序
$ git shortlog -sn

# 显示指定文件是什么人在什么时间修改过
$ git blame [file]

# 显示暂存区和工作区的差异
$ git diff

# 显示暂存区和上一个commit的差异
$ git diff --cached [file]

# 显示工作区与当前分支最新commit之间的差异
$ git diff HEAD

# 显示两次提交之间的差异
$ git diff [first-branch]...[second-branch]

# 显示今天你写了多少行代码
$ git diff --shortstat "@{0 day ago}"

# 显示某次提交的元数据和内容变化
$ git show [commit]

# 显示某次提交发生变化的文件
$ git show --name-only [commit]

# 显示某次提交时，某个文件的内容
$ git show [commit]:[filename]

# 显示当前分支的最近几次提交
$ git reflog
```



**远程同步**

```shell
# 下载远程仓库的所有变动
$ git fetch [remote]

# 显示所有远程仓库
$ git remote -v

# 显示某个远程仓库的信息
$ git remote show [remote]

# 增加一个新的远程仓库，并命名
$ git remote add [name] [url]

# 重命名远程版本库
$ git remote rename [name0] [name1]

# 删除远程版本库
$ git remote rm [name]

# 取回远程仓库的变化，并与本地分支合并
$ git pull [remote] [branch]

# 上传本地指定分支到远程仓库
$ git push [remote] [branch]

# 强制推送当前分支到远程仓库，即使有冲突(例如在rebase之后)
$ git push [remote] --force

# 推送所有分支到远程仓库
$ git push [remote] --all
```



**撤销**

```shell
# 恢复暂存区的指定文件到工作区
$ git checkout [file]

# 恢复某个commit的指定文件到暂存区和工作区
$ git checkout [commit] [file]

# 恢复暂存区的所有文件到工作区
$ git checkout .

# 重置暂存区的指定文件，与上一次commit保持一致，但工作区不变
$ git reset [file]

# 重置暂存区与工作区，与上一次commit保持一致
$ git reset --hard

# 重置当前分支的指针为指定commit，同时重置暂存区，但工作区不变
$ git reset [commit]

# 重置当前分支的HEAD为指定commit，同时重置暂存区和工作区，与指定commit一致
$ git reset --hard [commit]

# 重置当前HEAD为指定commit，但保持暂存区和工作区不变
$ git reset --keep [commit]

# 新建一个commit，用来撤销指定commit
# 后者的所有变化都将被前者抵消，并且应用到当前分支
$ git revert [commit]

# 将未提交的变化压栈
$ git stash

# 查看栈中所有保存的工作
$ git stash list

# 弹出最近压栈的工作
$ git stash pop

# 恢复栈中指定工作
$ git stash apply stash@{0}	

# 删除栈中指定工作
$ git stash drop stash@{0}
```