# Git

## 命令

### add

添加指定文件或目录（递归地添加目录下的所有文件）（或其修改）到暂存区。

```shell
$ git add [file1] [file2] ...   # 添加指定文件到暂存区
$ git add [dir]                 # 添加指定目录到暂存区（递归地添加目录下的所有文件）
$ git add .                     # 添加当前目录到暂存区

$ git add data-*.py             # 添加当前目录下的所有data-*.py脚本
$ git add dir/\*.txt            # 添加dir/目录下的所有.txt文件（包括子目录下的文件）

# oh-my-zsh:
# ga: git add
```

### branch

列出、创建或删除分支。

```shell
$ git branch               # 列出本地分支，当前的分支以绿色突出显示，并标有星号
$ git branch -r            # 列出远程跟踪的分支
$ git branch -a            # 列出本地分支和远程跟踪的分支

# oh-my-zsh:
# gb:  git branch
# gba: git branch -a
# gbr: git branch -r

$ git branch [branch]      # 创建指定分支
$ git branch -d [branch]   # 删除指定分支，该分支必须完全合并到其上游分支或HEAD中
$ git branch -D [branch]   # 删除指定分支，即使master或任意其他分支都没有该分支的所有提交

# oh-my-zsh:
# gbd: git branch -d

$ git branch -m [oldbranch] [newbranch]   # 重命名指定分支
$ git branch -c [oldbranch] [newbranch]   # 复制指定分支
```

### checkout

切换分支或恢复工作树的文件。相当于 switch 和 restore 命令的组合。

```shell
$ git checkout [branch]         # 切换到指定分支
$ git checkout -b [branch]      # 创建指定分支并切换到该分支
$ git checkout --track [remote]/[branch]              # 创建跟踪指定远程分支的本地分支并切换到该分支
$ git checkout -b [branch] --track [remote]/[branch]  # 创建指定分支并切换到该分支，其跟踪指定远程分支

# oh-my-zsh:
# gco: git checkout
# gcb: git checkout -b
# gcm: git checkout $(git_main_branch)
```

```shell
$ git checkout [commit]         # 切换到指定提交，用于检查和可以丢弃的实验
                                # 此时HEAD处于detached状态，意味着HEAD指向一个具体的提交
                                # 而不是一个命名的分支
$ edit; git add; git commit     # 创建一个新的提交（称为a）
(1) $ git checkout [branch]     # 切换到其他分支，此时提交a不被任何分支或HEAD引用而被删除
(2) $ git checkout -b [branch]  # 创建指向提交a的新的分支，此时HEAD指向该分支，因而不再处于detached状态
(3) $ git checkout [branch]     # 创建指向提交a的新的分支，此时HEAD仍处于detached状态
(4) $ git tag [tag]             # 为提交a添加新的标签，此时HEAD仍处于detached状态
```

```shell
$ git checkout [commit] [file]  # 恢复指定提交的指定文件
$ git checkout [file]           # 恢复当前提交的指定文件
```

### cherry-pick


```shell

```

### clone

克隆一个仓库到一个新建的目录下，为仓库中的每个分支创建远程跟踪分支，创建一个初始分支并切换到该分支，其派生自仓库的当前活动分支。

```shell
$ git clone https://github.com/tensorflow/tensorflow.git
$ git clone https://github.com/tensorflow/tensorflow.git tf  # 克隆到目录tf下
```

### commit

提交暂存区的文件（或其修改）到本地仓库中。

```shell
$ git commit                    # 提交暂存区中的所有文件或目录
$ git commit [file] [dir] ...   # 提交暂存区中的指定文件或目录
$ git commit -a                 # 自动添加所有已知文件的修改到暂存区（即不会添加未跟踪的文件到暂存区）
$ git commit --amend            # 新的提交将替代最后一次提交

$ git commit -m "Infomation"    # 附加消息
$ git commit -C [commit]        # 使用指定提交的附加消息

# oh-my-zsh:
# gc:    git commit -v
# gc!:   git commit -v --amend
# gcam:  git commit -a -m
# gca!:  git commit -a -m
# gcmsg: git commit -m
```

### config

获取和设定当前仓库和全局配置。

#### -e, --edit

打开一个编辑器以修改指定的配置文件。

```shell
$ git config -e              # 修改仓库配置
$ git config -e --system     # 修改系统配置
$ git config -e --global     # 修改全局配置
```

#### --global

```shell
$ git config --global user.name "xyx"                    # 设定用户名
$ git config --global user.email "xyx@tensorstack.com"   # 设定email
```

#### -l, --list

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

### diff

比较暂存区快照与当前文件或上一次提交之间的差异。

```shell
$ git diff              # 比较暂存区快照和当前文件之间的差异

$ git diff --staged     # 比较暂存区快照和最后一次提交之间的差异
$ git diff --cached     # 同上

$ git diff --stat       # 显示修改的摘要而非详情
```

### fetch

从远程分支获取更新。

```shell
$ git fetch origin             # 下载远程仓库的数据到本地仓库（但不会修改当前工作）
$ git fetch origin master      # 下载远程仓库的master分支的数据到本地仓库（但不会修改当前工作）
```

### init

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

### log

展示提交记录。

```shell
# 内容
$ git log             # 展示过去所有的提交记录
$ git log -- [file]   # 展示对指定文件进行修改的提交记录
$ git log --follow [file]   # 展示对指定文件进行修改的提交记录，包含文件重命名之前的记录
$ git log [dir]       # 展示对指定目录中的文件进行修改的提交记录

# 比较
$ git log master --not --remotes=origin/master   # 展示所有本地master有但远程仓库origin的
                                                 # master分支没有的提交

# 样式
$ git log --graph     # 绘制树状图
$ git log --oneline   # 每个提交只打印一行信息
$ git log --summary   # 展示文件增删改的信息
$ git log --stat      # 展示文件增删改的统计数据
$ git log -p          # 展示提交与上一次提交的差异

# oh-my-zsh:
# glg:   git log --stat
# glgg:  git log --graph
# glgp:  git log --stat -p
# glo:   git log --oneline --decorate
# glog:  git log --oneline --decorate --graph


# 范围
$ git log -3                      # 展示最近3次提交记录
$ git log --since="2 weeks ago"   # 展示最近两周的提交记录
```

### merge

合并分支，即整合指定分支中的变化到当前分支。

```shell
$ git merge [branch]       # 将指定分支合并到master

# oh-my-zsh:
# gm:  git merge
# gma: git merge --abort
```

假设历史提交如下图所示，当前分支为 `master`。

```
      A---B---C topic
     /
D---E---F---G master
```

那么执行 `git merge topic` 将会将 `topic` 分支从 `master` 分叉（从 `E` 开始）之后做出的所有改变在 `master` 之上重新进行一遍，结果将被记录在一个新的提交中，一起记录的还包括两个父提交的名称以及用户描述其改变的日志信息。

```
      A---B---C topic
     /         \
D---E---F---G---H master
```

如果合并过程中发生了冲突，或者使用 `git merge --abort` 命令取消这次合并，或者解决冲突（见 [pull](#pull)）之后使用 `git merge --continue` 继续完成这次合并。

### pull

拉取远程分支，即整合指定远程分支中的变化到当前分支。如果当前分支落后于远程分支，则默认快进到匹配远程分支。如果当前分支和远程分支出现分叉，则用户需要指定如何来解决冲突。

更精确地说，`git pull` 首先执行 `git fetch`，然后根据设置选项或命令行参数，调用 `git rebase` 或 `git merge` 其中之一来协调分叉的分支。


```shell
$ git pull origin master                     # 拉取指定远程分支到当前分支
$ git pull master                            # 默认为唯一的远程仓库

# oh-my-zsh:
# ggl: git pull origin $(current_branch)
# ggu: git pull --rebase origin $(current_branch)
```

### push

将本地提交推送到远程仓库。

```shell
$ git push origin master                     # 推送指定分支到同名的远程分支
$ git push                                   # 默认为唯一的远程仓库和当前分支
$ git push origin c46e5:branch1              # 推送指定提交到远程分支
$ git push origin c46e5:refs/heads/branch1   # 推送指定提交到新建的远程分支

# oh-my-zsh:
# ggp: git push origin $(current_branch)

$ git push -f origin [branch]          # 当远程分支不是本地分支的祖先时，强制进行推送
                                       # 此操作可能导致远程分支丢失一些提交，请谨慎使用

# oh-my-zsh:
# ggf: git push --force origin $(current_branch)
```

### rebase

#### --abort

#### --continue

#### -i

### remote

管理跟踪的远程仓库。

```shell
$ git remote                         # 列出所有远程仓库
$ git remote add [remote] [url]      # 添加远程仓库
$ git remote rename [old] [new]      # 重命名远程仓库
$ git remote remove [remote]         # 移除远程仓库

$ git remote get-url [remote]        # 获取远程仓库的URL
$ git remote get-url -all            # 获取所有远程仓库的URL
$ git remote set-url [remote] [url]  # 设置远程仓库的URL
```

### reset

重置当前 HEAD 为指定状态。

```shell
$ git reset                # 移除暂存区中的所有文件
$ git reset -- [file]      # 移除暂存区中的指定文件

$ git reset --soft HEAD^       # 重置HEAD为倒数第二次提交，保持工作区不变，暂存所有修改
$ git reset -N HEAD^           # 重置HEAD为倒数第二次提交，保持工作区不变，不暂存所有修改
$ git reset --keep HEAD^       # 重置HEAD为倒数第二次提交，保留未暂存和已暂存的修改
$ git reset --hard HEAD~3      # 重置HEAD为倒数第四次提交，同时恢复工作区
$ git reset --hard             # 重置HEAD为最后一次提交，同时恢复工作区
$ git reset --merge ORIG_HEAD  # 重置HEAD为合并前的提交，保留未暂存的修改
```

### restore

恢复工作树的文件。

```shell
$ git restore [commit] [file]  # 恢复指定提交的指定文件
$ git restore [file]           # 恢复当前提交的指定文件
```

### rm

移除暂存区中的指定文件或目录（递归地移除目录下的所有文件）（或其修改）。

```shell
$ git rm [file1] [file2] ...   # 移除暂存区中的指定文件
$ git rm [dir]                 # 移除指定目录下所有暂存区中的文件或目录
$ git rm .                     # 移除当前目录下所有暂存区中的文件或目录

$ git add data-*.py             # 添加当前目录下的所有data-*.py脚本
$ git add dir/\*.txt            # 添加dir/目录下的所有.txt文件（包括子目录下的文件）

# oh-my-zsh:
# ga: git add
```

### show

展示各种类型的对象。

```shell
$ git show                       # 展示最后一次提交的信息
$ git show v1.0.0                # 展示指定标签的信息
$ git show master~2:README.md    # 展示master倒数第三次提交中的README.md文件的内容
```

### stash

贮藏当前工作区。

```shell
$ git stash          # 将当前相对于HEAD的所有本地修改保存为一个新的贮藏项
                     # 不区分未暂存和已暂存的修改，恢复时全部为未暂存的修改
$ git stash -m ""    # 命名
$ git stash --keep-index  # 贮藏未暂存和已暂存的修改，但保留暂存区

$ git stash list     # 列出所有的贮藏项
$ git stash show     # 将贮藏项中记录的变化作为diff展示
$ git stash pop      # 移除一个贮藏项并应用于当前的工作区
$ git stash apply    # 将一个贮藏项应用于当前的工作区
$ git stash drop     # 移除一个贮藏项
```

你在开发过程中得知远程分支已经更新，如果你的本地修改与远程分支的更新没有冲突，直接执行 `git pull` 即可同步本地分支。但如果你的本地修改与远程分支的更新存在冲突，执行 `git pull` 将会失败，此时你可以贮藏你的本地修改，执行 `git pull` 之后再恢复本地修改。

```shell
$ git pull
 ...
file foobar not up to date, cannot merge.
$ git stash
$ git pull
$ git stash pop
```

你在开发过程中接到命令要紧急修改一个 bug，此时你可以贮藏你的本地修改，切换到 master 修复 bug，然后再切换回来并恢复本地修改。

```shell
$ git stash
$ git checkout master
$ edit emergency fix
$ git commit -m "Fix in a hurry"
$ git checkout dev
$ git stash pop
```

### status

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

### switch

切换分支。

```shell
$ git switch [branch]        # 相当于 git checkout [branch]
$ git switch -c [branch]     # 相当于 git checkout -b [branch]
$ git switch -c [branch] [commit]  # 相当于 git checkout -b [branch] [commit]
$ git switch --detach [commit]     # 相当于 git checkout [commit]
```

### tag

创建、列出、删除或验证 GPG 签名的标签对象。

```shell
$ git tag                # 列出所有标签
$ git tag -l "v1.10.*"   # 列出匹配指定模型的标签

$ git tag -a v1.11 -m "my version 1.11"   # 创建附注标签
$ git tag -a v1.11 [commit]               # 为之前的提交创建附注标签
$ git show v1.11                          # 展示辅助标签信息

$ git tag v1.11-1        # 创建轻量标签
$ git show v1.11-1       # 展示轻量标签信息

$ git push origin v1.11   # 推送标签
$ git push origin --tags  # 推送所有标签

$ git tag -d v1.11-1               # 删除标签
$ git push origin --delete v1.11   # 删除远程标签
```

## .gitignore 文件

参考：

+ [Git 官方教程](https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository#_ignoring)
+ [Git 官方文档](https://git-scm.com/docs/gitignore)
+ [.gitignore 模版合集](https://github.com/github/gitignore)

## 使用场景

### 撤销合并或拉取

```shell
$ git branch
  dev
* master

$ git merge dev                     # 合并dev分支
# 或
$ git pull origin master            # 拉取远程master分支

$ git reset --merge ORIG_HEAD       # 撤销合并或拉取
                                    # 如果存在未暂存的修改，则其将保留
                                    # 如果存在暂存的修改，则其将丢失
# 如果同时存在未暂存和暂存的修改，则此命令出错。
#   此时如果不需要区分未暂存和暂存的修改，则直接使用`--keep`重置，否则贮藏暂存的修改后再使用`--hard`重置，   
#   参见 https://stackoverflow.com/questions/14759748/stashing-only-staged-changes-in-git-is-it-possible
#   如果能预见这一情况，更好的方法是在合并或拉取之前提交已暂存的修改，stash未暂存的修改
```

### 合并提交

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

### 拆分提交

```shell
$ git reset -N HEAD^                # 回退一次提交，保持工作区不变，所有修改为未暂存

$ git add ...
$ git commit -m "..."               # 分批提交

...

$ git push -f origin branch1        # 拆分之后推送到远程仓库时，需要附加`-f`选项
```

### 重新提交

```shell
$ git commit -m "Some messages."                # 提交

$ git commit --amend -m "Modified messages."    # 修改上次提交的消息
# 或
$ git add forgot_file
$ git commit --amend                            # 修改上次提交的内容
```

### 合并冲突

```shell
$ git merge dev                     # 合并dev分支
# 或
$ git pull origin master            # 拉取远程master分支
Auto-merging file
CONFLICT (content): Merge conflict in file
Automatic merge failed; fix conflicts and then commit the result.
```

```shell
# 使用任意文本编辑器编辑冲突的部分，保留一种修改或保留全部
<<<<<<< HEAD
333
=======
222
>>>>>>> dev
```

```shell
$ git add file                      # 编辑完毕之后添加此修改

$ git merge --continue              # 继续合并或拉取
# 或
$ git pull --continue
```

## 归档

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