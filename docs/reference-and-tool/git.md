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

$ git add -p      # 添加每个变化前，都会要求确认
                  # 对于同一个文件的多处变化，可以实现分次提交

# oh-my-zsh:
# ga: git add
```

### blame

显示指定文件每一行的最后一次修改的提交。

```shell
$ git blame [file]
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

$ git branch [branch]           # 创建分支
$ git branch [branch] [commit]  # 基于指定提交创建分支
$ git branch -d [branch]        # 删除指定分支，该分支必须完全合并到其上游分支或HEAD中
$ git branch -D [branch]        # 删除指定分支，即使master或任意其他分支都没有该分支的所有提交

# oh-my-zsh:
# gbd: git branch -d

$ git branch -m [oldbranch] [newbranch]   # 重命名指定分支
$ git branch -c [oldbranch] [newbranch]   # 复制指定分支

$ git branch --set-upstream [branch] [remote-branch]  # 建立指定分支与指定远程分支之间的追踪关系
```

### checkout

切换分支或恢复工作树的文件。相当于 switch 和 restore 命令的组合。

```shell
$ git checkout [branch]         # 切换到指定分支
$ git checkout -                # 切换到上一个分支
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
$ git checkout [commit] [file]  # 恢复指定提交的指定文件到工作区和暂存区
$ git checkout [file]           # 恢复暂存区的指定文件到工作区
$ git checkout .                # 恢复暂存区的所有文件到工作区
```

### cherry-pick

应用指定提交的修改，相应地产生新的提交。

```shell
$ git cherry-pick [commit]              # 应用指定提交的修改到HEAD，并创建一个新的提交
$ git cherry-pick [branch]              # 应用指定分支的最后一次提交的修改到HEAD，并创建一个新的提交
$ git cherry-pick master~4 master~2     # 应用master分支的倒数第五和第三次提交的修改到HEAD，并创建两个新的提交
$ git cherry-pick [commit1]..[commit2]  # 应用commit1到commit2之间的所有提交的修改（不包括commit1，包括commit2）
$ git cherry-pick [branch1] [branch2] ^[branch3]  # 应用所有既属于branch1、又属于branch2、又不属于branch3的提交的修改 

# oh-my-zsh:
# gcp:   git cherry-pick
```

如果 cherry-pick 过程中发生了合并冲突，或者使用 `git cherry-pick --abort` 命令取消这次 cherry-pick，或者解决冲突之后使用 `git cherry-pick --continue` 继续完成这次 cherry-pick（见 [合并冲突](#合并冲突)）。

### clone

克隆一个仓库到一个新建的目录下，为仓库中的每个分支创建远程跟踪分支，创建一个初始分支并切换到该分支，其派生自仓库的当前活动分支。

```shell
$ git clone https://github.com/tensorflow/tensorflow.git
$ git clone https://github.com/tensorflow/tensorflow.git tf  # 克隆到目录tf下
$ git clone -b [branch] [url]                                # 克隆指定分支
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

$ git commit -v                 # 提交时显示所有diff信息

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
$ git config --global http.proxy http://us01.proxy.net   # 设定代理服务器
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

比较工作区、暂存区或指定提交之间的指定文件的差异。

```shell
$ git diff                       # 比较工作区和暂存区之间的差异
$ git diff [commit]              # 比较工作区和指定提交之间的差异
$ git diff [commit1] [commit2]   # 比较指定两次提交之间的差异
$ git diff [file]                # 比较指定文件的差异

$ git diff --staged     # 比较暂存区快照和最后一次提交之间的差异
$ git diff --cached     # 同上

$ git diff --stat       # 显示修改的摘要而非详情

# oh-my-zsh:
# gd:  git diff
# gds: git diff --staged
```

### fetch

从远程分支获取更新。

```shell
$ git fetch origin             # 下载远程仓库的数据到本地仓库（但不会修改当前工作）
$ git fetch origin master      # 下载远程仓库的master分支的数据到本地仓库（但不会修改当前工作）

# oh-my-zsh:
# gf:  git fetch
# gfo: git fetch origin
```

### init

初始化一个新的 Git 仓库。

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

```shell
# 新建一个目录并将其初始化为Git仓库
$ git init [repo]
```

### log

展示提交记录。

```shell
# 内容
$ git log             # 展示过去所有的提交记录
$ git log [commit]    # 展示指定提交及其之前的提交记录
$ git log [commit]..  # 展示指定提交之后的提交记录
$ git log -- [file]   # 展示对指定文件进行修改的提交记录
$ git log --follow [file]   # 展示对指定文件进行修改的提交记录，包含文件重命名
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

# 搜索
$ git log -S [keyword]            # 根据关键词搜索提交记录

# 用户

$ git shortlog -sn    # 展示所有提交过的用户，按提交次数排序
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

如果合并过程中发生了冲突，或者使用 `git merge --abort` 命令取消这次合并，或者解决冲突之后使用 `git merge --continue` 继续完成这次合并（见 [合并冲突](#合并冲突)）。

### mv

重命名指定文件或目录。

```shell
$ git mv [old] [new]   # 重命名工作区中的指定文件或目录，并将这次重命名放入暂存区
```


### pull

拉取远程分支，即整合指定远程分支中的变化到当前分支。如果当前分支落后于远程分支，则默认快进到匹配远程分支。如果当前分支和远程分支出现分叉，则用户需要指定如何来解决冲突（见 [合并冲突](#合并冲突)）。

更精确地说，`git pull` 首先执行 `git fetch`，然后根据设置选项或命令行参数，调用 `git rebase` 或 `git merge` 其中之一来协调分叉的分支。


```shell
$ git pull origin master                     # 拉取指定远程分支到当前分支
$ git pull master                            # 默认为唯一的远程仓库

# oh-my-zsh:
# gl:  git pull
# ggl: git pull origin $(current_branch)
# ggu: git pull --rebase origin $(current_branch)
```

### push

将本地提交推送到远程仓库。

```shell
$ git push origin master                     # 推送指定分支到同名的远程分支
$ git push                                   # 默认为唯一的远程仓库和当前分支
$ git push -all                              # 推送所有本地分支
$ git push origin c46e5:branch1              # 推送指定提交到远程分支
$ git push origin c46e5:refs/heads/branch1   # 推送指定提交到新建的远程分支

# oh-my-zsh:
# gp:  git push
# ggp: git push origin $(current_branch)

$ git push -f origin [branch]          # 当远程分支不是本地分支的祖先时，强制进行推送
                                       # 此操作可能导致远程分支丢失一些提交，请谨慎使用

# oh-my-zsh:
# ggf: git push --force origin $(current_branch)

$ git push origin --delete [branch]    # 删除远程分支
```

### rebase

移植（嫁接）指定提交。

```shell
$ git rebase [branch1] [branch2]       # 将所有属于branch1但不属于branch2的提交移植到branch2
                                       # 若没有指定branch2，则将当前分支作为branch2
$ git rebase --onto [branch3] [branch1] [branch2]    # 将所有属于branch1但不属于branch2的
                                                     # 提交移植到branch3
$ git rebase -i                        # 使用交互模式

# oh-my-zsh:
# grb:  git rebase
# grbi: git rebase -i
# grba: git rebase --abort
# grbc: git rebase --continue
```

如果 rebase 过程中发生了合并冲突，或者使用 `git rebase --abort` 命令取消这次 rebase，或者解决冲突之后使用 `git rebase --continue` 继续完成这次 rebase（见 [合并冲突](#合并冲突)）。

```shell
#       A---B---C topic
#      /
# D---E---F---G master

$ git rebase master topic   # 如果当前分支为topic，则可以省略topic

#               A'--B'--C' topic
#              /
# D---E---F---G master
```

```shell
#       A---B---C topic
#      /
# D---E---A'---F master

$ git rebase master topic   # 如果master分支已经应用过与A（或B、C）相同的提交，则
                            # 跳过（不再应用）这一提交并发出警告

#                B'---C' topic
#               /
# D---E---A'---F master
```

```shell
#                         H---I---J topicB
#                        /
#               E---F---G  topicA
#              /
# A---B---C---D  master

$ git rebase --onto master topicA topicB

#              H'--I'--J'  topicB
#             /
#             | E---F---G  topicA
#             |/
# A---B---C---D  master
```

```shell
# E---F---G---H---I---J  topicA

$ git rebase --onto topicA~5 topicA~3 topicA

# E---H'---I'---J'  topicA     提交F和G不被任何分支或HEAD引用而被删除
```

```shell
$ git rebase -i [commit]

pick 02a2c46 Update file2 to line 3
pick ad149fd Update file2 to line 4
pick 42d611e Update file2 to line 5

# Rebase d178804..42d611e onto d178804 (3 commands)
#
# Commands:
# p, pick <commit> = use commit
# r, reword <commit> = use commit, but edit the commit message
# e, edit <commit> = use commit, but stop for amending
# s, squash <commit> = use commit, but meld into previous commit
# f, fixup <commit> = like "squash", but discard this commit's log message
# x, exec <command> = run command (the rest of the line) using shell
# b, break = stop here (continue rebase later with 'git rebase --continue')
# d, drop <commit> = remove commit
# l, label <label> = label current HEAD with a name
# t, reset <label> = reset HEAD to a label
# m, merge [-C <commit> | -c <commit>] <label> [# <oneline>]
# .       create a merge commit using the original merge commit's
# .       message (or the oneline, if no original merge commit was
# .       specified). Use -c <commit> to reword the commit message.
#
# These lines can be re-ordered; they are executed from top to bottom.
#
# If you remove a line here THAT COMMIT WILL BE LOST.
#
# However, if you remove everything, the rebase will be aborted.
#
# Note that empty commits are commented out
```

### remote

管理跟踪的远程仓库。

```shell
$ git remote                         # 列出所有远程仓库
$ git remote show [remote]           # 展示指定远程仓库的信息
$ git remote add [remote] [url]      # 添加远程仓库
$ git remote rename [old] [new]      # 重命名远程仓库
$ git remote remove [remote]         # 移除远程仓库

$ git remote get-url [remote]        # 获取远程仓库的URL
$ git remote get-url -all            # 获取所有远程仓库的URL
$ git remote set-url [remote] [url]  # 设置远程仓库的URL

# oh-my-zsh:
# gr:    git remote
# gra:   git remote add
# grmv:  git remote rename
# grrm:  git remote remove
# grset: git remote set-url
```

### reset

重置当前 HEAD 为指定状态。

```shell
$ git reset (--mixed)          # 保留工作区，清除暂存区（即添加的修改回到工作区）；撤销的所有提交的修改进入工作区
$ git reset --soft             # 保留工作区和暂存区；撤销的所有提交的修改进入暂存区
$ git reset --keep             # 保留工作区和暂存区；撤销的所有提交的修改被丢弃；
                               # 如果保留的修改与丢弃的修改存在冲突（即存在于同一文件中），则此命令出错
$ git reset --hard             # 清除工作区和暂存区；撤销的所有提交的修改被丢弃
$ git reset --merge ORIG_HEAD  # 重置HEAD为合并前的提交，保留工作区

$ git reset [commit]           # 重置HEAD为指定提交，若不指定则为最后一次提交
$ git reset -- [file]          # 仅重置指定文件


# oh-my-zsh:
# grh:  git reset
# gru:  git reset --
# grhh: git reset --hard
```

### restore

恢复工作树的文件。

```shell
$ git restore [commit] [file]  # 恢复指定提交的指定文件
$ git restore [file]           # 恢复当前提交的指定文件
```

### revert

撤销指定提交。

```shell
$ git revert [commit]   # 创建一个用于抵消指定提交的提交并应用到HEAD
```

### rm

删除指定文件或目录（递归地删除目录下的所有文件）。

```shell
$ git rm [file1] [file2] ...   # 删除工作区中的指定文件，并将这次删除放入暂存区
$ git rm [dir]                 # 删除工作区中的指定目录，并将这次删除放入暂存区
$ git rm --cached [file]       # 停止追踪指定文件，但该文件会保留在工作区
```

### show

展示各种类型的对象。

```shell
$ git show                       # 展示最后一次提交的信息
$ git show [commit]              # 展示指定提交的信息
$ git show --name-only           # 仅展示被修改的文件的名称
$ git show [commit]:[file]       # 展示指定提交中的指定文件的内容

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

$ git stash list            # 列出所有的贮藏项
$ git stash show [stash]    # 将贮藏项中记录的变化作为diff展示，默认使用最后一个贮藏项
$ git stash pop [stash]     # 移除一个贮藏项并应用于当前的工作区
$ git stash apply [stash]   # 将一个贮藏项应用于当前的工作区
$ git stash drop [stash]    # 移除一个贮藏项
$ git stash clear           # 清空所有贮藏项

# oh-my-zsh:
# gsta:  git stash push
# gstaa: git stash apply
# gstd : git stash drop
# gstl : git stash list
# gstp : git stash pop
# gstc : git stash clear
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

# oh-my-zsh:
# gst: git status
# gss: git status -s
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

$ git tag v1.11-1                         # 创建轻量标签
$ git tag v1.11-1 [commit]                # 为之前的提交创建轻量标签
$ git show v1.11-1                        # 展示轻量标签信息

$ git push origin v1.11            # 推送标签
$ git push origin --tags           # 推送所有标签

$ git tag -d v1.11-1               # 删除标签
$ git push origin --delete v1.11   # 删除远程标签
```

## .gitignore 文件

参考：

* [Git 官方教程](https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository#_ignoring)
* [Git 官方文档](https://git-scm.com/docs/gitignore)
* [.gitignore 模版合集](https://github.com/github/gitignore)

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
