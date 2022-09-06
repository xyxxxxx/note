NPM（Node Package Manager）是JavaScript的包管理工具，随同Node.js一同安装。Node.js是基于Chrome V8 JavaScript引擎的JavaScript运行时。

NPM的功能包括允许用户：

+ 从NPM服务器下载别人编写的第三方包到本地使用
+ 从NPM服务器下载并安装别人编写的命令行程序到本地使用
+ 将自己编写的包或命令行程序上传到NPM服务器

## 命令

> 参考[npm 常用命令详解](https://www.cnblogs.com/PeunZhang/p/5553574.html)

### install & uninstall

`install`命令安装一个包，以及它的所有依赖。npm的包指包含js程序和`package.json`文件的目录，或者上述目录的压缩包，或者解析为上述压缩包的url，或者发布在npm registry上的`<name>@<version>`……

```shell
# 安装包
$ npm install <Package_Name>        # 安装指定包
$ npm install <Package_Name>@1.2.3  # 安装指定版本
$ npm install  # 安装 package.json 中列出的所有依赖，到开发目录下的node_modules下
#             -g 全局安装
$ npm ci       # 类似于 npm install，但它旨在用于自动化环境，如测试平台，持续集成和部署 

# 卸载包
$ npm uninstall <Package_Name>
# aliases: remove, rm, r, un, unlink
```

### list

```shell
# 查看包信息
$ npm ls        # 查看所有包
$ npm ls -g     # 查看所有全局安装的包及依赖
$ npm ls grunt  # 查看指定包
# aliases: list, la, ll
```

### update

```shell
# 更新包
$ npm update <Package_Name>
```

### version

```shell
# 查看包版本
$ npm version
```

### search

```shell
# 搜索包
$ npm search <Package_Name>
```

### root

```shell
# 输出node_modules的路径
$ npm root [-g]
```

### run

```shell
# 执行脚本
$ npm run <script>
# 相当于执行scripts对象中对应的命令

$ npm start  # 相当于 npm run start
$ npm stop   # 相当于 npm run stop
$ npm test   # 相当于 npm run test
```

```json
"scripts": {
    "start": "gulp -ws"
}
```

### init & publish

```shell
# 创建包
$ npm init
# 在引导下创建一个package.json文件

# 发布模块
$ npm publish
```

### config

```shell
# 设置代理
$ npm config set proxy=http://xxx.com:8080

# 设置镜像
$ npm config set registry="http://r.cnpmjs.org"

```

### cache

```shell
# 清除npm本地缓存
$ npm cache clean
```

## `package.json`

`package.json` 位于模块的根目录下，用于定义包的属性。这里以 express 包的 package.json 文件为例：

```json
{
  "name": "express", // 包名
  "description": "Fast, unopinionated, minimalist web framework",  // 描述
  "version": "4.13.3", // 包的版本号
  "author": {
    "name": "TJ Holowaychuk",
    "email": "tj@vision-media.ca"
  },
  "contributors": [
    {
      "name": "Aaron Heckmann",
      "email": "aaron.heckmann+github@gmail.com"
    },
    {
      "name": "Ciaran Jessup",
      "email": "ciaranj@gmail.com"
    },
    {
      "name": "Douglas Christopher Wilson",
      "email": "doug@somethingdoug.com"
    },
    {
      "name": "Guillermo Rauch",
      "email": "rauchg@gmail.com"
    },
    {
      "name": "Jonathan Ong",
      "email": "me@jongleberry.com"
    },
    {
      "name": "Roman Shtylman",
      "email": "shtylman+expressjs@gmail.com"
    },
    {
      "name": "Young Jae Sim",
      "email": "hanul@hanul.me"
    }
  ],
  "license": "MIT",
  "repository": { // 存放位置
    "type": "git",
    "url": "git+https://github.com/strongloop/express.git"
  },
  "homepage": "http://expressjs.com/",
  "keywords": [
    "express",
    "framework",
    "sinatra",
    "web",
    "rest",
    "restful",
    "router",
    "app",
    "api"
  ],
  "dependencies": { // 依赖列表
    "accepts": "~1.2.12",      // 兼容[1.2.12, 1.3)
    "array-flatten": "1.1.1",  // 兼容1.1.1
    "content-disposition": "0.5.0",
    "content-type": "~1.0.1",
    "cookie": "0.1.3",
    "cookie-signature": "1.0.6",
    "debug": "~2.2.0",
    "depd": "~1.0.1",
    "escape-html": "1.0.2",
    "etag": "~1.7.0",
    "finalhandler": "0.4.0",
    "fresh": "0.3.0",
    "merge-descriptors": "1.0.0",
    "methods": "~1.1.1",
    "on-finished": "~2.3.0",
    "parseurl": "~1.3.0",
    "path-to-regexp": "0.1.7",
    "proxy-addr": "~1.0.8",
    "qs": "4.0.0",
    "range-parser": "~1.0.2",
    "send": "0.13.0",
    "serve-static": "~1.10.0",
    "type-is": "~1.6.6",
    "utils-merge": "1.0.0",
    "vary": "~1.0.1"
  },
  "devDependencies": {
    "after": "0.8.1",
    "ejs": "2.3.3",
    "istanbul": "0.3.17",
    "marked": "0.3.5",
    "mocha": "2.2.5",
    "should": "7.0.2",
    "supertest": "1.0.1",
    "body-parser": "~1.13.3",
    "connect-redis": "~2.4.1",
    "cookie-parser": "~1.3.5",
    "cookie-session": "~1.2.0",
    "express-session": "~1.11.3",
    "jade": "~1.11.0",
    "method-override": "~2.3.5",
    "morgan": "~1.6.1",
    "multiparty": "~4.1.2",
    "vhost": "~3.0.1"
  },
  "engines": {
    "node": ">= 0.10.0"
  },
  "files": [
    "LICENSE",
    "History.md",
    "Readme.md",
    "index.js",
    "lib/"
  ],
  "scripts": { // 脚本
    "test": "mocha --require test/support/env --reporter spec --bail --check-leaks test/ test/acceptance/",
    "test-ci": "istanbul cover node_modules/mocha/bin/_mocha --report lcovonly -- --require test/support/env --reporter spec --check-leaks test/ test/acceptance/",
    "test-cov": "istanbul cover node_modules/mocha/bin/_mocha -- --require test/support/env --reporter dot --check-leaks test/ test/acceptance/",
    "test-tap": "mocha --require test/support/env --reporter tap --check-leaks test/ test/acceptance/"
  },
  "gitHead": "ef7ad681b245fba023843ce94f6bcb8e275bbb8e",
  "bugs": {
    "url": "https://github.com/strongloop/express/issues"
  },
  "_id": "express@4.13.3",
  "_shasum": "ddb2f1fb4502bf33598d2b032b037960ca6c80a3",
  "_from": "express@*",
  "_npmVersion": "1.4.28",
  "_npmUser": {
    "name": "dougwilson",
    "email": "doug@somethingdoug.com"
  },
  "maintainers": [
    {
      "name": "tjholowaychuk",
      "email": "tj@vision-media.ca"
    },
    {
      "name": "jongleberry",
      "email": "jonathanrichardong@gmail.com"
    },
    {
      "name": "dougwilson",
      "email": "doug@somethingdoug.com"
    },
    {
      "name": "rfeng",
      "email": "enjoyjava@gmail.com"
    },
    {
      "name": "aredridel",
      "email": "aredridel@dinhe.net"
    },
    {
      "name": "strongloop",
      "email": "callback@strongloop.com"
    },
    {
      "name": "defunctzombie",
      "email": "shtylman@gmail.com"
    }
  ],
  "dist": {
    "shasum": "ddb2f1fb4502bf33598d2b032b037960ca6c80a3",
    "tarball": "http://registry.npmjs.org/express/-/express-4.13.3.tgz"
  },
  "directories": {},
  "_resolved": "https://registry.npmjs.org/express/-/express-4.13.3.tgz",
  "readme": "ERROR: No README data found!"
}
```

