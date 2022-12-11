# PyYAML (yaml)

PyYAML 是一个 YAML 编码和解码器，使用方法类似于标准库的 json 包。

## load()

将 YAML 文档转换为 Python 对象。接受一个 Unicode 字符串、字节字符串、二进制文件对象或文本文件对象，其中字节字符串和文件必须使用 utf-8、utf-16-be 或 utf-16-le 编码（若没有指定编码，则默认为 utf-8 编码）。

出于[安全上的原因](https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation)，调用此函数时需要传入 `Loader` 参数，目前该参数有以下选项：

* `BaseLoader`

  仅加载最基本的 YAML，所有的数字都会被加载为字符串。

* `SafeLoader`

  安全地加载 YAML 语言的一个子集，是加载不被信任的数据时的推荐选项。对应于快捷函数 `yaml.safe_load()`。

* `FullLoader`

  加载完整的 YAML 语言，是当前的默认选项。存在明显的漏洞，暂时不要加载不被信任的数据。对应于快捷函数 `yaml.full_load()`。

* `UnsafeLoader`

  最初的 `Loader` 实现，可以轻易地被不被信任的数据输入利用。对应于快捷函数 `yaml.unsafe_load()`。

```python
>>> import yaml
>>> doc = """
# Project information
site_name: Test Docs
site_author: xyx

# Repository
repo_name: xyx/test-project/mkdocs/test
repo_url: http://gitlab.dev.tensorstack.net/xyx/test-project/tree/master/mkdocs/test-project
edit_uri: ""

# Copyright
copyright: Copyright &copy; 2016 - 2021 xxx

# Configuration
theme:
  name: material  # https://github.com/squidfunk/mkdocs-material
  custom_dir: overrides    # overrides HTML elements
  language: zh
  features:
    - navigation.sections  # keep this
    - navigation.tabs      # keep this
    - navigation.top       # keep this
  palette:
    scheme: default        # keep this
    primary: green         # primary color of theme
    accent: light green    # color of elements that can be interacted with
  favicon: assets/icon.svg # showed as tab icon
  logo: assets/logo.svg    # showed at top left of page
"""
>>> from pprint import pprint
>>> pprint(yaml.load(doc, Loader=yaml.SafeLoader))   # yaml to py dict
{'copyright': 'Copyright &copy; 2016 - 2021 xxx',
 'edit_uri': '',
 'repo_name': 'xyx/test-project/mkdocs/test',
 'repo_url': 'http://gitlab.dev.tensorstack.net/xyx/test-project/tree/master/mkdocs/test-project',
 'site_author': 'xyx',
 'site_name': 'Test Docs',
 'theme': {'custom_dir': 'overrides',
           'favicon': 'assets/icon.svg',
           'features': ['navigation.sections',
                        'navigation.tabs',
                        'navigation.top'],
           'language': 'zh',
           'logo': 'assets/logo.svg',
           'name': 'material',
           'palette': {'accent': 'light green',
                       'primary': 'green',
                       'scheme': 'default'}}}
```

```python
# Load yaml from file
>>> with open('test.yaml', 'rt') as f:
...   pprint(yaml.load(f, Loader=yaml.SafeLoader))
... 
{'copyright': 'Copyright &copy; 2016 - 2021 xxx',
 'edit_uri': '',
 'repo_name': 'xyx/test-project/mkdocs/test',
 'repo_url': 'http://gitlab.dev.tensorstack.net/xyx/test-project/tree/master/mkdocs/test-project',
 'site_author': 'xyx',
 'site_name': 'Test Docs',
 'theme': {'custom_dir': 'overrides',
           'favicon': 'assets/icon.svg',
           'features': ['navigation.sections',
                        'navigation.tabs',
                        'navigation.top'],
           'language': 'zh',
           'logo': 'assets/logo.svg',
           'name': 'material',
           'palette': {'accent': 'light green',
                       'primary': 'green',
                       'scheme': 'default'}}}
```

## dump()

将 Python 对象转换为 YAML 文档。

```python
>>> import yaml
>>> d = {'copyright': 'Copyright &copy; 2016 - 2021 xxx',
         'edit_uri': '',
         'repo_name': 'xyx/test-project/mkdocs/test',
         'repo_url': 'http://gitlab.dev.tensorstack.net/xyx/test-project/tree/master/mkdocs/test-project',
         'site_author': 'xyx',
         'site_name': 'Test Docs',
         'theme': {'custom_dir': 'overrides',
                   'favicon': 'assets/icon.svg',
                   'features': ['navigation.sections',
                                'navigation.tabs',
                                'navigation.top'],
                   'language': 'zh',
                   'logo': 'assets/logo.svg',
                   'name': 'material',
                   'palette': {'accent': 'light green',
                               'primary': 'green',
                               'scheme': 'default'}}}
>>> print(yaml.dump(d))                                # py dict to yaml
copyright: Copyright &copy; 2016 - 2021 xxx
edit_uri: ''
repo_name: xyx/test-project/mkdocs/test
repo_url: http://gitlab.dev.tensorstack.net/xyx/test-project/tree/master/mkdocs/test-project
site_author: xyx
site_name: Test Docs
theme:
  custom_dir: overrides
  favicon: assets/icon.svg
  features:
  - navigation.sections
  - navigation.tabs
  - navigation.top
  language: zh
  logo: assets/logo.svg
  name: material
  palette:
    accent: light green
    primary: green
    scheme: default

```

```python
>>> import yaml
>>> bart = Student('Bart Simpson', 59)
>>> print(yaml.dump(bart))                              # py normal object to yaml
!!python/object:__main__.Student
name: Bart Simpson
score: 59

```
