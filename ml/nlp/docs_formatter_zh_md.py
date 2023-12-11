#!/usr/bin/env python3
# -*- coding:utf8 -*-  

"""
A script that format Chinese markdown docs.

Refer to https://docs.google.com/document/d/1UR0etJkmvQJJjD6zmnWazIAWQc9XYEVEfcgR_JYQXwY/edit?usp=sharing 
to be aware of the standard format. The function of this script is limited, for 
example it cannot decide whether comma(，) or ideographic comma(、) should be used. 
Manual inspection is still required.

Usage: run the following command from the root directory of this project:

./docs_formatter_zh_md.py <relative-path-of-file-or-dir>

For example:

./docs_formatter_zh_md.py content/zh/userguide/overview
./docs_formatter_zh_md.py content/zh/userguide/overview/_index.md
"""


import os
import re
import sys

if len(sys.argv) == 2:
    format_path = sys.argv[1]
else:
    print('Error: Please pass in the relative path of file or directory.')
    print('For example:')
    print('./docs_formatter_zh_md.py content/zh/userguide/overview')
    print('./docs_formatter_zh_md.py content/zh/userguide/overview/_index.md')
    sys.exit()

format_path = re.sub(r'^\.?/?', './', format_path)

def correct_space_bold_italic(line):
    prefix = ''
    if re.match(r'\s*\* ', line):
        prefix = re.findall(r'(^\s*\* )', line)[0]
        line = re.split(r'^\s*\* ', line)[-1]

    strings = []
    while re.search(r'(\*\*?)[^\*]*\1', line):
        start_index, end_index = re.search(r'(\*\*?)[^\*]*\1', line).span()
        strings.append(line[:start_index])
        strings.append(line[start_index:end_index])
        line = line[end_index:]
    strings.append(line)

    for i, v in enumerate(strings):
        if '*' in v:
            if re.search(r'^\*\*?[\u4e00-\u9fff]', v) and re.search(r'[\d+×><=≈%a-zA-Z]$', strings[i-1]):
                strings[i] = ' ' + strings[i]
            if re.search(r'^\*\*?[\d+×><=≈%a-zA-Z]', v) and re.search(r'[\u4e00-\u9fff]$', strings[i-1]):
                strings[i] = ' ' + strings[i]
            if re.search(r'[\u4e00-\u9fff]\*\*?$', v) and re.search(r'^[\d+×><=≈%a-zA-Z\-]', strings[i+1]):
                strings[i] = strings[i] + ' '
            if re.search(r'[\d+×><=≈%a-zA-Z]\*\*?$', v) and re.search(r'^[\u4e00-\u9fff\-]', strings[i+1]):
                strings[i] = strings[i] + ' '

    line = ''.join(strings)
    if prefix:
        line = prefix + line

    return line

for root, _, files in os.walk('.'):
    for f in files:
        path = os.path.join(root, f)
        if path.endswith('.md') and path.startswith(format_path):
            print('handling', path)

            with open(path, 'rt') as f:
                lines = f.readlines()
                functional_block = False
                blockquote_table_block = False
                for index, line in enumerate(lines):
                    if not functional_block and not blockquote_table_block:
                        # Skip functional blocks.
                        if line == '---\n' or re.fullmatch(r'\s*```[a-z]*\n', line):
                            functional_block = True
                            continue
                        # Skip blockquotes and tables.
                        if re.match(r'^> ', line) or re.match(r'^\| .+ \|', line):
                            blockquote_table_block = True
                            continue
                        # Skip titles.
                        if re.match(r'#+ ', line):
                            continue
                        # Skip pictures.
                        if re.match(r'\s*\[\{\{< figure src=', line):
                            continue
                        # Skip pageinfo.
                        if line.startswith('{{%'):
                            continue
                        # Unordered lists use '*' instead of '+' or '-'.
                        if re.match(r'\s*[+-] ', line):
                            line = re.sub(r'^(\s*)[+-] ', r'\1* ', line)
                        # Use full-width parentheses, commas and colons.
                        # Use unicode character ™(U+2122) for trademark.
                        line = line.replace('(', '（').replace(')', '）').replace(
                            ',', '，').replace(':', '：').replace('&trade;', '™')
                        # Correct parentheses and colons of (markdown) links.
                        line = re.sub(r'(?:(?<=https)|(?<=http)|(?<=t9k))：(?=//)', ':', line)  # 'http://', 'https://', 't9k://'
                        line = re.sub(r'(?<=\])（', '(', line)
                        line = re.sub(r'(\]\([^）]*)）', r'\1)', line)
                        # Correct colons between image/package and version
                        line = re.sub(r'(?<=[\da-zA-Z])：(?=\d\.)', ':', line)
                        # Merge number and mathmatical symbol
                        line = re.sub(r'(\d)\s*([+×><=≈%])', r'\1\2', line)
                        line = re.sub(r'([+×><=≈%])\s*(\d)', r'\1\2', line)
                        # Leave a space between Chinese and English/number(mathmatical 
                        # symbols are regarded as part of number)/inline code.
                        line = re.sub(r'([\d+×><=≈%a-zA-Z`\$])\s*([\u4e00-\u9fff])', r'\1 \2', line)
                        line = re.sub(r'([\u4e00-\u9fff])\s*([\d+×><=≈%a-zA-Z`\$])', r'\1 \2', line)
                        # Leave no space between Chinese/English/number/inline code and
                        # symbol(punctuation, etc).
                        line = re.sub(r'([\d+×><=≈%a-zA-Z\u4e00-\u9fff`])\s*([^\w+×><=≈%\-`\s\$])', r'\1\2', line)
                        line = re.sub(r'([^\w+×><=≈%\-`™\s\$])\s*([\da-zA-Z+×><=≈%\u4e00-\u9fff`])', r'\1\2', line)
                        # Leave no space between Chinese and Chinese.
                        line = re.sub(r'(?<=[\u4e00-\u9fff])\s*(?=[\u4e00-\u9fff])', r'', line)
                        # Leave no space between symbol and symbol.
                        line = re.sub(r'(?<=[^\w\s])\s*(?=[^\w\s])', r'', line)
                        # Correct space between English/number with markdown links
                        # and Chinese and vice versa.
                        line = re.sub(r'([\u4e00-\u9fff])\[([\d+×><=≈%a-zA-Z][^\]]*\]\()', r'\1 [\2', line)
                        line = re.sub(r'([\d+×><=≈%a-zA-Z])\[([\u4e00-\u9fff][^\]]*\]\()', r'\1 [\2', line)
                        line = re.sub(r'(?<!\!)(\[[^\]]*[\u4e00-\u9fff]\]\([^\)]+)\)([\d+×><=≈%a-zA-Z])', r'\1) \2', line)
                        line = re.sub(r'(?<!\!)(\[[^\]]*[\d+×><=≈%a-zA-Z]\]\([^\)]+)\)([\u4e00-\u9fff])', r'\1) \2', line)
                        # Correct space inside markdown local links.
                        line = re.sub(r'(?<=\]\(#)[^)]+(?=\))', lambda s: s.group().replace(' ', ''), line)
                        # Restore unordered lists.
                        if line.count('*') % 2 == 1:
                            line = re.sub(r'^(\s*)\*', r'\1* ', line)
                        # Restore ordered lists.
                        line = re.sub(r'^(\s*)(\d).([\u4e00-\u9fff])', r'\1\2. \3', line)
                        # Correct space between bold/italic English/number and Chinese
                        # and vice versa.
                        if line.count('*') > 1:
                            line = correct_space_bold_italic(line)
                        # Correct space between `` and code inside
                        line = re.sub(r'`\s*([^`\s]+)\s*`', r'`\1`', line)
                        # Correct parenthese, commas and colons inside ``
                        line = re.sub(r'`[^`]+`', lambda s: s.group().replace('（', '(').replace('）', ')').replace(
                            '，', ',').replace('：', ':'), line)
                        # Correct parenthese, commas and colons inside $$
                        line = re.sub(r'\$[^\$]+\$', lambda s: s.group().replace('（', '(').replace('）', ')').replace(
                            '，', ',').replace('：', ':'), line)
                        # Use correct proper nouns
                        line = re.sub(r'(?<![\w+×><=≈%\-`:/\.])[Tt]9[Kk](?![\w+×><=≈%\-`:/\.])', 'T9k', line)
                        line = re.sub(r'(?<![\w+×><=≈%\-`:/\.])[Tt]ensorstack(?![\w+×><=≈%\-`:/\.])', 'TensorStack', line)
                        line = re.sub(r'(?<![\w+×><=≈%\-`:/\.])[Tt]rainingjob(?![\w+×><=≈%\-`:/\.])', 'TrainingJob', line)
                        line = re.sub(r'(?<![\w+×><=≈%\-`:/\.])k8s(?![\w+×><=≈%\-`:/\.])', 'K8s', line)
                        line = re.sub(r'(?<![\w+×><=≈%\-`:/\.])kubernetes(?![\w+×><=≈%\-`:/\.])', 'Kubernetes', line)
                        line = re.sub(r'(?<![\w+×><=≈%\-`:/\.])[Tt]ensorflow(?![\w+×><=≈%\-`:/\.])', 'TensorFlow', line)
                        line = re.sub(r'(?<![\w+×><=≈%\-`:/\.])[Tt]ensorboard(?![\w+×><=≈%\-`:/\.])', 'TensorBoard', line)
                        line = re.sub(r'(?<![\w+×><=≈%\-`:/\.])[Pp]ytorch(?![\w+×><=≈%\-`:/\.])', 'PyTorch', line)
                        line = re.sub(r'(?<![\w+×><=≈%\-`:/\.])xgboost(?![\w+×><=≈%\-`:/\.])', 'XGBoost', line)
                        lines[index] = line
                    elif functional_block:
                        # Functional blocks ends.
                        if line == '---\n' or re.fullmatch(r'\s*```\n', line):
                            functional_block = False
                            continue
                    else:
                        # Blockquotes and tables ends.
                        if line == '\n':
                            blockquote_table_block = False
                            continue
                    
            with open(path, 'wt') as f:
                f.write(''.join(lines))
