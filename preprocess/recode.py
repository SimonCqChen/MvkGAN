import chardet
import codecs

# !!! does not backup the origin file
content = codecs.open('../data/copyright.CSV', 'r', encoding='GBK').read()

codecs.open('../data/copyright.CSV', 'w', encoding='utf-8').write(content)
