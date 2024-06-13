import re
def cut_sent(para):
    para = re.sub('([。！？；;\?])([^”’])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    return para.split("\n")


pp_name = '58同城'
pp = open(f'./sentence/{pp_name}.txt', 'r', encoding='utf-8')
data = cut_sent(pp.read())
f = open(f"./out_sentences/{pp_name}.txt", 'w', encoding='utf-8')
for text in data:
    if not text == '':
        f.write(text + '\n')