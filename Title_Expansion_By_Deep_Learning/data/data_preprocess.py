from data.data_utils import all_chinese

dat = open('click_query_snipet.txt', 'r', encoding='utf8').readlines()
title_query = {}

for i in range(len(dat)):
    tokens = dat[i].strip().split('\t')
    title = tokens[0]
    seg_str = '\t'.join(tokens[1:][:4])
    title_query[title] = seg_str

data2 = open('output_all_title.txt', 'r', encoding='utf8').readlines()


out1 = open('title_seg_new.txt', 'w', encoding='utf8')
out2 = open('target_query_new.txt', 'w', encoding='utf8')

i = 0
while i < len(data2) - 1:
    # while i < 5:
    title = data2[i].strip()
    line = data2[i + 1]
    tokens = line.rstrip().split('\t')
    tokens = [word.split('@')[0] for word in tokens if len(word) >= 1 and all_chinese(word.split('@')[0])]
    tokens = [t for t in tokens if len(t) >= 1]
    if len(tokens) == 0 or title not in title_query.keys():
        i += 2
        continue

    seg_str = "\t".join(tokens)
    out1.writelines(seg_str)
    out1.writelines('\n')
    out2.writelines(title_query[title])
    out2.writelines('\n')
    if i % 10000 == 0:
        print(i)
    i += 2
out1.close()
out2.close()
