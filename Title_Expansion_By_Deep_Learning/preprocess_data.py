# -*- encoding: utf-8 -*-
import argparse
from data.data_utils import build_vocab, convert_text2idx, norm, filter_emoji
import joblib
import numpy as np

__author__ = "Yuehua Wang"


def prepare_data_for_train(opt):
    '''
    Usage: python preprocess.py
    this function is used to process the raw chinese  title-query data to the numpy array format
    this function build the vocabulary table and convert the chinese to the index number
    '''

    title_data = open(opt.title_path, 'r', encoding='utf8').readlines()
    query_data = open(opt.query_data, 'r', encoding='utf8').readlines()
    assert len(title_data) == len(query_data)
    print(len(title_data))
    print(len(query_data))

    # calculate the average length of all titles and querys

    train_src = []
    train_len = 0
    for d in title_data:
        temp = d.strip().split('\t')
        train_len += len(temp)
        train_src.append(temp)
    print(f'-[average len of title is]- {train_len / len(train_src)}')

    train_tgt = []
    query_len = 0
    for i in range(len(query_data)):
        temp_words = [d.strip().split('^a^')[0] for d in query_data[i].split('\t')]
        query_len += len(temp_words)
        train_tgt.append(temp_words)
    print(f'-[average len of query is]- {query_len / len(train_tgt)}')

    # get the target words and corresponding distribution score

    train_src = []
    train_tgt = []
    train_word_scores = []

    for i in range(len(query_data)):
        temp = title_data[i].strip().split('\t')
        temp_words = [d.strip().split('^a^')[0] for d in query_data[i].split('\t')]
        temp_word_score = [float(d.strip().split('^a^')[-1]) for d in
                           query_data[i].split('\t')]

        assert len(temp_words) == len(temp_word_score)
        if temp_words[0] == '':
            print(f' -[the {i}th sample has problem]- ')
            continue

        train_src.append(temp)
        train_tgt.append(temp_words)
        train_word_scores.append(temp_word_score)
    assert len(train_tgt) == len(train_word_scores)

    pad_token = "<pad>"
    unk_token = "<unk>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    extra_tokens = [pad_token, unk_token, bos_token, eos_token]

    # build vocab for the data

    _, src_word2idx, _ = build_vocab(train_src + train_tgt, opt.max_limit,
                                     opt.min_freq, extra_tokens)

    # filter the emoji word
    ind = 0
    new_src_word2idx = {}
    for word, _ in src_word2idx.items():
        if len(filter_emoji(word)) != 0:
            new_src_word2idx[word] = ind
            ind += 1
        else:
            continue

    # convert text to index
    train_src, train_tgt = convert_text2idx(train_src, new_src_word2idx), convert_text2idx(train_tgt, new_src_word2idx)

    # set the length of each sample equal

    label_len = 9
    print(f' -[the length of label_len is]-  :{label_len}')

    for i in range(len(train_src)):
        if len(train_src[i]) < label_len:
            train_src[i] = train_src[i] + [new_src_word2idx['<pad>']] * (label_len - len(train_src[i]))
        if len(train_src[i]) > label_len:
            train_src[i] = train_src[i][:label_len]

    train_src = np.array(train_src)
    print(f' -[the shape of train_src is :{train_src.shape} "]-')

    for i in range(len(train_tgt)):
        assert len(train_tgt[i]) == len(train_word_scores[i])
        if len(train_tgt[i]) < 8:
            train_tgt[i] += [new_src_word2idx['<pad>']] * (8 - len(train_tgt[i]))
            train_word_scores[i] += [0] * (8 - len(train_word_scores[i]))
        if len(train_tgt[i]) > 8:
            train_tgt[i] = train_tgt[i][:8]
            train_word_scores[i] = train_word_scores[i][:8]
        assert len(train_tgt[i]) == 8
        assert len(train_word_scores[i]) == 8

    train_src = np.array(train_src)
    train_tgt = np.array(train_tgt)
    train_word_scores = np.array(train_word_scores)
    print(f'-[the shape of train_src is]- : {train_src.shape}')
    print(f'-[the shape of  train_word_scores is]- :{train_word_scores}')
    print(f'-[the shape of train_tgt is]- : {train_tgt.shape}')
    assert len(train_src) == len(train_word_scores)

    print(' -[ Nomalization data]- ')
    train_word_scores = norm(train_word_scores)

    print(' -[saving data ]-')
    joblib.dump(new_src_word2idx, opt.src_word2idx)
    np.save(opt.title_data, train_src)
    np.save(opt.target_index, train_tgt)
    np.save(opt.target_scores, train_word_scores)

    print(f'-[the final vocab size is]- ：{len(new_src_word2idx)}')
    print(f'-[the len of emoji is ]-：{len(src_word2idx) - len(new_src_word2idx)}')
    print(f'-[the len of whole training data is] {len(train_word_scores)}')
    print('-[all data needed saved.] ')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-title_path', default='./data/title_seg_new.txt', help='the raw title data')
    parser.add_argument('-query_data', default='./data/target_query_new.txt', help='the raw query data')
    parser.add_argument('-max_limit', type=int, default=None, help='the upper limit of the vocabulary ')
    parser.add_argument('-min_freq', type=int, default=3, help='the minimum number of the vocabulary ')
    parser.add_argument('-title_data', default='./data/train_src.npy')
    parser.add_argument('-target_index', default='./data/train_tgt.npy')
    parser.add_argument('-target_scores', default='./data/train_word_scores.npy')
    parser.add_argument('-src_word2idx', default='./data/src_word2idx.pkl')

    opt = parser.parse_args()
    print(opt)
    prepare_data_for_train(opt)


if __name__ == '__main__':
    main()
