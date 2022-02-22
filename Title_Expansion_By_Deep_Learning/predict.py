# -*- encoding: utf-8 -*-
import argparse
from transformer.Models import Transformer
from fasttxt.Models import Fasttxt
import numpy as np
import torch
import joblib

__author__ = "Yuehua Wang"


def process_one_prediction(opt, title, query, wight, model):

    assert len(title) == 9
    test_src = []
    test_src.append(title)
    src = torch.from_numpy(np.array(test_src))

    model.eval()
    device = opt.device
    src_indx2word = opt.src_indx2word
    pred = model(src.long().to(device))
    pred_soft = pred

    top = opt.top_num
    top_index = pred_soft[0].argsort()[-top:].tolist()
    top_index.reverse()
    res = [(src_indx2word[index], pred_soft[0][index].item()) for index in top_index]

    out = open(opt.output_path, 'a', encoding='utf8')
    title_text = ' '.join([src_indx2word[index] for index in title])
    out.write('-[title]-')
    out.write('\n')
    out.write(title_text)
    out.write('\n')

    query_text = [src_indx2word[index] for index in query]

    out.write('-[the true label dis]-')
    out.write('\n')
    assert len(query_text) == len(wight)
    for i in range(len(query_text)):
        out.write(str(query_text[i]) + '^a^' + str(wight[i]))
        out.write('\t')
    out.write('\n')

    out.write('-[the predict label dis]-')
    out.write('\n')
    pred_weight = [pred_soft[0][index] for index in query]
    assert len(query_text) == len(wight)
    for i in range(len(query_text)):
        out.write(query_text[i] + '^a^' + str(pred_weight[i].item()))
        out.write('\t')
    out.write('\n')

    out.write('-[the top predict dis]-')
    out.write('\n')
    for r in res:
        out.write(str(r[0]) + '^a^' + str(r[1]))
        out.write('\t')
    out.write('\n' * 3)
    print('done 1')
    out.close()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='cuda')
    parser.add_argument('-model_path', default='./final_model/final_tf.chkpt')
    parser.add_argument('-top_num', default=10)
    parser.add_argument('-prediction_nums', default=1000)
    parser.add_argument('-output_path', default='pred_fasttxt_mse_train.txt')
    parser.add_argument('-src_word2idx', default='./data/src_word2idx.pkl')

    opt = parser.parse_args()
    print(opt)

    checkpoint = torch.load(opt.model_path, map_location=opt.device)
    model_opt = checkpoint['settings']
    opt.src_word2idx = joblib.load(model_opt.src_word2idx)
    print(f'the size of vocab is {model_opt.n_src_vocab}')

    src_indx2word = {}
    for key, val in list(opt.src_word2idx.items()):
        src_indx2word[val] = key
    opt.src_indx2word = src_indx2word

    print('model_opt.dropout', model_opt.dropout)
    print('model_opt.warmup', model_opt.n_warmup_steps)

    if model_opt.model_type == 'transformer':
        model = Transformer(
            model_opt.n_src_vocab,
            model_opt.src_pad_idx,
            d_k=model_opt.d_k,
            d_v=model_opt.d_v,
            d_model=model_opt.d_model,
            d_word_vec=model_opt.d_word_vec,
            d_inner=model_opt.d_inner_hid,
            n_layers=model_opt.n_layers,
            n_head=model_opt.n_head,
            dropout=model_opt.dropout).to(opt.device)

    elif model_opt.model_type == 'fasttxt':
        model = Fasttxt(
            model_opt.n_src_vocab,
            model_opt.src_pad_idx,
            d_model=model_opt.d_model,
            d_word_vec=model_opt.d_word_vec,
            d_inner=model_opt.d_inner_hid,
            dropout=model_opt.dropout).to(opt.device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    model.eval()
    train_src = np.load(model_opt.title_data)
    train_index = np.load(model_opt.target_index)
    train_scores = np.load(model_opt.target_scores)
    for i in range(opt.prediction_nums):
        process_one_prediction(opt, train_src[i], train_index[i], train_scores[i], model)


if __name__ == '__main__':
    main()
