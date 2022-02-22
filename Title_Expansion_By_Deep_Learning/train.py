# -*- encoding: utf-8 -*-
'''
This script handles the training process.
'''
import argparse
import os
import time
from torchtext.data import Field, BucketIterator
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from data.dataset import ParallelDataset
from loss.loss_function import cross_entropy, kl_loss, mse_loss
from fasttxt.Models import Fasttxt
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
import joblib


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

__author__ = "Yuehua Wang"


def train_epoch(model, training_data, optimizer, opt, device):
    ''' Epoch operation in training phase'''

    model = model.cuda()
    model.train()
    total_loss = 0

    desc = '  - (Training)   '
    batch_count = 0
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):

        # prepare data
        src_seq = batch.src.to(device)
        trg_index = batch.trg_index.to(device)
        trg_score = batch.trg_score.to(device)

        optimizer.zero_grad()
        pred = model(src_seq)

        true_dis = np.zeros([pred.shape[0], pred.shape[1]])
        for i in range(len(true_dis)):
            for j, ind in enumerate(trg_index[i]):
                true_dis[i][ind] = trg_score[i][j]
        true_dis = torch.from_numpy(true_dis).type_as(pred)

        if opt.loss == 'kl':
            loss = kl_loss(pred, true_dis)
        elif opt.loss == 'ce':
            loss = cross_entropy(pred, true_dis)
        elif opt.loss == 'mse':
            loss = mse_loss(pred, true_dis)

        loss.backward()
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()

        # record how many batchs in one epoch
        batch_count += 1

    loss_per_batch = total_loss / batch_count

    return loss_per_batch


def eval_epoch(model, validation_data, device, opt):

    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss = 0

    desc = '  - (Validation) '
    batch_count = 0
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):
            # prepare data
            src_seq = batch.src.to(device)
            trg_index = batch.trg_index.to(device)
            trg_score = batch.trg_score.to(device)
            pred = model(src_seq)

            true_dis = np.zeros([pred.shape[0], pred.shape[1]])
            for i in range(len(true_dis)):
                for j, ind in enumerate(trg_index[i]):
                    true_dis[i][ind] = trg_score[i][j]
            true_dis = torch.from_numpy(true_dis).type_as(pred)

            if opt.loss == 'kl':
                loss = kl_loss(pred, true_dis)
            elif opt.loss == 'ce':
                loss = cross_entropy(pred, true_dis)
            elif opt.loss == 'mse':
                loss = mse_loss(pred, true_dis)

            # note keeping
            total_loss += loss.item()

            # record how many batchs in one epoch
            batch_count += 1

    loss_per_batch = total_loss / batch_count
    return loss_per_batch


def train(model, training_data, validation_data, optimizer, device, opt):

    ''' Start training '''

    log_train_file, log_valid_file = None, None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss\n')
            log_vf.write('epoch,loss\n')

    def print_performances(header, loss, start_time):
        print('  - {header:12} loss: {loss: 12.5f}, '
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", loss=loss,
                  elapse=(time.time() - start_time) / 60))

    valid_losses = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss = train_epoch(
            model, training_data, optimizer, opt, device)

        print_performances('Training', train_loss, start)
        start = time.time()
        valid_loss = eval_epoch(model, validation_data, device, opt)
        print_performances('Validation', valid_loss, start)
        valid_losses += [valid_loss]
        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_loss_{loss:8.5f}.chkpt'.format(loss=valid_losses)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_loss <= min(valid_losses):
                    torch.save(checkpoint, model_name)
                    print(' - [Info] The checkpoint file has been updated.')

        if epoch_i > 0 and epoch_i % 30 == 0:
            print(' - [Info]  regular model file has been updated.')
            model_name = opt.save_model + 'epoch' + str(epoch_i) + '.chkpt'
            torch.save(checkpoint, model_name)

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 12.5f}\n'.format(epoch=epoch_i, loss=train_loss))
                log_vf.write('{epoch},{loss: 12.5f}\n'.format(epoch=epoch_i, loss=valid_loss))


def main():
    '''
    Usage:
    python train.py -b 512 -log ....
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('-title_data', default='./data/train_src.npy')
    parser.add_argument('-target_index', default='./data/train_tgt.npy')
    parser.add_argument('-target_scores', default='./data/train_word_scores.npy')
    parser.add_argument('-src_word2idx', default='./data/src_word2idx.pkl')

    parser.add_argument('-epoch', type=int, default=300)
    parser.add_argument('-b', '--batch_size', type=int, default=768)
    parser.add_argument('-d_model', type=int, default=200)
    parser.add_argument('-d_inner_hid', type=int, default=1024)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup', '--n_warmup_steps', type=int, default=100)
    parser.add_argument('-dropout', type=float, default=0.1)

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default='./final_model/final_tf')
    parser.add_argument('-model_type', default='transformer', help='[fasttxt, transformer]')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-val_num', type=int, default=5000)
    parser.add_argument('-loss', type=str, default='kl')

    opt = parser.parse_args()
    print(opt)

    src_word2idx = joblib.load(opt.src_word2idx)
    opt.n_src_vocab = len(src_word2idx)
    opt.src_pad_idx = src_word2idx['<pad>']
    opt.d_word_vec = opt.d_model
    print('the len of src_vocab is : ', opt.n_src_vocab)

    if not opt.log and not opt.save_model:
        print('No experiment result will be saved.')
        raise

    if opt.batch_size < 2048 and opt.n_warmup_steps <= 4000:
        print('[Warning] The warmup steps may be not enough.\n'
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n'
              'Using smaller batch w/o longer warmup may cause '
              'the warmup stage ends with only little data trained.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('-[device is]- :', device)

    # Loading Dataset
    training_data, validation_data = prepare_dataloaders_title(opt, device)

    if opt.model_type == 'transformer':
        model = Transformer(
            opt.n_src_vocab,
            opt.src_pad_idx,
            d_k=opt.d_k,
            d_v=opt.d_v,
            d_model=opt.d_model,
            d_word_vec=opt.d_word_vec,
            d_inner=opt.d_inner_hid,
            n_layers=opt.n_layers,
            n_head=opt.n_head,
            dropout=opt.dropout).to(device)

    elif opt.model_type == 'fasttxt':
        model = Fasttxt(
            opt.n_src_vocab,
            opt.src_pad_idx,
            d_model=opt.d_model,
            d_word_vec=opt.d_word_vec,
            d_inner=opt.d_inner_hid,
            dropout=opt.dropout).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        2.0, opt.d_model, opt.n_warmup_steps)

    train(model, training_data, validation_data, optimizer, device, opt)


def prepare_dataloaders_title(opt, device):
    batch_size = opt.batch_size
    train_src = np.load(opt.title_data)
    train_index = np.load(opt.target_index)
    train_scores = np.load(opt.target_scores)

    pad_token = "<pad>"
    unk_token = "<unk>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    extra_tokens = [pad_token, unk_token, bos_token, eos_token]

    PAD = extra_tokens.index(pad_token)
    UNK = extra_tokens.index(unk_token)

    src_field = Field(sequential=True, use_vocab=False, include_lengths=False, batch_first=True,
                      pad_token=PAD, unk_token=UNK, init_token=None, eos_token=None, )
    trg_index_field = Field(sequential=False, use_vocab=False)
    trg_score_field = Field(sequential=False, use_vocab=False, dtype=torch.float32)
    fields = (src_field, trg_index_field, trg_score_field)

    val_num = opt.val_num
    train_data = ParallelDataset(train_src[:-val_num], train_index[:-val_num], train_scores[:-val_num], fields=fields)
    train_iterator = BucketIterator(train_data, batch_size=batch_size, device=device, train=True, shuffle=True)
    print(f'the len of train is {len(train_iterator) * batch_size} ')

    val_data = ParallelDataset(train_src[-val_num:], train_index[-val_num:], train_scores[-val_num:], fields=fields)
    val_iterator = BucketIterator(val_data, batch_size=batch_size, device=device, train=False)
    print(f'the len of val is {len(val_iterator) * batch_size} ')

    return train_iterator, val_iterator


if __name__ == '__main__':
    main()
