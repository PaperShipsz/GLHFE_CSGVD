import argparse
import logging
import os
import pickle
import sys
import torch
import json
import torch.optim as optimizer
from data_loader.dataset import DataSet
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaModel)
from utils.ops import tally_param, debug, set_logger,set_seed, split_dataset, random_undersample,random_oversample
from trainer import train
torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True
from utils.TextModel import TextEncoder
from utils.Model import GLHFE_CSGVD
def warn(*args, **kwargs):
    pass
import warnings

warnings.warn = warn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Name of the dataset for experiment.', default='devign')
    parser.add_argument('--input_dir', type=str, help='Input Directory of the parser',
                        default='')
    parser.add_argument('--log_dir', default='result.log', type=str)
    parser.add_argument('--seed', type=int, default=10, help="random seed for initialization")
    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=100)
    parser.add_argument('-l_num', type=int, default=2, help='graph layer num')
    parser.add_argument('-t_num', type=int, default=1, help='text layer num')
    parser.add_argument('-h_dim', type=int, default=512, help='hidden dim')
    parser.add_argument('-l_dim', type=int, default=128, help='layer dim')
    parser.add_argument('-drop_n', type=float, default=0.2, help='drop TEXT')
    parser.add_argument('-drop_c', type=float, default=0.2, help='drop GRAPH')
    # parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('-ts', nargs='+', type=float, default=0.9)
    parser.add_argument('-ks', nargs='+', type=float, default=0.8)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=16)
    args = parser.parse_args()
    model_dir = os.path.join('models', args.dataset)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    log_dir = os.path.join(model_dir, args.dataset + '_'+'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_name = os.path.join(model_dir, args.dataset + '_'+'log', args.log_dir)
    set_logger(log_name)

    args.model_name = 'new_model' + '.bin'
    args.save_model_dir = model_dir+'/saved_model'
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    args.tokenizer_name = './utils/codebert-base'
    args.model_name_or_path = './utils/codebert-base'


    args.train_data_file = '../Dataset/' + args.dataset + '_dataset/train' + '.json'
    args.eval_data_file = '../Dataset/' + args.dataset + '_dataset/valid_' + '.json'
    args.test_data_file = '../Dataset/' + args.dataset + '_dataset/test_' + '.json'
    dataset_path = './dataset/' + args.dataset + '_dataset/mixed_' + '.pkl'
    path = '../Dataset/' + args.dataset + '_dataset/'+args.dataset + '_data.json'
    set_seed(args)

    if not os.path.exists(args.train_data_file):
        logging.info("Creating train/eval/test dataset...")
        logging.info('set seed: %d', args.seed)
        train_data, valid_data, test_data = split_dataset(path)
        with open(args.train_data_file, 'w', encoding='utf-8') as f1:
            for item in train_data:
                json.dump(item, f1)
                f1.write('\n')
        with open(args.eval_data_file, 'w', encoding='utf-8') as f2:
            for item in valid_data:
                json.dump(item, f2)
                f2.write('\n')
        with open(args.test_data_file, 'w', encoding='utf-8') as f3:
            for item in test_data:
                json.dump(item, f3)
                f3.write('\n')

    # Setup CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1
    args.device = device


    logging.info("Training/evaluation parameters %s", args)
    torch.cuda.empty_cache()

    # Load Tokenizer
    args.num_attention_heads = 12
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = 1
    config.num_attention_heads = args.num_attention_heads
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    codebert = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config,
                                                                ignore_mismatched_sizes=True)
    # Training
    if os.path.exists(dataset_path):
        with open(dataset_path, 'rb') as f:
            print('load dataset from pickle from %s' % dataset_path)
            dataset = pickle.load(f)
    else:
        print('create dataset')
        dataset = DataSet(train_src=args.train_data_file,

                          valid_src=args.eval_data_file,
                          test_src=args.test_data_file,
                          batch_size=args.batch_size,
                          tokenizer=tokenizer)
        directory_path = './dataset/' + args.dataset + '_dataset'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
    args.num_heads = 8
    args.max_edge_types = dataset.max_etype

    text_model = TextEncoder(codebert, config, tokenizer, args)
    model = GLHFE_CSGVD(text_model, in_dim=args.feature_size, args=args)

    model.cuda()

    LR = 2e-5
    optim = optimizer.AdamW(model.parameters(), lr=LR, eps=1e-8, weight_decay=1e-6)
    train(model=model, dataset=dataset, epoches=100, dev_every=len(dataset.train_batches),
          optimizer=optim, save_path=args.save_model_dir+'/'+args.model_name, max_patience=50, log_every=5)



if __name__ == "__main__":
    main()

