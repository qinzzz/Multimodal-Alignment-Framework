import re
import utils
import random
import json
import os
import sys
import h5py
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import _pickle as cPickle
from tqdm import tqdm
from xml.etree.ElementTree import parse
from dataset import Indexer, WordEmbeddings, Flickr30dataset, read_word_embeddings, load_train_flickr30k
from utils import largest, confidence, union, bbox_is_match, get_match_index, my_load_flickr30k
from model import NN
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py
from train_model import train, evaluate, load_entries, entries_id2img, model_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--mini', action = 'store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--bert', action='store_true')
    parser.add_argument('--lite_bert', action = 'store_true')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--save_name', type =str, default = 'model')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vgg', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() >= 1:
        print("Use {} gpus".format(torch.cuda.device_count()))

    wordEmbedding = read_word_embeddings("data/glove/glove.6B.300d.txt")
    if args.test:
        test_dset = Flickr30dataset(wordEmbedding, "test", use_bert = args.bert ,lite_bert=args.lite_bert, vgg=args.vgg )
    else:
        test_dset = Flickr30dataset(wordEmbedding, "val", use_bert = args.bert, lite_bert = args.lite_bert, vgg=args.vgg)

    test_loader = DataLoader(test_dset, batch_size=args.batch, num_workers=4, drop_last=True, shuffle=True)
    model = NN(wordEmbedding, vgg=args.vgg)
    if torch.cuda.is_available():
        print("CUDA available")
        # model = nn.DataParallel(model) # GPU parallel (have bug in parapllel?)
        # model.to(device)
        model.cuda()

    if args.eval:
        score = model_eval(test_loader, model, use_bert = args.bert, lite_bert=args.lite_bert)
        print("untrained eval score:", score)
    elif args.mini:
        train_dset = Flickr30dataset(wordEmbedding, "val", use_bert = args.bert, lite_bert=args.lite_bert)
        train_loader = DataLoader(train_dset, batch_size=args.batch, num_workers=4, drop_last=True, shuffle=True)
        train(model, train_loader, test_loader, batch=args.batch, lr=args.lr, epochs = args.epochs, use_bert = args.bert, lite_bert=args.lite_bert)
    else:
        train_dset = Flickr30dataset(wordEmbedding, "train", use_bert = args.bert, lite_bert=args.lite_bert, vgg =args.vgg)
        train_loader = DataLoader(train_dset, batch_size=args.batch, num_workers=4, drop_last=True, shuffle=True)
        train(model, train_loader, test_loader, batch=args.batch, lr=args.lr, epochs = args.epochs, use_bert = args.bert, lite_bert=args.lite_bert)
        save_path = os.path.join("saved", args.save_name+'.pt')
        torch.save(model.cpu().state_dict(),save_path)
        print("save modle to", save_path)
