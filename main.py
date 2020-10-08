import os
import random
import warnings
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import Flickr30dataset
from utils.utils import read_word_embeddings
from model import MATnet
from train_model import train

with warnings.catch_warnings():
	warnings.filterwarnings("ignore", category = FutureWarning)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch', type = int, default = 64,
						help = "batch size for training")
	parser.add_argument('--lr', type = float, default = 1e-5,
						help = "learning rate")
	parser.add_argument('--eval', action = 'store_true',
						help = "evaluation mode")
	parser.add_argument('--epochs', type = int, default = 25,
						help = "training epochs")
	parser.add_argument('--save_name', type = str, default = 'model',
						help = "name for saved model")
	parser.add_argument('--test_set', action = 'store_true',
						help = "use test set for evaluation")
	parser.add_argument('--seed', type = int, default = 42,
						help = "random seed")
	parser.add_argument('--vgg', action = 'store_true',
						help = "use vgg features")
	# TODO
	parser.add_argument('--unsupervised', action = "store_true",
						help = "test unsupervised accuracy")
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_args()
	print(args)

	# set random seed
	torch.manual_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if torch.cuda.device_count() >= 1:
		print("Use {} gpus".format(torch.cuda.device_count()))

	wordEmbedding = read_word_embeddings("data/glove/glove.6B.300d.txt")
	if args.test_set:
		test_dset = Flickr30dataset(wordEmbedding, "test", vgg = args.vgg)
	else:
		test_dset = Flickr30dataset(wordEmbedding, "val", vgg = args.vgg)

	test_loader = DataLoader(test_dset, batch_size = args.batch, num_workers = 4, drop_last = True, shuffle = True)
	model = MATnet(wordEmbedding, vgg = args.vgg)
	if torch.cuda.is_available():
		print("CUDA available")
		model.cuda()

	if args.eval:
		score = (test_loader, model)
		print("untrained eval score:", score)
	else:
		train_dset = Flickr30dataset(wordEmbedding, "train", vgg = args.vgg)
		train_loader = DataLoader(train_dset, batch_size = args.batch, num_workers = 4, drop_last = True, shuffle = True)
		train(model, train_loader, test_loader, lr = args.lr, epochs = args.epochs)
		save_path = os.path.join("saved", args.save_name + '.pt')
		torch.save(model.cpu().state_dict(), save_path)
		print("save model to", save_path)
