import time
import warnings

import torch
import torch.nn as nn
from tqdm import tqdm

from utils.evaluator import Evaluator
from utils.utils import union_target

with warnings.catch_warnings():
	warnings.filterwarnings("ignore", category = FutureWarning)


def train(model, train_loader, test_loader, lr = 1e-4, epochs = 25):
	use_gpu = torch.cuda.is_available()

	model = model.float()

	optimizer = torch.optim.Adam(model.parameters(), lr = lr)
	ceLoss = nn.CrossEntropyLoss(reduction = "mean")

	print("---Before Training...")
	score = evaluate(test_loader, model)
	print("     eval score on test dataset:", score)

	for epoth in range(epochs):
		t = time.time()
		total_loss = 0
		correct_preds = 0
		all_preds = 0
		n_batches = 0

		for idx, labels, attrs, feature, query, bboxes, target_bboxes, num_obj, num_query in tqdm(train_loader):
			if (use_gpu):
				idx, labels, attrs, feature, query, bboxes, target_bboxes, num_obj, num_query = \
				idx.cuda(), labels.cuda(), attrs.cuda(), feature.cuda(), query.cuda(), bboxes.cuda(), target_bboxes.cuda(), num_obj.cuda(), num_query.cuda()

			n_batches += 1

			model.train(True)
			optimizer.zero_grad()

			probs, target, _ = model.forward(idx, query, labels, feature, attrs, bboxes, num_obj, num_query)

			target_pred = torch.argmax(target, dim = 1)  # [B]
			prediction = torch.argmax(probs, dim = 1)  # [all_querys]
			correct_preds += int(prediction.eq(target_pred).sum())
			all_preds += len(prediction)

			loss = ceLoss(probs, target_pred)
			total_loss += loss

			loss.backward()
			optimizer.step()

		t1 = time.time()
		print("--- EPOCH", epoth)
		print("     time:", t1 - t)
		print("     total loss:", total_loss.item() / n_batches)
		print("     supervised accuracy on training set: ", correct_preds / all_preds)

		t2 = time.time()
		# evaluate
		score, supacc = evaluate(test_loader, model)
		print("     eval time:", time.time() - t2)
		print("     supervised accuracy on test dataset:", supacc)
		print("     eval score on test dataset:", score)


def evaluate(test_loader, model):
	use_gpu = torch.cuda.is_available()

	correct_preds = 0
	all_preds = 0

	model = model.float()

	pred_bboxes_list = []
	target_bboxes_list = []
	num_query_list = []

	for idx, labels, attrs, feature, query, bboxes, target_bboxes, num_obj, num_query in tqdm(
			test_loader):
		if (use_gpu):
			idx, labels, attrs, feature, query, bboxes, target_bboxes, num_obj, num_query = \
			idx.cuda(), labels.cuda(), attrs.cuda(), feature.cuda(), query.cuda(), bboxes.cuda(), target_bboxes.cuda(), num_obj.cuda(), num_query.cuda()

		model.eval()

		pred_bboxes, pred_labels, probs, target = model.predict(
			idx, query, labels, feature, attrs, num_obj, num_query, bboxes)	# [B, 32, 4]

		# sup acc
		target_pred = torch.argmax(target, dim = 1)  # [B]
		prediction = torch.argmax(probs, dim = 1)  # [all_querys]
		correct_preds += int(prediction.eq(target_pred).sum())
		all_preds += len(prediction)

		pred_bboxes_list += pred_bboxes.cpu().tolist()
		target_bboxes_list += target_bboxes.cpu().tolist()
		num_query_list += num_query.cpu().tolist()

	score = evaluate_helper(pred_bboxes_list, target_bboxes_list, num_query_list)
	supacc = correct_preds / all_preds

	return score, supacc


def evaluate_helper(pred_bboxes, target_bboxes, num_query):
	evaluator = Evaluator()
	gtbox_list = []
	pred_list = []
	for pred, targ, nq in zip(pred_bboxes, target_bboxes, num_query):
		# ipred: [query, 5]
		# itarget: [query, 12, 4]
		if nq > 0:
			pred_list += pred[:nq]
			gtbox_list += union_target(targ[:nq])  # [query, 4]

	accuracy, _ = evaluator.evaluate(pred_list, gtbox_list)  # [query, 4]
	return accuracy
