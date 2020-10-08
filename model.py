import random

import numpy as np
import torch
import torch.nn as nn


class MATnet(nn.Module):
	def __init__(self, wordvec, emb_dim = 300, feature_dim = 2048, bert_dim = 768, vgg = False):
		super(MATnet, self).__init__()

		if vgg:
			feature_dim = 4096

		self.bert_dim = bert_dim
		self.eps = 1e-6
		self.wordemb = wordvec
		self.indexer = wordvec.word_indexer
		self.wv = nn.Embedding.from_pretrained(torch.from_numpy(wordvec.vectors), freeze = False)

		self.linear_p = nn.Linear(emb_dim, emb_dim)
		self.linear_a = nn.Linear(emb_dim, emb_dim)
		self.linear_f = nn.Linear(feature_dim, emb_dim)
		self.linear_mini = nn.Linear(emb_dim, emb_dim)

		self.relu = nn.ReLU()
		self.softmax = nn.Softmax(dim = -1)
		self.cosSim = nn.CosineSimilarity(dim = -1, eps = self.eps)

		self.linear_p.weight.data = torch.eye(emb_dim)
		self.linear_a.weight.data = torch.zeros(emb_dim, emb_dim)
		self.linear_f.weight.data = torch.zeros(emb_dim, feature_dim)

	def _encode(self, query, label, feature, attrs):
		"""
		:param query: query phrases [B, queries, words]
		:param label: object labels, predicted by the detector [B, objects]
		:param feature: object features, predicted by the detector [B, objects, feature_dim]
		:param attrs: object attributes, predicted by the detector [B, objects]

		:return: 	q_emb[B, queries, words, dim] for query word embedding;
					k_emb[B, objects, dim] for object embedding
		"""

		q_emb = self.wv(query)
		k_emb = self.wv(label)

		f_emb = self.linear_f(feature)
		k_emb += f_emb
		# k_emb += f_emb + a_emb

		return q_emb, k_emb

	def _encode_pk(self, query, label, feature, attrs):
		"""
		:param query: query phrases [B, queries, words]
		:param label: object labels, predicted by the detector [B, objects]
		:param feature: object features, predicted by the detector [B, objects, feature_dim]
		:param attrs: object attributes, predicted by the detector [B, objects]

		:return:	p_emb[B, queries, dim] for phrase embedding
					k_emb[B, objects, dim] for object embedding
		"""
		eps = 1e-5

		q_emb, k_emb = self._encode(query, label, feature, attrs)  # [B, querys, Q, dim] & [B, K, dim]

		# q_emb [B, querys, Q, dim]
		scale = 1.0 / np.sqrt(k_emb.size(-1))
		att = torch.einsum('byqd,bkd ->byqk', q_emb, k_emb)
		att = self.softmax(att.mul_(scale))  # [B, querys, Q, K]

		q_max_att = torch.max(att, dim = 3).values  # [B, querys, Q]
		q_max_norm_att = self.softmax(q_max_att)

		# attended
		p_emb = torch.einsum('byq,byqd -> byd', q_max_norm_att, q_emb)  # [B, querys, dim]

		# average
		# len_query = torch.sum((query==0), dim=-1) # [B, querys]
		# len_query = len_query.unsqueeze(-1).expand(query.size(0), query.size(1), q_emb.size(3))# [B, querys, dim]
		# p_emb = torch.sum(q_emb, dim=-2) / (len_query+eps)

		p_emb = self.linear_p(p_emb) + eps * self.linear_mini(p_emb)

		return p_emb, k_emb

	def forward(self, idx, query, label, feature, attrs, bboxes, num_obj, num_query):
		"""

		:param idx:
		:param query: [B, all_query=32, Q=12] pad with 0
		:param label: [B, K=64] pad with 0
		:param feature: B, K, feature_dim]
		:param attrs: [B, K=64] pad with 0
		:param bboxes: [B, K, 4]
		:param num_obj:
		:param num_query:
		:return:
		"""

		p_emb, k_emb = self._encode_pk(query, label, feature, attrs)

		attmap = torch.einsum('avd, bqd -> baqv', k_emb,
							  p_emb)  # [B1, K, dim] x [B2, querys, dim] => [B2, B1, querys, K]

		attmap_sm = self.softmax(attmap)  # [B2, B1, querys, K]
		att_obj_sum = torch.sum(attmap_sm, dim = -2)  # [B2, B1, K]

		maxatt, _ = attmap.max(dim = -1)  # [B1, B2, querys]: B1th sentence to B2th image
		logits = torch.sum(maxatt, dim = -1) \
			.div(num_query.unsqueeze(1)
				 .expand(maxatt.size(0), maxatt.size(1)))  # [B1, B2]: B1th sentence to B2th image

		n_obj = int(query.size(0))
		target = torch.eye(n_obj).cuda()  # [b, b]
		return logits, target, att_obj_sum

	def forward_embedding(self, query, label, num_obj, num_query):
		"""
		:return: anchor, positive, negative
		"""

		q_emb, k_emb = self._encode(query, label)  # q: [B, querys, Q, dim] , [B, K, dim]

		scale = 1.0 / np.sqrt(q_emb.size(3))
		att = torch.einsum('byqd,bkd ->byqk', q_emb, k_emb)
		mask = (att == 0)
		att.masked_fill_(mask, -float('inf'))
		att = self.softmax(att.mul_(scale))  # [B, querys, Q, K]
		att[torch.isnan(att)] = 0

		# embedding of every phrase with regard to the labels
		q_max_att = torch.max(att, dim = 3).values  # [B, querys, Q]
		q_max_norm_att = q_max_att.div(q_max_att.sum(dim = 2, keepdim = True))  # [B, querys, Q]
		p_emb = torch.einsum('byq,byqd -> byd', q_max_norm_att, q_emb)  # [B, querys, dim]

		i_att = torch.einsum('bkd, byd -> byk', k_emb,
							 p_emb)  # [B, querys, K, dim] x [B, querys, dim] => [B, querys, K]
		i_emb = torch.einsum('byk, bykd -> byd', i_att,
							 k_emb)  # [B, querys, K] x [B, queys, K, dim] -> [B, querys, dim]

		random_idx = [i for i in range(i_emb.size(0))]
		random.shuffle(random_idx)
		shuffle_i_emb = i_emb[random_idx]

		anchor = torch.cat(tuple(p[:n, :] for p, n in zip(p_emb, num_query)), 0)  # [total_query, dim]
		positive = torch.cat(tuple(i[:n, :] for i, n in zip(i_emb, num_query)), 0)
		negative = torch.cat(tuple(i[:n, :] for i, n in zip(shuffle_i_emb, num_query)), 0)

		return anchor, positive, negative

	def predict(self, idx, query, label, feature, attrs, num_obj, num_query, bboxes = None):
		p_emb, k_emb = self._encode_pk(query, label, feature, attrs)

		i_att = torch.einsum('bkd, byd -> byk', k_emb, p_emb)  # [B, K, dim] x [B, querys, dim] => [B, querys, K]

		prediction = torch.argmax(i_att, dim = -1)  # [B, querys]
		maxval, _ = i_att.max(dim = 2, keepdim = True)
		predictions = (i_att == maxval)  # [B, querys, K]

		attmap = torch.einsum('avd, bqd -> baqv', k_emb, p_emb)
		maxatt, _ = attmap.max(dim = -1)  # [B1, B2, querys]: B1th sentence to B2th image
		logits = torch.sum(maxatt, dim = -1).div(num_query.unsqueeze(1).expand(maxatt.size(0), maxatt.size(
			1)))  # [B1, B2]: B1th sentence to B2th image
		n_obj = int(query.size(0))
		target = torch.eye(n_obj).cuda()  # [b, b]

		pred_labels = []
		for i in range(prediction.size(0)):
			for j in range(prediction.size(1)):
				pred_label = self.indexer.get_object(int(label[i, prediction[i, j]]))
				pred_labels.append(pred_label)

		pred_bboxes = []
		for i in range(predictions.size(0)):
			# predictions[i]:querys x K
			select_box = bboxes[i].unsqueeze(0).expand(predictions.size(1), predictions.size(2),
													   5).long()  # querys x K x 5
			select_mask = predictions[i].unsqueeze(-1).long()  # querys x K x 1

			avail_box = select_box * select_mask  # # querys x K x 5
			_, maxidx = avail_box.max(dim = 1)  # querys x 5

			# maxidx[:, -1] -- [querys] location of the max bbox
			bbox = select_box[torch.arange(select_box.size(0)), maxidx[:, -1]]
			pred_bboxes.append(bbox[:, :4])

		pred_bboxes = torch.stack(pred_bboxes)  # [B, querys, 4]
		return pred_bboxes, pred_labels, logits, target
