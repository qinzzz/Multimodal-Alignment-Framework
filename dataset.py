import _pickle as cPickle
import json
import os
import re
from xml.etree.ElementTree import parse

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import utils


class Flickr30dataset(Dataset):
	def __init__(self, wordEmbedding, name = 'train', dataroot = 'data/flickr30k/', vgg = False):
		super(Flickr30dataset, self).__init__()
		self.vgg = vgg
		self.entries, self.img_id2idx = load_dataset(name, dataroot, vgg = self.vgg)
		# img_id2idx: dict {img_id -> val} val can be used to retrieve image or features
		self.indexer = wordEmbedding.word_indexer

		if self.vgg:
			print("load vgg features")
			h5_path = os.path.join(dataroot, 'vgg_dataset_pascal_vgbbox.hdf5')
		else:
			h5_path = os.path.join(dataroot, '%s_features_compress.hdf5' % name)

		with h5py.File(h5_path, 'r') as hf:
			self.features = np.array(hf.get('features'))
			self.pos_boxes = np.array(hf.get('pos_bboxes'))

	def _get_entry(self, index):
		entry = self.entries[index]
		if self.vgg:
			attrs = []
		else:
			attrs = []
			# attrs = entry['attrs']
		return entry['image'], entry['labels'], entry['query'], attrs, entry['detected_bboxes'], entry['target_bboxes']

	def __getitem__(self, index):
		'''
		:return: labels, query, deteced_bboxes, number of querys

		labels: [K=64] index
		attrs: [K=64] index
		bboxes: [K=64, 5] index
		querys: [Q=32, len=12] index
		query_feats: [Q, dim]
		label_feats: [K, dim]
		target_bboxes: [Q=32, Boxes, 4] index
		'''

		K = 100
		Q = 32
		lens = 12
		B = 20

		imgid, labels, querys, attrs, bboxes, target_bboxes = self._get_entry(index)

		idx = self.img_id2idx[int(imgid)]  # to retrieve pos in pos_box
		pos = self.pos_boxes[idx]

		feature = torch.from_numpy(self.features[pos[0]:pos[1]]).float()

		if feature.size(0) < K:
			pad = nn.ZeroPad2d((0, 0, 0, K - feature.size(0)))
			feature = pad(feature)
		else:
			feature = feature[:K]

		num_obj = min(len(labels), K)
		num_query = min(len(querys), Q)

		labels_idx = [0] * K
		labels_idx[:num_obj] = [max(self.indexer.index_of(re.split(' ,', w)[-1]), 1) for w in labels]
		labels_idx = labels_idx[:K]

		if self.vgg:
			attr_idx = [0] * K
		else:
			attr_idx = [0] * K
			# attr_idx[:num_obj] = [max(self.indexer.index_of(w), 1) for w in attrs]
			# attr_idx = attr_idx[:K]

		querys_idx = []
		for q in querys:
			q = q.lower().split()
			lis = [0] * lens
			for i in range(min(len(q), lens)):
				lis[i] = max(self.indexer.index_of(q[i]), 1)
			querys_idx.append(lis)
		while (len(querys_idx) < Q):
			querys_idx.append([0] * lens)
		querys_idx = querys_idx[:Q]

		padbox = [0, 0, 0, 0]

		while (len(bboxes) < K):
			bboxes.append(padbox)
		bboxes = bboxes[:K]

		bboxes = torch.tensor(bboxes)
		area = (bboxes[..., 3] - bboxes[..., 1]) * (bboxes[..., 2] - bboxes[..., 0])
		bboxes = torch.cat((bboxes, area.unsqueeze_(-1)), -1)

		for bbox in target_bboxes:
			while (len(bbox) < B):
				bbox.append(padbox)
		target_bboxes = [b[:B] for b in target_bboxes]
		padline = [padbox for i in range(B)]
		while (len(target_bboxes) < Q):
			target_bboxes.append(padline)
		target_bboxes = target_bboxes[:Q]

		assert len(labels_idx) == K
		assert len(bboxes) == K
		assert len(querys_idx) == Q
		assert len(target_bboxes) == Q

		return torch.tensor(int(imgid)), torch.tensor(labels_idx), torch.tensor(attr_idx), feature, \
			   torch.tensor(querys_idx), bboxes, torch.tensor(target_bboxes), torch.tensor(num_obj), torch.tensor(
			num_query)

	def __len__(self):
		return len(self.entries)


def load_train_flickr30k(dataroot, img_id2idx, obj_detection, vgg = False):
	"""Load entries

	img_id2idx: dict {img_id -> val} val can be used to retrieve image or features
	dataroot: root path of dataset
	name: 'train', 'val', 'test-dev2015', test2015'
	"""
	pattern_phrase = r'\[(.*?)\]'
	pattern_no = r'\/EN\#(\d+)'
	missing_entity_count = dict()
	entries = []

	for image_id, idx in tqdm(img_id2idx.items()):

		phrase_file = os.path.join(dataroot, 'Flickr30kEntities/Sentences/%d.txt' % image_id)
		anno_file = os.path.join(dataroot, 'Flickr30kEntities/Annotations/%d.xml' % image_id)

		with open(phrase_file, 'r', encoding = 'utf-8') as f:
			sents = [x.strip() for x in f]

		# Parse Annotation
		root = parse(anno_file).getroot()
		obj_elems = root.findall('./object')

		target_bboxes_dict = {}
		entitywbox = []

		for elem in obj_elems:
			if elem.find('bndbox') == None or len(elem.find('bndbox')) == 0:
				continue
			left = int(elem.findtext('./bndbox/xmin'))
			top = int(elem.findtext('./bndbox/ymin'))
			right = int(elem.findtext('./bndbox/xmax'))
			bottom = int(elem.findtext('./bndbox/ymax'))
			assert 0 < left and 0 < top

			for name in elem.findall('name'):
				entity_id = int(name.text)
				assert 0 < entity_id
				entitywbox.append(entity_id)
				if not entity_id in target_bboxes_dict.keys():
					target_bboxes_dict[entity_id] = []
				target_bboxes_dict[entity_id].append([left, top, right, bottom])

		if vgg:
			image_id = str(image_id)
			bboxes = obj_detection[image_id]['bboxes']
			labels = obj_detection[image_id]['classes']  # [B, 4]
		# features =  obj_detection[image_id]['features']
		else:
			image_id = str(image_id)
			bboxes = obj_detection[image_id]['bboxes']
			labels = obj_detection[image_id]['classes']  # [B, 4]
			attrs = obj_detection[image_id]['attrs'] if 'attrs' in obj_detection[image_id].keys() else []

		assert (len(bboxes) == len(labels))

		# Parse Sentence
		for sent_id, sent in enumerate(sents):
			sentence = utils.remove_annotations(sent)
			entities = re.findall(pattern_phrase, sent)
			entity_ids = []
			entity_types = []
			entity_names = []
			entity_indices = []
			target_bboxes = []
			query = []

			for i, entity in enumerate(entities):
				info, phrase = entity.split(' ', 1)
				entity_id = int(re.findall(pattern_no, info)[0])
				entity_type = info.split('/')[2:]
				entity_idx = utils.find_sublist(sentence.split(' '), phrase.split(' '))

				if not entity_id in target_bboxes_dict:
					if entity_id >= 0:
						missing_entity_count[entity_type[0]] = missing_entity_count.get(entity_type[0], 0) + 1
					continue

				assert 0 < entity_id

				# in entity order
				# entity_feat = sent_feat[entity_id[0]] # 取第一个的feature作为整个phrase的feature
				target_bboxes.append(target_bboxes_dict[entity_id])
				query.append(phrase)

				entity_names.append(phrase)
				entity_ids.append(entity_id)
				entity_types.append(entity_type)
				assert len(entity_names) == len(entity_ids)

				entity_indices.append(entity_idx)

			if 0 == len(entity_ids):
				continue

			if vgg:
				entry = {
					'image': image_id,
					'target_bboxes': target_bboxes,  # in order of entities
					"detected_bboxes": bboxes,  # list, in order of labels
					'labels': labels,
					'query': query
				}
				entries.append(entry)
			else:
				entry = {
					'image': image_id,
					'target_bboxes': target_bboxes,  # in order of entities
					"detected_bboxes": bboxes,  # list, in order of labels
					'labels': labels,
					'attrs': attrs,
					'query': query
				}
				entries.append(entry)

	print("Load Done!")
	return entries


def load_dataset(name = 'train', dataroot = 'data/flickr30k/', vgg = False):
	if vgg:
		print("load vgg object det dict")
		obj_detection_dict = json.load(open("data/obj_detection_vgg_pascal_vgbbox.json", "r"))
		img_id2idx = cPickle.load(open(os.path.join(dataroot, 'vgg_pascal_vgbbox_%s_imgid2idx.pkl' % name), 'rb'))
	else:
		obj_detection_dict = json.load(open("data/%s_detection_dict.json" % name, "r"))
		img_id2idx = cPickle.load(open(os.path.join(dataroot, '%s_imgid2idx.pkl' % name), 'rb'))

	entries = load_train_flickr30k(dataroot, img_id2idx, obj_detection_dict, vgg = vgg)
	print("load flickr30k dataset successfully.")
	return entries, img_id2idx


def gen_obj_dict(obj_detection):
	"""
	generate object detection dictionary
	"""
	obj_detect_dict = {}
	for img in obj_detection:
		try:
			img_id = int(img["image"].split('.')[0])
		except:
			continue

		tmp = {"bboxes": [], "classes": [], "scores": [], "features": []}
		for dic in img['objects']:
			bbox = [int(i) for i in dic["bbox"][1:-1].split(',')]
			tmp["bboxes"].append(bbox)
			tmp["classes"].append(dic["class"])
			tmp["scores"].append(dic["score"])
			tmp["features"].append(dic["feature"])

		obj_detect_dict[img_id] = tmp
	return obj_detect_dict
