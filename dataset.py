import os
import re
import h5py
import json
import utils
import torch
import torch.nn as nn
import numpy as np
from scipy import spatial
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import _pickle as cPickle
from xml.etree.ElementTree import parse
from tqdm import tqdm
from torch.nn import functional as F

class Indexer(object):
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        if (index not in self.ints_to_objs):
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):

        return self.index_of(object) != -1

    def index_of(self, object):

        if (object not in self.objs_to_ints):
            return -1
        else:
            return self.objs_to_ints[object]

    def add_and_get_index(self, object, add=True):

        if not add:
            return self.index_of(object)
        if (object not in self.objs_to_ints):
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]

class WordEmbeddings:
    def __init__(self, word_indexer, vectors):
        self.word_indexer = word_indexer
        self.vectors = vectors

    def get_embedding_length(self):
        return len(self.vectors[0])

    def get_embedding(self, word):
            word_idx = self.word_indexer.index_of(word)
            if word_idx != -1:
                return self.vectors[word_idx]
            else:
                return self.vectors[self.word_indexer.index_of("UNK")]

    def get_embeddings(self, word_list):
            emb_list = []
            for word in word_list:
                emb = self.get_embedding(word)
                emb_list.append(emb)
            return np.array(emb_list)

    def similarity(self, w1, w2):
        return 1 - spatial.distance.cosine(self.get_embedding(w1), self.get_embedding(w2))


class Flickr30dataset(Dataset):
    def __init__(self, wordEmbedding, name='train', dataroot='data/flickr30k/', use_bert = False, lite_bert=False, vgg=False):
        super(Flickr30dataset, self).__init__()
        self.use_bert = use_bert
        self.lite_bert = lite_bert
        self.vgg = vgg
        self.entries, self.img_id2idx = load_dataset(name, dataroot, use_bert = use_bert, lite_bert = lite_bert, vgg=self.vgg)
        # img_id2idx: dict {img_id -> val} val can be used to retrieve image or features
        self.indexer = wordEmbedding.word_indexer

        dataroot='data/flickr30k/'
        if self.vgg:
            print("load vgg features")
            h5_path = os.path.join(dataroot, 'vgg_dataset_pascal_vgbbox.hdf5')
        else:
            h5_path = os.path.join(dataroot, 'my%s.hdf5' % name)

        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('features'))
            self.pos_boxes = np.array(hf.get('pos_bboxes'))

        if use_bert or lite_bert:
            h5_path = os.path.join(dataroot, 'bert_feature_%s' % name)
            with h5py.File(h5_path, 'r') as hf:
                self.label_features = np.array(hf.get('label_features'))

    def __getitem__(self, index):
        '''
        return : labels, query, deteced_bboxes, number of querys

        labels: [K=64] index
        attrs: [K=64] index
        bboxes: [K=64, 5] index
        querys: [Q=32, len=12] index
        query_feats: [Q, dim]
        label_feats: [K, dim]
        target_bboxes: [Q=32, Boxes, 4] index
        '''
        K=100
        Q=32
        lens=12
        tok_lens = 12
        B=20
        bert_dim = 768

        entry = self.entries[index]
        imgid = entry['image']
        labels = entry['labels']
        querys = entry['query']
        idx = self.img_id2idx[int(imgid)]
        if not self.vgg:
            attrs = entry['attrs']

        label_toks = torch.tensor([0])
        sent_toks = torch.tensor([0])
        phrase_toks = torch.tensor([0])
        entity_indices = torch.tensor([0])
        entity_feats = torch.tensor([0])
        label_feats = torch.tensor([0])

        if self.use_bert:
            sent_toks = torch.tensor(entry['sent_toks'])
            # label_toks = torch.tensor(entry['label_toks'])
            entity_indices = torch.tensor(entry['entity_indices'])

            # print(sent_toks.shape, label_toks.shape, entity_indices.shape)
            '''
            if label_toks.size(0)<K:
                label_toks = F.pad(label_toks, (0, K-label_toks.size(0)))
            else:
                label_toks = label_toks[:K]
            '''
            if sent_toks.size(0)<K:
                sent_toks = F.pad(sent_toks, (0, K-sent_toks.size(0)))
            else:
                sent_toks = sent_toks[:K]

            if entity_indices.size(0)<Q:
                pad = nn.ZeroPad2d((0,0,0,Q-entity_indices.size(0)))
                entity_indices = pad(entity_indices)
            else:
                entity_indices = entity_indices[:Q]

            entity_feats = [[[0]*bert_dim for i in range(lens)] for j in range(Q)] # (torch.zeros(Q, lens, bert_dim)
            raw_entity_feats = entry['entity_feats'][:Q]
            for i, entity_feat in enumerate(raw_entity_feats):
                entity_feats[i][:min(len(entity_feat),lens)] =entity_feat[:min(len(entity_feat),lens)]
            entity_feats = torch.tensor(entity_feats)
            # print("entity_feats", entity_feats.shape) # [32, 12, 768]

            label_feats = torch.from_numpy(self.label_features[idx]).float() # [64, 768]
            pad = nn.ZeroPad2d((0,0,0,K-label_feats.size(0)))
            label_feats = pad(label_feats)

        elif self.lite_bert:
            '''
            label_toks = torch.tensor(entry['label_toks'])

            if label_toks.size(0)<K:
                label_toks = F.pad(label_toks, (0, K-label_toks.size(0)))
            else:
                label_toks = label_toks[:K]
            '''
            if sent_toks.size(0)<K:
                sent_toks = F.pad(sent_toks, (0, K-sent_toks.size(0)))
            else:
                sent_toks = sent_toks[:K]

            phrase_toks = entry['phrase_toks']
            new_phrase_toks=[] # [Q, tok_lens]
            for tok in phrase_toks:
                tok+=[0] * (tok_lens-len(tok))
                new_phrase_toks.append(tok[:tok_lens])
            while(len(new_phrase_toks)<Q):
                new_phrase_toks.append([0]*tok_lens)
            new_phrase_toks = new_phrase_toks[:Q]
            # print(new_phrase_toks)
            phrase_toks = torch.tensor(new_phrase_toks)
            # print("phrase",phrase_toks.shape) # [32, 12]

            idx = self.img_id2idx[int(imgid)]
            label_feats = torch.from_numpy(self.label_features[idx]).float() # [64, 768]
            pad = nn.ZeroPad2d((0,0,0,K-label_feats.size(0)))
            label_feats = pad(label_feats)

            '''
            label_feats = entry['label_features']
            query_feats = entry['query_features']
            query_feats = torch.tensor(query_feats)
            if query_feats.size(0)<Q:
                pad = nn.ZeroPad2d((0,0,0,Q-query_feats.size(0)))
                # print("q feature shape", query_feats.shape)
                query_feats = pad(query_feats)
            else:
                query_feats = query_feats[:Q]

            if label_feats.size(0)<K:
                # label_feats = label_feats.no_grad()
                pad = nn.ZeroPad2d((0,0,0,K-label_feats.size(0)))
                label_feats = pad(label_feats)
            else:
                label_feats = label_feats[:K]
            '''

        idx = self.img_id2idx[int(imgid)]# to retrieve pos in pos_box
        pos = self.pos_boxes[idx]

        # if self.vgg:
        #     feature = torch.tensor(entry["features"]).float()
        #     # print("feature shape", feature.shape)
        # else:

        feature = self.features[pos[0]:pos[1]]

        feature = torch.from_numpy(feature).float()

        if feature.size(0)<K:
            pad = nn.ZeroPad2d((0,0,0,K-feature.size(0)))
            feature = pad(feature)
        else:
            feature = feature[:K]

        num_obj = min(len(labels), K)
        num_query = min(len(querys),Q)

        # print(num_obj)

        labels_idx = [0]* K
        labels_idx[:num_obj] = [max(self.indexer.index_of(re.split(' ,',w)[-1]), 1) for w in labels]
        labels_idx = labels_idx[:K]

        if self.vgg:
            attr_idx = [0]*K
        else:
            attr_idx = [0]*K
            attr_idx[:num_obj] = [max(self.indexer.index_of(w), 1) for w in attrs]
            attr_idx = attr_idx[:K]


        querys_idx=[]
        for q in querys:
            q = q.lower().split()
            lis=[0]*lens
            for i in range(min(len(q), lens)):
                lis[i] = max(self.indexer.index_of(q[i]), 1)
            querys_idx.append(lis)
        while(len(querys_idx)<Q):
            querys_idx.append([0]*lens)
        querys_idx = querys_idx[:Q]


        bboxes = entry['detected_bboxes'] # [x1,y1,x2,y2]
        target_bboxes = entry['target_bboxes']

        padbox = [0,0,0,0]

        while(len(bboxes)<K):
            bboxes.append(padbox)
        bboxes = bboxes[:K]

        bboxes = torch.tensor(bboxes)
        area = (bboxes[...,3]-bboxes[...,1])*(bboxes[...,2]-bboxes[...,0])
        # print(area[:3], area.shape)
        bboxes = torch.cat((bboxes, area.unsqueeze_(-1)), -1)
        # print(bboxes.shape)

        for bbox in target_bboxes:
            while(len(bbox)<B):
                bbox.append(padbox)
        target_bboxes = [b[:B] for b in target_bboxes]
        padline = [padbox for i in range(B)]
        while(len(target_bboxes)<Q):
            target_bboxes.append(padline)
        target_bboxes = target_bboxes[:Q]

        assert len(labels_idx)==K
        assert len(attr_idx)==K
        assert len(bboxes)==K
        assert len(querys_idx)==Q
        assert len(target_bboxes)==Q

        return torch.tensor(int(imgid)), torch.tensor(labels_idx), torch.tensor(attr_idx), feature, torch.tensor(querys_idx), label_feats, sent_toks, phrase_toks, entity_indices, entity_feats, bboxes, torch.tensor(target_bboxes), torch.tensor(num_obj), torch.tensor(num_query)
        # return torch.tensor(int(imgid)), torch.tensor(labels_idx), torch.tensor(attr_idx), torch.tensor(querys_idx), bboxes, torch.tensor(target_bboxes), torch.tensor(num_obj), torch.tensor(num_query)

    def __len__(self):
        return len(self.entries)


def read_word_embeddings(embeddings_file: str) -> WordEmbeddings:
    f = open(embeddings_file)
    word_indexer = Indexer()
    vectors = []
    
    word_indexer.add_and_get_index("PAD")

    word_indexer.add_and_get_index("UNK")
    for line in f:
        if line.strip() != "":
            space_idx = line.find(' ')
            word = line[:space_idx]
            numbers = line[space_idx+1:]
            float_numbers = [float(number_str) for number_str in numbers.split()]
            vector = np.array(float_numbers)
            word_indexer.add_and_get_index(word)

            if len(vectors) == 0:
                vectors.append(np.zeros(vector.shape[0]))
                vectors.append(np.zeros(vector.shape[0]))
            vectors.append(vector)
    f.close()
    print("Read in " + repr(len(word_indexer)) + " vectors of size " + repr(vectors[0].shape[0]))

    return WordEmbeddings(word_indexer, np.array(vectors))


def load_train_flickr30k(dataroot, img_id2idx, obj_detection, bert_feature_dict, use_bert = False, lite_bert=False, vgg=False):
    """Load entries

    img_id2idx: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test-dev2015', test2015'
    """
    pattern_phrase = r'\[(.*?)\]'
    pattern_no = r'\/EN\#(\d+)'
    missing_entity_count = dict()
    entries = []

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained('bert-base-uncased')

    for image_id, idx in tqdm(img_id2idx.items()):

        phrase_file = os.path.join(dataroot, 'Flickr30kEntities/Sentences/%d.txt' % image_id)
        anno_file = os.path.join(dataroot, 'Flickr30kEntities/Annotations/%d.xml' % image_id)

        with open(phrase_file, 'r', encoding='utf-8') as f:
            sents = [x.strip() for x in f]

        # Parse Annotation
        root = parse(anno_file).getroot()
        obj_elems = root.findall('./object')

        target_bboxes_dict = {}
        entitywbox=[]

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

        # image_id = str(image_id)
        # bboxes = obj_detection[image_id]['bboxes']
        # labels = obj_detection[image_id]['classes'] # [B, 4]
        # attrs = obj_detection[image_id]['attrs']
        if vgg:
            image_id = str(image_id)
            bboxes = obj_detection[image_id]['bboxes']
            labels = obj_detection[image_id]['classes'] # [B, 4]
            # features =  obj_detection[image_id]['features']
        else:
            image_id = str(image_id)
            bboxes = obj_detection[image_id]['bboxes']
            labels = obj_detection[image_id]['classes'] # [B, 4]
            attrs = obj_detection[image_id]['attrs']

        assert(len(bboxes)==len(labels))

        # assert(len(labels)==len(attrs))

        # class -> bboxes
        '''
        detected_bboxes_dict = {}
        for label, box, attr in zip(labels, bboxes, attrs):
            box = [int(round(x)) for x in box]
            if label in detected_bboxes_dict:
                detected_bboxes_dict[label].append(box)
            else:
                detected_bboxes_dict[label]=[box]

        detected_bboxes = []
        unique_labels = list(set(labels))
        for l in unique_labels:
            detected_bboxes.append(detected_bboxes_dict[l])
        '''

        if use_bert:
            # print(labels, len(labels))

            # for label in labels:
            # label_toks = tokenizer.encode(labels)[1:-1] # len = labels

            # with torch.no_grad():
            #     outputs = model(input_ids)
            # label_feats = outputs[0][0][1:-1]
            # print(label_feats.shape)
            bert_feature = bert_feature_dict[image_id]

            sent_entries=[]
            entity_feats = []

            # for sent_id, sent in enumerate(sents):
            #     sentence = utils.remove_annotations(sent)
            #     entities = re.findall(pattern_phrase, sent)
            #     entity_ids = []
            #     entity_types = []
            #     entity_names = []
            #     entity_indices = []
            #     target_bboxes = []
            #     query=[]
            for sent_entry in bert_feature:
                sent_feat = sent_entry["sent_feature"]
                entities = sent_entry["entities"]
                # entity_indices = sent_entry["entity_indices"]# [[start, end]]
                entity_ids = sent_entry["entity_ids"]
                sent_toks = sent_entry["sentence"]
                target_bboxes = []
                entity_indices = []
                query=[]
                # assert len(entity_indices)==len(entities)
                assert len(sent_feat) == len(sent_toks)

                # sent_toks = tokenizer.encode(sentence)[1:-1]

                for phrase, entity_id in zip(entities, entity_ids):
                    phra_toks =  tokenizer.encode(phrase)[1:-1]
                    entity_idx = utils.find_sublist(sent_toks, phra_toks)
                    entity_feat = sent_feat[entity_idx[0]:entity_idx[1]]
                    # print(entity_idx)
                    # print("entity feature", phrase, len(entity_feat)) # [len, dim]

                    # info, phrase = entity.split(' ', 1)

                    # entity_id = int(re.findall(pattern_no, info)[0])
                    # entity_type = info.split('/')[2:]


                    # print(sent_toks, phrase, entity_idx)
                    # input()

                    if not entity_id in target_bboxes_dict:
                        continue
                    assert 0 < entity_id

                    target_bboxes.append(target_bboxes_dict[entity_id])
                    query.append(phrase)
                    # entity_names.append(phrase)
                    # entity_ids.append(entity_id)
                    # entity_types.append(entity_type)
                    entity_indices.append(entity_idx)
                    entity_feats.append(entity_feat)
                    # assert len(entity_names) == len(entity_ids)

                if len(entity_indices)==0 or len(sent_toks)==0:
                    continue

                entry = {
                    'image'          : image_id,
                    'target_bboxes'  : target_bboxes, # in order of entities
                    "detected_bboxes" : bboxes, # list, in order of labels
                    'labels' : labels,
                    'attrs': attrs,
                    'query' : query,
                    # 'label_toks': label_toks,
                    'sent_toks': sent_toks,
                    'entity_feats': entity_feats, # [querys, len_query, dim]
                    'entity_indices':entity_indices
                    }
                # print(entry)
                entries.append(entry)

        elif lite_bert:
            # label_toks = tokenizer.encode(labels)
            sent_entries=[]
            entity_feats = []

            for sent_id, sent in enumerate(sents):
                sentence = utils.remove_annotations(sent)
                sent_toks = tokenizer.encode(sentence)

                entities = re.findall(pattern_phrase, sent)
                entity_ids = []
                entity_types = []
                # entity_names = []
                entity_indices = []
                target_bboxes = []
                phrase_toks = []
                query=[]

                for i, entity in enumerate(entities):
                    info, phrase = entity.split(' ', 1)
                    phra_toks =  tokenizer.encode(phrase)

                    entity_id = int(re.findall(pattern_no, info)[0])
                    entity_type = info.split('/')[2:]

                    if not entity_id in target_bboxes_dict:
                        if entity_id >= 0:
                            missing_entity_count[entity_type[0]] = missing_entity_count.get(entity_type[0], 0) + 1
                        continue

                    assert 0 < entity_id

                    # in entity order
                    # entity_feat = sent_feat[entity_id[0]] # 取第一个的feature作为整个phrase的feature
                    target_bboxes.append(target_bboxes_dict[entity_id])
                    query.append(phrase)
                    phrase_toks.append(phra_toks)
                    # entity_names.append(phrase)
                    entity_ids.append(entity_id)
                    entity_types.append(entity_type)

                    # entity_indices.append(entity_idx)

                if 0 == len(entity_ids):
                    continue

                entry = {
                    'image'          : image_id,
                    'target_bboxes'  : target_bboxes, # in order of entities
                    "detected_bboxes" : bboxes, # list, in order of labels
                    'labels' : labels,
                    'attrs': attrs,
                    'query' : query,
                    # 'label_toks': label_toks,
                    'phrase_toks': phrase_toks, # [querys, len_query]
                    'sentence_toks':sent_toks
                    }
                # print(entry)
                entries.append(entry)
        else:
            # Parse Sentence
            sent_entries=[]
            for sent_id, sent in enumerate(sents):
                sentence = utils.remove_annotations(sent)
                entities = re.findall(pattern_phrase, sent)
                entity_ids = []
                entity_types = []
                entity_names = []
                entity_indices = []
                target_bboxes = []
                query=[]

                for i, entity in enumerate(entities):
                    info, phrase = entity.split(' ', 1)
                    entity_id = int(re.findall(pattern_no, info)[0])
                    entity_type = info.split('/')[2:]
                    entity_idx = utils.find_sublist(sentence.split(' '), phrase.split(' '))

                    # assert 0 <= entity_idx

                    if not entity_id in target_bboxes_dict:
                        if entity_id >= 0:
                            missing_entity_count[entity_type[0]] = missing_entity_count.get(entity_type[0], 0) + 1
                        continue

                    assert 0 < entity_id

                    # in entity order
                    # entity_feat = sent_feat[entity_id[0]] # 取第一个的feature作为整个phrase的feature
                    target_bboxes.append(target_bboxes_dict[entity_id])
                    query.append(phrase)
                    # query_feats.append(entity_feat)

                    entity_names.append(phrase)
                    entity_ids.append(entity_id)
                    entity_types.append(entity_type)
                    assert len(entity_names) == len(entity_ids)

                    entity_indices.append(entity_idx)

                if 0 == len(entity_ids):
                    continue

                if vgg:
                    entry = {
                        'image'          : image_id,
                        'target_bboxes'  : target_bboxes, # in order of entities
                        "detected_bboxes" : bboxes, # list, in order of labels
                        'labels' : labels,
                        'query' : query
                        }
                    entries.append(entry)
                else:
                    entry = {
                        'image'          : image_id,
                        'target_bboxes'  : target_bboxes, # in order of entities
                        "detected_bboxes" : bboxes, # list, in order of labels
                        'labels' : labels,
                        'attrs': attrs,
                        'query' : query
                        }
                    entries.append(entry)

    print("Load Down!")
    return entries


def load_dataset(name='train', dataroot='data/flickr30k/', use_bert = False, lite_bert=False, vgg=False):
    '''
    "xxxx":{
        bboxes:
        classes:
        attrs:
    }
    '''
    bert_feature_dict = None
    # obj_detection_dict = json.load(open("data/%s_dataset.json"%name, "r"))
    if vgg:
        print("load vgg object det dict")
        obj_detection_dict = json.load(open("data/obj_detection_vgg_pascal_vgbbox.json", "r"))
        # obj_detection_dict = gen_obj_dict(obj_detection)
        img_id2idx = cPickle.load(open(os.path.join(dataroot, 'vgg_pascal_vgbbox_%s_imgid2idx.pkl' % name), 'rb'))
    else:
        obj_detection_dict = json.load(open("data/%s_dataset.json"%name, "r"))
        img_id2idx = cPickle.load(open(os.path.join(dataroot, '%s_imgid2idx.pkl' % name), 'rb'))
        if use_bert:
            bert_feature_dict = json.load(open("bert_feature_%s.json"%name, "r"))


    # h5_path = os.path.join(dataroot, '%s.hdf5' % name)

    entries = load_train_flickr30k(dataroot, img_id2idx, obj_detection_dict, bert_feature_dict, use_bert = use_bert, lite_bert = lite_bert, vgg=vgg)
    print("load flickr30k data successfully.")
    return entries, img_id2idx


# generate object detection dictionary
def gen_obj_dict(obj_detection):
    obj_detect_dict={}
    for img in obj_detection:
        try:
            img_id = int(img["image"].split('.')[0])
        except:
            continue

        # print(img_id)
        tmp={"bboxes":[], "classes":[], "scores":[], "features":[]}
        for dic in img['objects']:
            bbox = [int(i) for i in dic["bbox"][1:-1].split(',')]
            tmp["bboxes"].append(bbox)
            tmp["classes"].append(dic["class"])
            tmp["scores"].append(dic["score"])
            tmp["features"].append(dic["feature"])

        obj_detect_dict[img_id]=tmp
    return obj_detect_dict


if __name__=="__main__":
    name = "test"
    obj_detection = json.load(open("data/obj_detection_0.1.json", "r"))
    obj_detection_dict = gen_obj_dict(obj_detection)
    with open("data/%s_detect_dict.json"%name, "w") as f:
        json.dump(obj_detection_dict, f)
