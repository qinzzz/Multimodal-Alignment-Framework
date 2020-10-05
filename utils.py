"""
This code is extended from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
from __future__ import print_function

import errno
import os
import re
import collections
import numpy as np
import operator
import functools
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import string_classes
from xml.etree.ElementTree import parse
from torch.utils.data.dataloader import default_collate


EPS = 1e-7

class Evaluator(object):
    """ Utility class for evaluating phrase localization
    """
    def __init__(self):
        pass;

    def compute_iou(self, predictedBoxList, gtBoxList):
        """ Computes list of areas of IoU for all given instances.
        Parameters
        ----------
        predictedBoxList : list
            [[x,y,w,h],[x,y,w,h],[x,y,w,h],...]
            List of predicted bounding box instances [x,y,w,h] for each query instance.
            x and y are the (x,y) coordinates of the top-left of the bounding box for the query term
            w and h are the width and height of the bounding box for the query test
        gtBoxList : list
            Same as above, but for ground truth bounding boxes. Must be the same length as predictedBoxList
        Returns
        -------
        iouList : list(float)
            The area of IoU for each prediction in predictedBoxList
        """

        assert len(predictedBoxList) == len(gtBoxList), \
            "The list of predicted bounding boxes ({}) should be the same size as the list of ground truth bounding boxes ({}).".format(len(predictedBoxList), len(gtBoxList));

        # compute iou for each bounding box instance
        iouList = [];
        for (box1, box2) in zip(gtBoxList, predictedBoxList):
            iou = self._iou(box1, box2);
            iouList.append(iou);

        return iouList;


    def accuracy(self, iouList, iouThreshold=0.5):
        """ Computes the overall accuracy from a given list of iou and an iouThreshold
        Parameters
        ----------
        iouList : list(float)
            List of areas of IoU
        iouThreshold : float
            The threshold for the IoU, such that item in iouList is True if IoU >= iouThreshold.
        Returns
        -------
        accuracy : float
            Overall accuracy (or recall to be more precise). 
            Proportion of predicted boxes that overlaps with the ground truth boxes by an IoU of >= iouThreshold.

        """

        matches = len([1 for iou in iouList if iou >= iouThreshold]);
        accuracy = matches * 1.0 / len(iouList);
        return accuracy;


    def evaluate(self, predictedBoxList, gtBoxList, iouThreshold=0.5):
        """ Computes the overall accuracy and list of areas of IoU for each test instance.
        Parameters
        -------
        predictedBoxList : list
            [[x,y,w,h],[x,y,w,h],[x,y,w,h],...]
            List of predicted bounding box instances [x,y,w,h] for each query instance.
            x and y are the (x,y) coordinates of the top-left of the bounding box for the query term
            w and h are the width and height of the bounding box for the query test
        gtBoxList : list
            Same as above, but for ground truth bounding boxes. Must be the same length as predictedBoxList
        iouThreshold : float
            The threshold for the IoU, such that two bounding boxes are considered overlapping when IoU >= iouThreshold.
        Returns
        -------
        accuracy : float
            Overall accuracy (or recall to be more precise). 
            Proportion of predicted boxes that overlaps with the ground truth boxes by an IoU of >= iouThreshold.

        iouList : list(float)
            The area of IoU for each prediction in predictedBoxList
        """

        iouList = self.compute_iou(predictedBoxList, gtBoxList);
        accuracy = self.accuracy(iouList, iouThreshold);
        return (accuracy, iouList);


    def evaluate_perclass(self, predictedBoxList, gtBoxList, boxCategoriesList, iouThreshold=0.5):
        """ Computes the overall accuracy, per-category accuracies, and list of areas of IoU for each test instance.
        Parameters
        ----------
        predictedBoxList : list
            [[x,y,w,h],[x,y,w,h],[x,y,w,h],...]
            List of predicted bounding box instances [x,y,w,h] for each query instance.
            x and y are the (x,y) coordinates of the top-left of the bounding box for the query term
            w and h are the width and height of the bounding box for the query test
        gtBoxList : list
            Same as above, but for ground truth bounding boxes. Must be the same length as predictedBoxList
        iouThreshold : float
            The threshold for the IoU, such that two bounding boxes are considered overlapping when IoU >= iouThreshold.
        boxCategoriesList : list of list
            List of categories per box instance. Each box can have more than one category. Must be the same length as gtBoxList
        Returns
        -------
        accuracy : float
            Overall accuracy (or recall to be more precise). 
            Proportion of predicted boxes that overlaps with the ground truth boxes by an IoU of >= iouThreshold.

        perclassAccuracies : dict
            Per-class accuracy. Key: category label; Value: accuracy (float).
        iouList : list(float)
            The area of IoU for each prediction in predictedBoxList
        """

        # get all categories
        categorySet = set();
        for categoryList in boxCategoriesList:
            categorySet.update(categoryList);

        iouList = self.compute_iou(predictedBoxList, gtBoxList);
        accuracy = self.accuracy(iouList, iouThreshold);

        # evaluate for each category
        perClassAccDict = {};
        for category in categorySet:
            # get subset of instances for this category
            subPredictedBoxList = [];
            subGtBoxList = [];
            for (pred, gt, categoryList) in zip(predictedBoxList, gtBoxList, boxCategoriesList):
                if category in categoryList:
                    subPredictedBoxList.append(pred);
                    subGtBoxList.append(gt);
            #print("{}: {}".format(category, len(subGtBoxList)));

            # and evaluate subset
            subIouList = self.compute_iou(subPredictedBoxList, subGtBoxList);
            perClassAccDict[category] = self.accuracy(subIouList, iouThreshold);

        return (accuracy, perClassAccDict, iouList);


    def evaluate_upperbound_perclass(self, predictedBoxList, gtBoxList, boxCategoriesList, iouThreshold=0.5):
        """ Computes the overall accuracy, per-category accuracies, and list of areas of IoU for each test instance.
        Assumes that there are multiple candidate bounding boxes per test instance in predictedBoxList, 
        and we keep the max iou across all candidates to get the best iou per test instance
        Parameters
        ----------
        predictedBoxList : list of list
            [[[x,y,w,h],[x,y,w,h]],[[x,y,w,h],[x,y,w,h]],...]
            List of predicted bounding box instances [x,y,w,h] for each query instance.
            x and y are the (x,y) coordinates of the top-left of the bounding box for the query term
            w and h are the width and height of the bounding box for the query test
        gtBoxList : list
            Same as above, but for ground truth bounding boxes. Must be the same length as predictedBoxList
        iouThreshold : float
            The threshold for the IoU, such that two bounding boxes are considered overlapping when IoU >= iouThreshold.
        boxCategoriesList : list of list
            List of categories per box instance. Each box can have more than one category. Must be the same length as gtBoxList
        Returns
        -------
        accuracy : float
            Overall accuracy (or recall to be more precise). 
            Proportion of predicted boxes that overlaps with the ground truth boxes by an IoU of >= iouThreshold.

        perclassAccuracies : dict
            Per-class accuracy. Key: category label; Value: accuracy (float).
        iouList : list(float)
            The area of max IoU for each prediction set in predictedBoxList
        argmaxList : list(int)
            The index of the box that maximizes the IoU for each prediction set in predictedBoxList
        """

        # get all categories
        categorySet = set();
        for categoryList in boxCategoriesList:
            categorySet.update(categoryList);

        iouList = [];
        argmaxList = [];
        for (i, gtBox) in enumerate(gtBoxList):
            # replicate gt boxes to be the same number as number of candidates in prediction
            nCandidates = len(predictedBoxList[i]);
            replicatedGtBoxList = [];
            for j in range(nCandidates):
                replicatedGtBoxList.append(gtBox);
            instanceIouList = self.compute_iou(predictedBoxList[i], replicatedGtBoxList);
            maxIou = max(instanceIouList);
            iouList.append(maxIou);
            argmaxList.append(instanceIouList.index(maxIou));
        accuracy = self.accuracy(iouList, iouThreshold);

        # evaluate for each category
        perClassAccDict = {};
        for category in categorySet:
            # get subset of instances for this category
            subPredictedBoxList = [];
            subGtBoxList = [];
            for (pred, gt, categoryList) in zip(predictedBoxList, gtBoxList, boxCategoriesList):
                if category in categoryList:
                    subPredictedBoxList.append(pred);
                    subGtBoxList.append(gt);
            #print("{}: {}".format(category, len(subGtBoxList)));

            # and evaluate subset
            subIouList = [];
            for (i, subGtBox) in enumerate(subGtBoxList):
                # replicate gt boxes to be the same number as number of candidates in prediction
                nCandidates = len(subPredictedBoxList[i]);
                replicatedGtBoxList = [];
                for j in range(nCandidates):
                    replicatedGtBoxList.append(subGtBox);
                instanceIouList = self.compute_iou(subPredictedBoxList[i], replicatedGtBoxList);
                maxIou = max(instanceIouList);
                subIouList.append(maxIou);

            perClassAccDict[category] = self.accuracy(subIouList, iouThreshold);

        return (accuracy, perClassAccDict, iouList, argmaxList);

    

    def _iou(self, box1, box2):
        """Computes intersection over union (IoU) for two boxes.
        where each box = [x, y, w, h]
        Parameters
        ----------
        box1 : list
            [x, y, w, h] of first box
        box2 : list
            [x, y, w, h] of second box
        Returns
        -------
        float
            intersection over union for box1 and box2

        """

        (box1_left_x, box1_top_y, box1_right_x, box1_bottom_y) = box1;
        # 		box1_right_x = box1_left_x + box1_w - 1;
        # 		box1_bottom_y = box1_top_y + box1_h - 1;
        box1_w = box1_right_x-box1_left_x+1
        box1_h = box1_bottom_y - box1_top_y +1

        (box2_left_x, box2_top_y, box2_right_x, box2_bottom_y) = box2;
        # 		box2_right_x = box2_left_x + box2_w - 1;
        # 		box2_bottom_y = box2_top_y + box2_h - 1;
        box2_w = box2_right_x-box2_left_x+1
        box2_h = box2_bottom_y - box2_top_y +1

        # get intersecting boxes
        intersect_left_x = max(box1_left_x, box2_left_x);
        intersect_top_y = max(box1_top_y, box2_top_y);
        intersect_right_x = min(box1_right_x, box2_right_x);
        intersect_bottom_y = min(box1_bottom_y, box2_bottom_y);

        # compute area of intersection
        # the "0" lower bound is to handle cases where box1 and box2 don't overlap
        overlap_x = max(0, intersect_right_x - intersect_left_x + 1);
        overlap_y = max(0, intersect_bottom_y - intersect_top_y + 1);
        intersect = overlap_x * overlap_y;

        # get area of union
        union = (box1_w * box1_h) + (box2_w * box2_h) - intersect;

        # return iou
        return intersect * 1.0 / union;


def rotate(array):
    return torch.cat((array[1:], array[0:1]), dim=0)

a = torch.tensor([[1,1.], [2,2.], [3,4.]])
aa = torch.tensor([[1,2.], [2,3.], [3,6.]])
a = a.view(6,)
a.shape


def largest(bboxes):
    maxS=0
    use_gpu = torch.cuda.is_available()
    maxBox=torch.tensor([0,0,0,0])
    if (use_gpu):
        maxBox = maxBox.cuda()
    for box in bboxes:
        left, top, right, bottom = box[0], box[1], box[2], box[3]
        s = (right-left) * (bottom-top)
        if s>maxS:
            maxS=s
            maxBox=box
    return maxBox

def confidence(score, bboxes):
    maxIdx=np.argmax(score)
    return bboxes[maxIdx]


def union(bboxes):
    # leftmin, topmin, rightmax, bottommax = float('inf'),  float('inf'),  -float('inf'),  -float('inf')
    leftmin, topmin, rightmax, bottommax = 999,  999,  0,  0
    for box in bboxes:
        left, top, right, bottom = box
        if left==0 and top==0:
            continue
        leftmin, topmin, rightmax, bottommax = min(left, leftmin), min(top, topmin), max(right, rightmax), max(bottom, bottommax)
    # print(bboxes, leftmin, topmin)
    return [leftmin, topmin, rightmax, bottommax]

def union_target(bboxes_list):
    '''
        bboxes:[query, 12, 4]
    '''
    target_box_list = []
    # print(bboxes_list)
    for boxes in bboxes_list:
        # boxes: [12, 5]
        target_box = union(boxes) # target_box: [4]
        target_box_list.append(target_box)
    return target_box_list # [query, 4]


def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)


def assert_array_eq(real, expected):
    assert (np.abs(real-expected) < EPS).all(), \
        '%s (true) vs %s (expected)' % (real, expected)

def assert_tensor_eq(real, expected, eps=EPS):
    assert (torch.abs(real-expected) < eps).all(), \
        '%s (true) vs %s (expected)' % (real, expected)


def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def load_imageid(folder):
    images = load_folder(folder, 'jpg')
    img_ids = set()
    for img in images:
        img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        img_ids.add(img_id)
    return img_ids


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def weights_init(m):
    """custom weights initialization."""
    cname = m.__class__
    if cname == nn.Linear or cname == nn.Conv2d or cname == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.02)
    elif cname == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        print('%s is not initialized.' % cname)


def init_net(net, net_file):
    if net_file:
        net.load_state_dict(torch.load(net_file))
    else:
        net.apply(weights_init)


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def print_model(model, logger):
    print(model)
    nParams = 0
    for w in model.parameters():
        nParams += functools.reduce(operator.mul, w.size(), 1)
    if logger:
        logger.write('nParams=\t'+str(nParams))


def save_model(path, model, epoch, optimizer=None):
    model_dict = {
            'epoch': epoch,
            'model_state': model.state_dict()
        }
    if optimizer is not None:
        model_dict['optimizer_state'] = optimizer.state_dict()

    torch.save(model_dict, path)


# Select the indices given by `lengths` in the second dimension
# As a result, # of dimensions is shrinked by one
# @param pad(Tensor)
# @param len(list[int])
def rho_select(pad, lengths):
    # Index of the last output for each sequence.
    idx_ = (lengths-1).view(-1,1).expand(pad.size(0), pad.size(2)).unsqueeze(1)
    extracted = pad.gather(1, idx_).squeeze(1)
    return extracted


def trim_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    _use_shared_memory = True
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        out = None
        if 1 < batch[0].dim(): # image features
            max_num_boxes = max([x.size(0) for x in batch])
            if _use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = len(batch) * max_num_boxes * batch[0].size(-1)
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            # warning: F.pad returns Variable!
            return torch.stack([F.pad(x, (0,0,0,max_num_boxes-x.size(0))).data for x in batch], 0, out=out)
        else:
            if _use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [trim_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb

# Remove Flickr30K Entity annotations in a string
def remove_annotations(s):
    return re.sub(r'\[[^ ]+ ','',s).replace(']', '')

def get_sent_data(file_path):
    phrases = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for sent in f:
            str = remove_annotations(sent.strip())
            phrases.append(str)

    return phrases


# Find position of a given sublist
# return the index of the last token
def find_sublist(arr, sub):
    sublen = len(sub)
    first = sub[0]
    indx = -1
    while True:
        try:
            indx = arr.index(first, indx + 1)
        except ValueError:
            break
        if sub == arr[indx: indx + sublen]:
            return indx + sublen - 1
    return -1
# def find_sublist(arr, sub):
#     sublen = len(sub)
#     first = sub[0]
#     indx = -1
#     while True:
#         try:
#             indx = arr.index(first, indx + 1)
#         except ValueError:
#             break
#         if sub == arr[indx: indx + sublen]:
#             return [indx, indx + sublen]
#     return [-1, -1]

def calculate_iou(obj1, obj2):
    area1 = calculate_area(obj1)
    area2 = calculate_area(obj2)
    intersection = get_intersection(obj1, obj2)
    area_int = calculate_area(intersection)
    return area_int / ((area1 + area2 - area_int)+EPS)

def calculate_area(obj):
    return (obj[2] - obj[0]) * (obj[3] - obj[1])

def get_intersection(obj1, obj2):
    left = obj1[0] if obj1[0] > obj2[0] else obj2[0]
    top = obj1[1] if obj1[1] > obj2[1] else obj2[1]
    right = obj1[2] if obj1[2] < obj2[2] else obj2[2]
    bottom = obj1[3] if obj1[3] < obj2[3] else obj2[3]
    if left > right or top > bottom:
        return [0, 0, 0, 0]
    return [left, top, right, bottom]


def get_match_index(src_bboxes, dst_bboxes):
    indices = set()
    for src_bbox in src_bboxes:
        for i, dst_bbox in enumerate(dst_bboxes):
            iou = calculate_iou(src_bbox, dst_bbox)
            if iou >= 0.5:
                indices.add(i) ## match iou>0.5!!
    return list(indices)

def bbox_is_match(src_bbox, dst_bboxes):
    # for src_bbox in src_bboxes:
    for i, dst_bbox in enumerate(dst_bboxes):
            iou = calculate_iou(src_bbox, dst_bbox)
            if iou >= 0.5:
                return True
    return False

def unsupervised_get_match_index(src_bboxes, dst_bboxes):
    '''
    src_bboxes: dict (for all entities)
    '''
    indices = set()
    for entity, src_bboxes_list in src_bboxes.items():
        for src_bbox in src_bboxes_list:
            for i, dst_bbox in enumerate(dst_bboxes):
                iou = calculate_iou(src_bbox, dst_bbox)
                if iou >= 0.5:
                    indices.add(i) 
    return list(indices)


# Batched index_select
def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy) # b x e x f
    return out

def my_load_flickr30k(dataroot, img_id2idx, bbox, pos_boxes):
    """Load entries

    img_id2idx: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test-dev2015', test2015'
    """
    pattern_phrase = r'\[(.*?)\]'
    pattern_no = r'\/EN\#(\d+)'

    missing_entity_count = dict()
    multibox_entity_count = 0

    entries = []
    for image_id, idx in img_id2idx.items():

        phrase_file = os.path.join(dataroot, 'Flickr30kEntities/Sentences/%d.txt' % image_id)
        anno_file = os.path.join(dataroot, 'Flickr30kEntities/Annotations/%d.xml' % image_id)

        with open(phrase_file, 'r', encoding='utf-8') as f:
            sents = [x.strip() for x in f]

        # Parse Annotation
        root = parse(anno_file).getroot()
        obj_elems = root.findall('./object')
        pos_box = pos_boxes[idx]
        bboxes = bbox[pos_box[0]:pos_box[1]]
        target_bboxes = {}

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
                if not entity_id in target_bboxes:
                    target_bboxes[entity_id] = []
                else:
                    multibox_entity_count += 1
                target_bboxes[entity_id].append([left, top, right, bottom])
        
        # print("bbox", bboxes)
        # print(target_bboxes) #{entity_id: [xmin, ymin, xmax, ymax]}

        # Parse Sentence
        for sent_id, sent in enumerate(sents):
            sentence = remove_annotations(sent)
            entities = re.findall(pattern_phrase, sent)
            entity_indices = []
            target_indices = []
            entity_ids = []
            entity_types = []
            entity_names = []

            for entity_i, entity in enumerate(entities):
                info, phrase = entity.split(' ', 1)
                # print("info--", info, "phrase--", phrase) # info:entity id & type, phrase: entity phrase
                entity_id = int(re.findall(pattern_no, info)[0])
                entity_type = info.split('/')[2:]

                entity_idx = find_sublist(sentence.split(' '), phrase.split(' '))
                # print(sentence)
                # print("entity_idx---", entity_idx)
                assert 0 <= entity_idx

                if not entity_id in target_bboxes:
                    if entity_id >= 0:
                        missing_entity_count[entity_type[0]] = missing_entity_count.get(entity_type[0], 0) + 1
                    continue

                assert 0 < entity_id

                entity_names.append(phrase)
                entity_ids.append(entity_id)
                entity_types.append(entity_type)
                assert len(entity_names) == len(entity_ids)
                
                target_idx = get_match_index(target_bboxes[entity_id], bboxes) # iou match
                entity_indices.append(entity_idx)
                target_indices.append(target_idx)
                
            if 0 == len(entity_ids):
                continue
            entry = {
                'image'          : image_id,
                'sentence'       : sentence,
                'entity_names'   : entity_names,
                'target_indices' : target_indices,
                'entity_ids'     : entity_ids,
                'entity_types'   : entity_types,
                'entity_num'     : len(entity_ids),
                'target_bboxes'  : target_bboxes
                }
            entries.append(entry)

    return entries