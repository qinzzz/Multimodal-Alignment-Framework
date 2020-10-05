# # Finetune word2vec on flickr30k
# input:
# - query q e.g. "the boy in read shirt"
# - labels e e.g.["boy", "footwear", "ball", "shirt"]
#
# dataload:
# - wordembedding: glove wordembedding
# - image dict: all sentences in one image (from bottom up attention)
#     - labels -> list of indexes [x, x, x, .... x] pad to 64
#     - features (x)
#     - bboxes
#     for each sentence: (from flickr loader)
#     - sentence
#     - query -> list of indexes [x, x, x, ..., 0] pad to 16

# model:
# matmul (query, labels) -> attention map [len(query) x len(labels)]

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
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import _pickle as cPickle
from tqdm import tqdm
from xml.etree.ElementTree import parse
from dataset import Indexer, WordEmbeddings, Flickr30dataset, read_word_embeddings, load_train_flickr30k
from utils import largest, confidence, union, bbox_is_match, get_match_index, my_load_flickr30k, Evaluator, union_target
from model import NN
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py


def train(model, train_loader, test_loader, batch, lr=1e-4, epochs=25, use_bert = False, lite_bert = False):
    use_gpu = torch.cuda.is_available()

    model = model.float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    ceLoss = nn.CrossEntropyLoss(reduction="mean")
    bceLoss = nn.BCEWithLogitsLoss()
    rankingLoss = nn.MarginRankingLoss(margin=0.2)
    cosembLoss = nn.CosineEmbeddingLoss()

    print("---Before Training...")
    score = model_eval(test_loader, model, use_bert, lite_bert)
    print("     eval score on test dataset:", score)

    for epoth in range(epochs):
        t= time.time()
        total_loss = 0
        correct_preds = 0
        all_preds = 0

        all_hits = 0
        all_counts = 0
        n_batches = 0

        # if True:
        '''
            torch.Size([16, 1]) torch.Size([16, 64]) torch.Size([16, 32, 12]) torch.Size([16, 64, 5])
            torch.Size([16, 32, 12, 5]) torch.Size([16, 1]) torch.Size([16])
        '''
        for idx, labels, attrs, feature, query, label_feats, sent_toks, phrase_toks, entity_indices, entity_feats, bboxes, target_bboxes, num_obj, num_query in tqdm(train_loader):
            if (use_gpu):
                idx, labels, attrs, feature, query, label_feats, sent_toks, phrase_toks, entity_indices, entity_feats, bboxes, target_bboxes, num_obj, num_query = \
                    idx.cuda(), labels.cuda(), attrs.cuda(), feature.cuda(), query.cuda(), label_feats.cuda(), sent_toks.cuda(), phrase_toks.cuda(), entity_indices.cuda(), entity_feats.cuda(), bboxes.cuda(), target_bboxes.cuda(), num_obj.cuda(), num_query.cuda()
            n_batches+=1
            
            model.train(True)
            optimizer.zero_grad()
            # print(labels, query, bboxes, target_bboxes)
            '''
                Crossentropy Loss Contrastive loss

            probs, target = model.forward(query, labels, num_obj, num_query)  # [B, B, all_query]
            # print(probs.shape, target.shape)

            merged_probs = torch.cat(tuple(prob[:, :n] for prob, n in zip(probs, num_query)), -1).permute(1,0) # [all querys, B]
            merged_target = torch.cat(tuple(targ[:, :n] for targ, n in zip(target, num_query)), -1).permute(1,0) # [all querys, B]

            target_pred = torch.argmax(merged_target, dim=1) # [all_querys]
            # print(merged_probs[:5])
            # print(merged_target[:5])
            # print(target_pred[:5])

            loss = ceLoss(merged_probs, target_pred)
            # print(loss)
            '''
            '''
                TripletLoss
            '''
            # anchor, positive, negative = model.forward_embedding(query, labels, num_query)  # [total_query, dim]
            # total_query = torch.sum(num_query)
            # loss = tripletLoss(anchor, positive, negative)
            '''
                3

            probs, target = model.forward_all(query, labels, num_obj, num_query)  # [B, query, B, label]
            # print("porbs", probs.shape)
            # print("target", target.shape)
            merge_q_probs = torch.cat(tuple(probs[i, :num_query[i]] for i in range(probs.size(0))), 0) # [all_qeurys, B, label]
            # print("merge_q_probs", merge_q_probs.shape)
            merged_probs = torch.cat(tuple(merge_q_probs[:, i, :num_obj[i]] for i in range(probs.size(2))), -1) # [all querys, B]
            # print("merge_v_probs", merged_probs.shape)
            merge_q_target = torch.cat(tuple(target[i, :num_query[i]] for i in range(probs.size(0))), 0) # [all_qeurys, B, label]
            merged_target = torch.cat(tuple(merge_q_target[:, i, :num_obj[i]] for i in range(probs.size(2))), -1) # [all querys, B]
            # print("merged_target", merged_target.shape)
            # print(num_query)
            loss = bceLoss(merged_probs, merged_target)
            '''

            '''
            4
            '''
            # require grad
            # print(feature.requires_grad, query_feats.requires_grad, label_feats.requires_grad)
            # if use_bert:
            #     query_feats.requires_grad=True
            #     label_feats.requires_grad=True
            # print("index", idx)

            probs, target, att_obj_sum = model.forward(idx, query, labels, feature, attrs, bboxes, num_obj, num_query, label_feats, sent_toks, phrase_toks, entity_indices, entity_feats, use_bert, lite_bert)  # [B, B]
            
            # print("prob", probs)
            ## rangking loss ##
            # B = batch
            # probs = nn.Softmax(dim=-1)(probs)
            # pos_mat = probs[:B//2, :B//2]
            # neg_mat = probs[:B//2, B//2:]
            # # print("mat", pos_mat.shape, neg_mat.shape)
            # index = torch.tensor([[i for i in range(B//2)]]).cuda()
            # pos_ex = pos_mat.gather(0, index)[0]
            # neg_ex,_ = torch.max(neg_mat, dim=-1)
            # # print(pos_ex, neg_ex)
            # ys = torch.ones(B//2).cuda()
            # loss = rankingLoss(pos_ex, neg_ex, ys)

            target_pred = torch.argmax(target, dim=1) # [B]
            prediction = torch.argmax(probs, dim=1) # [all_querys]
            correct_preds += int(prediction.eq(target_pred).sum())
            all_preds += len(prediction)

            loss = ceLoss(probs, target_pred)
            # loss_unique = max(0, torch.sum(att_obj_sum-1))
            # print("loss", loss_unique)
            # input()

            total_loss+=loss
            loss.backward()
            optimizer.step()

            # prediction
            '''
            grounding evaluation
            '''
            # pred_bboxes, pred_labels, _, _ = model.predict(query, labels, num_obj, bboxes) # [B, 32, 4]
            # hit, count = calculate_score(pred_bboxes, target_bboxes, num_query)
            # all_hits+=hit
            # all_counts+=count

        t1 = time.time()
        print("--- EPOCH", epoth)
        print("     time:", t1-t)
        print("     total loss:", total_loss.item()/n_batches)
        print("     supervised accuracy on training set: ", correct_preds/all_preds)
        # test grounding acc on training set
        # print("     eval score on training dataset:", float(all_hits)/float(all_counts))
        t2 = time.time()
        score, supacc = model_eval(test_loader, model, use_bert, lite_bert)
        print("     eval time:", time.time()-t2)
        print("     supervised accuracy on test dataset:", supacc)
        print("     eval score on test dataset:", score)

'''
    idx.shape, label.shape, query.shape, bboxes.shape, target.shape, nobj.shape, nquery.shape
    torch.Size([16, 1]) torch.Size([16, 64]) torch.Size([16, 32, 12]) torch.Size([16, 64, 5]) torch.Size([16, 32, 12, 5]) torch.Size([16, 1]) torch.Size([16, 1])
'''
def model_eval(train_loader, model, use_bert = False, lite_bert = False):
    t= time.time()
    use_gpu = torch.cuda.is_available()
    all_hits = 0
    all_counts = 0

    correct_preds = 0
    all_preds = 0

    model = model.float()

    # record = []
    pred_bboxes_list = []
    target_bboxes_list = []
    num_query_list = []

    for idx, labels, attrs, feature, query, label_feats, sent_toks, phrase_toks, entity_indices, entity_feats, bboxes, target_bboxes, num_obj, num_query in tqdm(train_loader):
        if (use_gpu):
                idx, labels, attrs, feature, query, label_feats, sent_toks, phrase_toks, entity_indices, entity_feats, bboxes, target_bboxes, num_obj, num_query = \
                    idx.cuda(), labels.cuda(), attrs.cuda(), feature.cuda(), query.cuda(), label_feats.cuda(), sent_toks.cuda(), phrase_toks.cuda(), entity_indices.cuda(), entity_feats.cuda(), bboxes.cuda(), target_bboxes.cuda(), num_obj.cuda(), num_query.cuda()

        model.eval()
        # show labels & query
        # label_words = [model.indexer.get_object(int(i)) for i in labels[0]]
        # query_words = [[model.indexer.get_object(int(i)) for i in query[0][j]] for j in range(query.size(1))]
        
        pred_bboxes, pred_labels, probs, target = model.predict(idx, query, labels, feature, attrs, num_obj, num_query, bboxes, label_feats, sent_toks, phrase_toks, entity_indices, entity_feats, use_bert, lite_bert) # [B, 32, 4]

        # pred_bboxes = pred_bboxes.tolist()
        # idx = idx.squeeze().tolist()
        # print("predict res", idx, pred_labels, pred_bboxes)
        # print(target_bboxes[0].tolist())
        # print(prediction[0], pred_labels)
        # print(i_att[0, :num_query].cpu().numpy())

        # sup acc
        target_pred = torch.argmax(target, dim=1) # [B]
        prediction = torch.argmax(probs, dim=1) # [all_querys]
        correct_preds += int(prediction.eq(target_pred).sum())
        all_preds += len(prediction)

        # unsup acc
        # pred: [B, query, 4]
        # target: [B, query, 12, 5]

        # hit, count = calculate_score(pred_bboxes, target_bboxes, num_query)
        # all_hits+=hit
        # all_counts+=count
        # print("pred_bboxes", pred_bboxes.shape, "target_bboxes", target_bboxes.shape)
        pred_bboxes_list += pred_bboxes.cpu().tolist()
        target_bboxes_list += target_bboxes.cpu().tolist()
        num_query_list +=num_query.cpu().tolist()

    # print(len(pred_bboxes_list), len(target_bboxes_list), len(num_query_list))
    score = evaluator_acc(pred_bboxes_list, target_bboxes_list, num_query_list)
    # print("     hit: ", all_hits, "total: ", int(all_counts))
    # score = float(all_hits)/float(all_counts)
    # print("     correct preds:", correct_preds, "all preds: ", all_preds)
    supacc = correct_preds/all_preds
    
    return score, supacc

def untrained_eval(train_loader, wordEmbedding):
    all_hits = 0
    all_counts = 0
    model = NN(wordEmbedding)
    model = model.float()
    for batch in train_loader:
        idx, labels, query, bboxes, target_bboxes, num_obj, num_query = batch
        print(idx)
        pred_bboxes = model.predict(query, labels, num_obj, bboxes) # [B, 32, 4]
        hit, count = calculate_score(pred_bboxes, target_bboxes, num_query)
        all_hits+=hit
        all_counts+=count
    print(all_hits, all_counts)
    score = float(all_hits)/float(all_counts)
    return score

def testcaseEval(wordEmbedding):
    model = NN(wordEmbedding)
    model = model.float()
    q = "a boy in red shirt"
    l = ['person', 'ball', 'glove', 'boy', 'clothing', 'footwear', 'red']
    labels_idx = [0]* 64
    labels_idx[:len(l)] = [max(wordEmbedding.word_indexer.index_of(w), 1) for w in l]

    querys_idx = []
    q_ = q.split()
    lis=[0]*12

    for i in range(len(q_)):
        lis[i] = max(wordEmbedding.word_indexer.index_of(q_[i]), 1)
    querys_idx.append(lis)
    while(len(querys_idx)<32):
        querys_idx.append([0]*12)

    query = torch.tensor([querys_idx])
    labels = torch.tensor([labels_idx])
    print("query tensor", query, "label tensor", labels)
    num_obj = torch.tensor([7])
    att, prediction = model.predict(query, labels, num_obj)
    print(att[0,0])
    print(prediction[0,0])

    w2v = w2v_att(q_, l, model.wordemb)
    print(q_, l)
    print("w2v att:", w2v)

    label_words = [model.indexer.get_object(int(i)) for i in labels[0]]
    query_words = [[model.indexer.get_object(int(i)) for i in query[0][j]] for j in range(query.size(1))]
    w2v_ = w2v_att(query_words[0], label_words, model.wordemb)
    print(query_words[0], label_words)
    print("w2v_", w2v_)


def evaluator_acc(pred_bboxes, target_bboxes, num_query):
    evaluator = Evaluator()
    gtbox_list=[]
    pred_list=[]
    for ipred, itarget, nq in zip(pred_bboxes, target_bboxes, num_query):
        # ipred: [query, 5]
        # itarget: [query, 12, 4]
        if nq > 0:
            gtbox_list += union_target(itarget[:nq]) # [query, 4]
            pred_list += ipred[:nq]

    accuracy, _ = evaluator.evaluate(pred_list, gtbox_list) # [query, 4]
    return accuracy


def calculate_score(pred_bboxes, target_bboxes, num_query):
    '''
    pred: [B, query, 4]
    target: [B, query, 12, 5]
    num_query: [B]
    '''
    match_sum=0
    for ipred, itarget, nq in zip(pred_bboxes, target_bboxes, num_query):
        # ipred: [query, 4]
        # itarget: [query, 12, 5]
        # print(ipred, itarget, nq)
        for j in range(nq):
            # ipred[j]: [4]
            # itarget[j]: [12, 4]
            # print(j, itarget[j])
            union_target = union(itarget[j]) # [4]
            print(union_target)
            ismatch = bbox_is_match(ipred[j], [union_target])
            if ismatch:
                match_sum+=1
    all_query = sum(num_query)
    return match_sum, all_query


def w2v_att(query, det, glove):
    query = query.lower().split()

    softmax = nn.Softmax(dim=1)
    cos = nn.CosineSimilarity()

    if glove:
        q_emb = torch.from_numpy(glove.get_embeddings(query))
        k_emb = torch.from_numpy(glove.get_embeddings(det))
    else:
        q_emb = torch.from_numpy(model_fast.wv[query])
        k_emb = torch.from_numpy(model_fast.wv[det])

    # print(q_emb.shape, k_emb.shape)
    scale = 1.0/np.sqrt(q_emb.size(-1))
    att = torch.matmul(q_emb, k_emb.transpose(0,1))
    # mask = (att==0)
    # att.masked_fill_(mask, -float('inf'))
    att = softmax(att.mul_(scale))
    # att[torch.isnan(att)] = 0

    max_att = torch.max(att, dim=1).values
    max_norm_att = max_att.div(max_att.sum()).unsqueeze(0)

    p_emb = torch.matmul(max_norm_att, q_emb).repeat(len(det), 1)
    # print("att phrase embedding", p_emb[0])

    sim = cos(p_emb, k_emb).unsqueeze(0)

    return sim.tolist()[0]


def load_entries(name='train'):
    dataroot='data/flickr30k/'

    img_id2idx = cPickle.load(
        open(os.path.join(dataroot, '%s_imgid2idx.pkl' % name), 'rb'))
    h5_path = os.path.join(dataroot, '%s.hdf5' % name)

    print('loading features from h5 file...')
    with h5py.File(h5_path, 'r') as hf:
        features = np.array(hf.get('image_features'))
        spatials = np.array(hf.get('spatial_features'))
        bbox = np.array(hf.get('image_bb'))
        pos_boxes = np.array(hf.get('pos_boxes'))

    print("load flickr30k data successfully.")
    entries = my_load_flickr30k(dataroot, img_id2idx, bbox, pos_boxes)
    return entries


def entries_id2img(train_entries):
    train_id2img = {}
    for entry in train_entries:
        img_id = entry['image']
        if img_id not in train_id2img.keys():
            train_id2img[img_id] = [entry]
        else:
            train_id2img[img_id].append(entry)
    return train_id2img


def evaluate(object_detect, eval_entries, glove, strategy = 'largest'):
    print("start calculating....")
    total_image = 0
    total_entity2img = 0
    correct=0
    correct_all = 0
    hit_entity = 0
    match_info = {}

    object_detect = sorted(object_detect, key=lambda x:x["image"].split('.')[0])

    # f = open("att_result.json", "w")

    for img in object_detect: # one picture
        img_id = int(img["image"].split('.')[0])
        true_match={}
        false_match = {}

        if img_id in eval_entries.keys(): # has a groung truths
            total_image+=1
            target_entries = eval_entries[img_id]
        else:
            continue

        det_objects = img["objects"]
        det_bboxes = []
        dic_class2bbox={} # {class1: [box1, box2,...];...}
        dic_class2score={} #  {class1: [score1, score2,...];...}
        for d in det_objects:
            box_str = d["bbox"].strip('(').strip(')')
            bbox = box_str.split(',')
            bbox = [int(i) for i in bbox]
            det_bboxes.append(bbox)
            if d["class"] not in dic_class2bbox.keys():
                dic_class2bbox[d["class"]] = [bbox]
                dic_class2score[d["class"]] = [float(d["score"])]
            else:
                dic_class2bbox[d["class"]].append(bbox)
                dic_class2score[d["class"]].append(float(d["score"]))
        # write to json
        # dic = {"img":img_id, "box":dic_class2bbox}
        # json.dump(dic, f)
        labels = list(dic_class2bbox.keys())
        for target_entry in target_entries:
            for i, entity in zip(target_entry["entity_ids"], target_entry["entity_names"]): # entity is given
                total_entity2img+=1

                if len(labels)==0:
                    continue
                w2v = w2v_att(entity, labels, glove)

                # use max similarity score
                if len(w2v)>0:
                    score, idx = np.max(w2v), np.argmax(w2v) # idx is the index of label
                else:  # dic_class2bbox is empty??
                    continue

                pred_class = labels[idx]
                print(entity, pred_class)
                pred_bboxes = dic_class2bbox[pred_class] # bbox in obj_detection.json
                scores = dic_class2score[pred_class]

                # DEBUG

                # print("att res:", img_id, entity, pred_class)
                # print(target_entry['target_bboxes'][i])
                # print(w2v)

                # line = {"image":img_id, "entity": entity, "pred class":pred_class, "pred box":pred_bboxes}
                # json.dump(line, f, indent=4)

                # select one from pred bboxes
                if strategy=="largest":
                    pred_bbox = largest(pred_bboxes)
                elif strategy == "confidence":
                    pred_bbox = confidence(scores, pred_bboxes)
                elif strategy == "union":
                    pred_bbox = union(pred_bboxes)
                else:
                    pred_bbox = random.sample(pred_bboxes, 1)[0]

                match = get_match_index([pred_bbox], target_entry['target_bboxes'][i])
                match_all = get_match_index(pred_bboxes, target_entry['target_bboxes'][i])
                # print("image ",target_entry['image'], ", entity: ", entity)
                # print("pred class:", pred_class)

                if len(match)>0:
                    true_match[entity] = pred_class
                    correct+=1
                else:
                    false_match[entity] = pred_class
                if len(match_all)>0:
                    correct_all+=1

        # match_info[img_id] = {"labels": labels, "true": true_match, "false": false_match}

    print('# of images', total_image)
    print('# of entity-image pairs', total_entity2img)
    print("# of correct", correct)
    print('# of correct predict in a image:', correct/total_image)
    print("acc:", 1.0*correct/total_entity2img)
    print("acc upper bound:", 1.0 * correct_all/total_entity2img)
    # with open("att_match.json", "w") as f:
    #     json.dump(match_info, f, indent=4)


