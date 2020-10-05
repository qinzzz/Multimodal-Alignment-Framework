import torch
import torch.nn as nn
import numpy as np
from utils import rotate, largest, confidence, union
import random
import time
# from transformers import *


class NN(nn.Module):
    def __init__(self, wordvec, emb_dim=300, feature_dim=2048, bert_dim=768, vgg = False):
        '''
        wv: (*) -> (*, dim)
        '''
        super(NN, self).__init__()

        if vgg:
            feature_dim = 4096

        self.bert_dim = bert_dim
        self. eps = 1e-6
        self. wordemb = wordvec
        self. indexer = wordvec.word_indexer
        self. wv = nn.Embedding.from_pretrained(torch.from_numpy(wordvec.vectors), freeze=False)
        # self.bertmodel = BertModel.from_pretrained('bert-base-uncased')

        # self. wv = nn.Embedding(len(self.indexer), emb_dim)
        self.linear_p = nn.Linear(emb_dim, emb_dim)
        self.linear_a = nn.Linear(emb_dim, emb_dim)
        self.linear_f = nn.Linear(feature_dim, emb_dim)
        self.linear_mini = nn.Linear(emb_dim, emb_dim)

        # self.linear_f_bert = nn.Linear(feature_dim, bert_dim)
        # self.linear_q_bert = nn.Linear(bert_dim, emb_dim)
        # self.linear_k_bert = nn.Linear(bert_dim, emb_dim)
        # self.linear_bert = nn.Linear(bert_dim, bert_dim)
        # self.linear_q = nn.Linear(emb_dim, emb_dim)
        # self.linear_k = nn.Linear(emb_dim, emb_dim)

        self. relu = nn.ReLU()
        self. softmax = nn.Softmax(dim=-1)
        self. cosSim = nn.CosineSimilarity(dim=-1, eps=self.eps)

        self.linear_p.weight.data = torch.eye(emb_dim)
        # self.linear_a.weight.data = torch.zeros(emb_dim, emb_dim)
        self.linear_f.weight.data = torch.zeros(emb_dim, feature_dim)

        # nn.init.xavier_uniform_(self.linear_p.weight)
        # nn.init.xavier_uniform_(self.linear_f.weight)

        # self.linear_bert.weight.data = torch.eye(bert_dim)
        # self.linear_q.weight.data = torch.eye(emb_dim)
        # self.linear_k.weight.data = torch.eye(emb_dim)


    def encode(self, query, label, feature, attrs, bboxes=None, use_bert=False, lite_bert=False):
        # labels = label.unsqueeze(1).repeat(1, query.size(1), 1) #  labels: [B, 1, K=32] -> [B, all_query, K=32]
        # print("bboxes", bboxes[0])
        q_emb = self.wv(query) # [B, all_query, Q, dim]
        k_emb = self.wv(label) #  [B, K, dim]
        # a_emb = self.wv(attrs)
        # q_emb = self.linear_q(q_emb)
        # k_emb = self.linear_k(k_emb)

        f_emb = self.linear_f(feature) # [B, K, dim]
        # a_emb = self.linear_a(a_emb)
        # k_emb = f_emb
        k_emb = k_emb + f_emb
        # k_emb += f_emb + a_emb

        return q_emb, k_emb


    def encode_pk(self, query, label, feature, attrs, label_feats=None, sent_toks=None, phrase_toks=None, entity_indices=None, entity_feats=None, use_bert=False, lite_bert=False):
        eps = 1e-5

        if use_bert:
            with torch.no_grad():
                pass
                # sent_emb = self.bertmodel(sent_toks)[0] # [B, S, dim]
                # if idx in self.label_feature_dict.keys():
                #     k_emb = self.bertmodel(label_toks)[0] # [B, K, dim]
                #     self.label_feature_dict[idx] = k_emb
                # else:
                #     k_emb = self.label_feature_dict[idx]
                # k_emb = self.bertmodel(label_toks)[0] # [B, K, dim]
                # k_emb = label_feats
            k_emb = self.linear_f_bert(feature) # [B, K, dim]
            q_emb = entity_feats # [B, querys, Q, dim]

            # q_emb = self.linear_q_bert(q_emb)
            # k_emb = self.linear_k_bert(k_emb)

            # q_emb = torch.zeros(entity_indices.size(0),entity_indices.size(1),sent_emb.size(1),sent_emb.size(2)).cuda() # [B, querys, Q, dim]
            # for i, entity_idx in enumerate(entity_indices):
            #     for j in range(entity_indices.size(1)):
            #         q_emb[i, j, :(entity_idx[j,1]-entity_idx[j,0])] = sent_emb[i, entity_idx[j,0]:entity_idx[j,1]] # [querys, Q, dim]

            # p_emb = torch.zeros(entity_indices.size(0), entity_indices.size(1), sent_emb.size(2)).cuda()
            # for i, entity_idx in enumerate(entity_indices):
            #     for j in range(entity_indices.size(1)):
            #         p_emb[i, j] = sent_emb[i, entity_idx[j,0]] # [B, querys, dim]

        elif lite_bert:
            with torch.no_grad():
                k_emb = label_feats # [B, K, dim]
                # k_emb = self.bertmodel(label_toks)[0]
                long_phrase_toks = phrase_toks.view(phrase_toks.size(0)*phrase_toks.size(1), phrase_toks.size(2)) # [v1*v0, v2]
                q_emb = self.bertmodel(long_phrase_toks)[0] # [v0*v1, v2, dim]
                q_emb = q_emb.view(phrase_toks.size(0), phrase_toks.size(1), phrase_toks.size(2), self.bert_dim)

            # q_emb = self.linear_q_bert(q_emb)
            # k_emb = self.linear_k_bert(k_emb)

            # print(k_emb.shape, q_emb.shape)
            # print(k_emb.requires_grad, q_emb.requires_grad) --- False

        else:
            q_emb, k_emb = self.encode(query, label, feature, attrs) # [B, querys, Q, dim] & [B, K, dim]

        # q_emb [B, querys, Q, dim]
        scale =1.0/np.sqrt(k_emb.size(-1))
        att = torch.einsum('byqd,bkd ->byqk', q_emb, k_emb)
        att = self.softmax(att.mul_(scale))  # [B, querys, Q, K]

        q_max_att = torch.max(att, dim=3).values # [B, querys, Q]
        # print("q_max_att", q_max_att)
        q_max_norm_att = self.softmax(q_max_att)

        # attended
        p_emb = torch.einsum('byq,byqd -> byd', q_max_norm_att, q_emb) # [B, querys, dim]

        # ablation study
        # average
        # len_query = torch.sum((query==0), dim=-1) # [B, querys]
        # len_query = len_query.unsqueeze(-1).expand(query.size(0), query.size(1), q_emb.size(3))# [B, querys, dim]
        # p_emb = torch.sum(q_emb, dim=-2) / (len_query+eps)

        if use_bert or lite_bert:
            p_emb = self.linear_bert(p_emb)
        else:
            # p_emb = self.linear_p(p_emb)
            p_emb = self.linear_p(p_emb) + eps * self.linear_mini(p_emb)

        return p_emb, k_emb


    def forward(self, idx, query, label, feature, attrs, bboxes, num_obj, num_query, label_feats=None, sent_toks=None, phrase_toks=None, entity_indices=None, entity_feats=None, use_bert=False, lite_bert=False):
        '''
        labels: [B, K=64] pad with 0
        query: [B, all_query=32, Q=12] pad with 0
        feature: [B, K, dim = 2048]
        bboxes: [B, K, 5]
        return: match score [Batch, batch, querys]
        '''
        p_emb, k_emb = self.encode_pk(query, label, feature, attrs, label_feats, sent_toks, phrase_toks, entity_indices, entity_feats, use_bert, lite_bert)
        '''
            M = querys x K
        '''
        attmap = torch.einsum('avd, bqd -> baqv', k_emb, p_emb) # [B1, K, dim] x [B2, querys, dim] => [B2, B1, querys, K]

        attmap_sm = self.softmax(attmap) # [B2, B1, querys, K]
        att_obj_sum = torch.sum(attmap_sm, dim = -2) # [B2, B1, K]
        # print(att_obj_sum.shape)
        # print(attmap_sm[0][0])

        maxatt, _ = attmap.max(dim=-1) # [B1, B2, querys]: B1th sentence to B2th image
        logits = torch.sum(maxatt, dim=-1).div(num_query.unsqueeze(1).expand(maxatt.size(0), maxatt.size(1)))# [B1, B2]: B1th sentence to B2th image 求和后需要除以query的个数！
        # print("similarity mat shape:", logits.shape)
        # print(logits[:10])

        n_obj = int(query.size(0))
        target = torch.eye(n_obj).cuda() # [b, b]
        # print("res", logits.shape, target.shape)
        return logits, target, att_obj_sum # [B, B]

    def forward_old(self, query, label, num_obj, num_query): # labels are negative examples in the batch
        '''
        labels: [B, K=64] pad with 0
        query: [B, all_query=32, Q=12] pad with 0

        return: match score [Batch, batch, querys]
        '''
        eps = 1e-5

        # mask = (query!=0) # [B, querys, Q]
        # len_query = mask.sum(dim=2) # [B, querys]
        # print(len_query[:5])

        q_emb, k_emb = self.encode(query, label) # [B, querys, Q, dim] & [B, K, dim]
        '''
        p_emb, i_emb = self.phrase_encode(q_emb, k_emb, num_obj, len_query)
        # print(p_emb.shape, p_emb[:5])
        # print(i_emb.shape, i_emb[:5])
        # input()
        logits = torch.einsum('ad,bqd -> baq', (i_emb, p_emb))
        '''
        scale =1.0/np.sqrt(k_emb.size(2))
        att = torch.einsum('byqd,bkd ->byqk', q_emb, k_emb)
        q_max_att = torch.max(att, dim=3).values # [B, querys, Q]
        # print("q_max_att", q_max_att)

        q_max_norm_att = self.softmax(q_max_att)
        # q_max_norm_att = q_max_att.div(q_max_att.sum(dim=2, keepdim=True) + eps) # [B, querys, Q]
        # print("q_max_norm_att", q_max_norm_att[:5])
        p_emb = torch.einsum('byq,byqd -> byd', q_max_norm_att, q_emb) # [B, querys, dim]
        # p_emb = self.linear(p_emb)

        attmap = torch.einsum('avd, bqd -> baqv', k_emb, p_emb) # [B, K, dim] x [B, querys, dim] => [B, B, querys, K]
        attmap = self.softmax(attmap.mul_(scale))

        i_emb = torch.einsum('baqv, avd -> baqd', (attmap, k_emb))
        logits = torch.einsum('baqd, bqd -> baq', (i_emb, p_emb))

        n_obj = int(q_emb.size(0))
        target = torch.eye(n_obj).cuda() # [b, b]
        target = target.unsqueeze(2).repeat(1,1,logits.size(-1)) # [b, b, q]

        ### old version ###
        # scores = []
        # for i in range(query.size(0)):
        #     k_emb = rotate(k_emb)
        #     sim = self.similarity_score(q_emb, k_emb)# [B, all_query] ->[B, all_query]
        #     scores.append(sim)
        # scores= torch.stack(scores, dim=-1)

        return logits, target # [B, B, querys] & [B, B, query]

    def forward_avg(self, query, label, num_obj, num_query):
        '''
        labels: [B, K=64] pad with 0
        query: [B, all_query=32, Q=12] pad with 0

        return: match score [Batch, querys, batch]
        '''
        mask = (query!=0) # [B, querys, Q]
        len_query = mask.sum(dim=2) # [B, querys]
        # print(len_query[:5])

        q_emb, k_emb = self.encode(query, label) # [B, querys, Q, dim] & [B, K, dim]

        p_emb, i_emb = self.phrase_encode(q_emb, k_emb, num_obj, len_query)

        logits = torch.einsum('ad,bqd -> baq', (i_emb, p_emb))

        n_obj = int(q_emb.size(0))
        target = torch.eye(n_obj).cuda() # [b, b]
        target = target.unsqueeze(2).repeat(1,1,logits.size(-1)) # [b, b, q]
        return logits, target

    def forward_embedding(self, query, label, num_obj, num_query):
        '''
            return: anchor, positive, negative
        '''
        q_emb, k_emb = self.encode(query, label) # q: [B, querys, Q, dim] , [B, K, dim]

        scale =1.0/np.sqrt(q_emb.size(3))
        att = torch.einsum('byqd,bkd ->byqk', q_emb, k_emb)
        mask = (att==0)
        att.masked_fill_(mask, -float('inf'))
        att = self.softmax(att.mul_(scale)) # [B, querys, Q, K]
        att[torch.isnan(att)] = 0

        # embedding of every phrase with regard to the labels
        q_max_att = torch.max(att, dim=3).values # [B, querys, Q]
        q_max_norm_att = q_max_att.div(q_max_att.sum(dim=2, keepdim=True)) # [B, querys, Q]
        p_emb = torch.einsum('byq,byqd -> byd', q_max_norm_att, q_emb) # [B, querys, dim]

        i_att = torch.einsum('bkd, byd -> byk', k_emb, p_emb) # [B, querys, K, dim] x [B, querys, dim] => [B, querys, K]
        i_emb = torch.einsum('byk, bykd -> byd', i_att, k_emb) # [B, querys, K] x [B, queys, K, dim] -> [B, querys, dim]

        random_idx = [i for i in range(i_emb.size(0))]
        random.shuffle(random_idx)
        shuffle_i_emb = i_emb[random_idx]

        anchor = torch.cat(tuple(p[:n, :] for p, n in zip(p_emb, num_query)), 0) # [total_query, dim]
        positive = torch.cat(tuple(i[:n, :] for i, n in zip(i_emb, num_query)), 0)
        negative = torch.cat(tuple(i[:n, :] for i, n in zip(shuffle_i_emb, num_query)), 0)

        return anchor, positive, negative

    def predict(self, idx, query, label, feature, attrs, num_obj, num_query, bboxes= None, label_feats=None, sent_toks=None, phrase_toks=None, entity_indices=None, entity_feats=None, use_bert=False, lite_bert=False):
        '''
        labels: [B, K=64]
        query: [B, querys=32, Q=12]
        bboxes: [B, K=64, 5]
        '''
        eps = 1e-3

        t0 = time.time()
        p_emb, k_emb = self.encode_pk(query, label, feature, attrs, label_feats, sent_toks, phrase_toks, entity_indices, entity_feats, use_bert, lite_bert)

        # mask
        # mask = (att==0)
        # att.masked_fill_(mask, -float('inf'))
        # att = self.softmax(att.mul_(scale)) # [B, querys, Q, K]
        # att[torch.isnan(att)] = 0

        i_att = torch.einsum('bkd, byd -> byk', k_emb, p_emb) # [B, K, dim] x [B, querys, dim] => [B, querys, K]
        # k_emb_ = k_emb.unsqueeze(1).expand(k_emb.size(0), p_emb.size(1), k_emb.size(1), k_emb.size(2))
        # p_emb_ = p_emb.unsqueeze(2).expand(p_emb.size(0), p_emb.size(1), k_emb.size(1), p_emb.size(2))
        # i_att = self.cosSim(k_emb_, p_emb_)

        prediction = torch.argmax(i_att, dim=-1) # [B, querys]
        maxval, _ = i_att.max(dim=2, keepdim=True)
        predictions = (i_att == maxval) # [B, querys, K]

        attmap = torch.einsum('avd, bqd -> baqv', k_emb, p_emb)
        maxatt, _ = attmap.max(dim=-1) # [B1, B2, querys]: B1th sentence to B2th image
        logits = torch.sum(maxatt, dim=-1).div(num_query.unsqueeze(1).expand(maxatt.size(0), maxatt.size(1)))# [B1, B2]: B1th sentence to B2th image 求和后需要除以query的个数！
        n_obj = int(query.size(0))
        target = torch.eye(n_obj).cuda() # [b, b]

        pred_labels=[]
        for i in range(prediction.size(0)):
            for j in range(prediction.size(1)):
                pred_label = self.indexer.get_object(int(label[i,prediction[i,j]]))
                # words = ' '.join(words)
                pred_labels.append(pred_label)
        t2 = time.time()
        # print("point2", t2-t1)

        # /////修改//////
        # take largest
        if bboxes is not None: # bboxes: [B, K=64, 5]
            pred_bboxes = []
            for i in range(predictions.size(0)):
                # predictions[i]:querys x K
                select_box = bboxes[i].unsqueeze(0).expand(predictions.size(1),predictions.size(2),5).long() # querys x K x 5
                select_mask = predictions[i].unsqueeze(-1).long() # querys x K x 1
                # print(select_box.shape, select_mask.shape)
                avail_box = select_box * select_mask # # querys x K x 5
                _, maxidx = avail_box.max(dim=1) # querys x 5
                # print(maxidx)
                # maxidx[:, -1] -- [querys] location of the max bbox
                bbox = select_box[torch.arange(select_box.size(0)), maxidx[:,-1]] # querys x 5
                # print(bbox.shape)
                # print(bbox)
                pred_bboxes.append(bbox[:,:4])

            pred_bboxes =torch.stack(pred_bboxes) # [B, querys, 4]

            return pred_bboxes, pred_labels, logits, target
        else: # for debugging
            return i_att, prediction

    def similarity_score(self, q_emb, k_emb):
        '''
        q_emb: [B, querys, Q, dim]
        k_emb: [B, querys, K, dim]
        '''
        scale =1.0/np.sqrt(q_emb.size(3))
        att = torch.einsum('xyab,xybc ->xyac', q_emb, z.transpose(2,3))
        att = self.softmax(att.mul_(scale)) # [B, all_query, Q, K]

        # embedding of every phrase with regard to the labels
        q_max_att = torch.max(att, dim=3).values # [B, all_query, Q]
        q_max_norm_att = q_max_att.div(q_max_att.sum(dim=2, keepdim=True)).unsqueeze(2) # [B, all_query, 1, Q]
        p_emb = torch.einsum('xyab,xybc ->xyac', q_max_norm_att, q_emb).squeeze(2) # [B, all_query, dim]

        # l_max_att = torch.max(att, dim=2, keepdim=True).values # [B, all_query, 1, K]
        # l_max_norm_att = l_max_att.div(l_max_att.sum(dim=3,keepdim=True))# [B, all_query, 1, K]
        # l_emb = torch.einsum('xyab,xybc ->xyac', l_max_norm_att, k_emb).squeeze(2)  # [B, all_query, dim]

        # embedding of every image with regard to the phrase
        i_att = torch.einsum('xykd, xyd -> xyk', k_emb, p_emb) # [B, querys, K, dim] x [B, querys, dim] => [B, querys, K]
        i_emb = torch.einsum('xyk, xykd -> xyd', i_att, k_emb) # [B, querys, K] x [B, queys, K, dim] -> [B, querys, dim]

        return torch.einsum('xyd,xyd->xy', p_emb, i_emb) # [B, querys]

    def phrase_encode(self, q_emb, k_emb, num_obj, len_query):
        '''
        q_emb: b x querys x Q x dim
        k_emb: b x K x dim
        num_obj: b
        len_query: b x querys

        return:
        p_emb: b x querys x dim
        i_emb: b x b x querys x dim
        '''

        len_query = len_query.unsqueeze(2).repeat(1, 1, q_emb.size(3)) # b x querys -> b x querys x dim
        num_obj = num_obj.unsqueeze(1).repeat(1, k_emb.size(2)) # b -> b x dim

        p_emb = torch.sum(q_emb, dim=2).div(len_query) # b x querys x dim
        i_emb = torch.sum(k_emb, dim=1).div(num_obj) # b x dim

        p_emb[torch.isnan(p_emb)] = 0

        return p_emb, i_emb

    def forward_all(self, query, label, num_obj, len_query):
        '''
        labels: [B, K=64] pad with 0
        query: [B, all_query=32, Q=12] pad with 0

        return: match score [Batch, batch, querys, objects]
        '''
        q_emb, k_emb = self.encode(query, label) # [B, querys, Q, dim] & [B, K, dim]
        mask = (query!=0) # [B, querys, Q]
        len_query = mask.sum(dim=2) # [B, querys]
        p_emb, _ = self.phrase_encode(q_emb, k_emb, num_obj, len_query) # [b x querys x dim]
        logits = torch.einsum('bqd, akd->bqak', (p_emb, k_emb))

        n_obj = int(q_emb.size(0))
        target = torch.eye(n_obj).cuda() # [b, b]
        target = target.unsqueeze(1).unsqueeze(3).repeat(1,logits.size(1), 1, logits.size(3)) # [b, q, b, k]
        return logits, target

