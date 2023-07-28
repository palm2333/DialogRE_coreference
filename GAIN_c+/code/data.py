import json
import math
import os
import pickle
import random
from collections import defaultdict

import dgl
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import *

from models.GAIN import Bert
from utils import get_cuda

IGNORE_INDEX = -100


class DGLREDataset(IterableDataset):

    def __init__(self, src_file, save_file, word2id, ner2id, rel2id,
                 dataset_type='train', instance_in_train=None, opt=None):

        super(DGLREDataset, self).__init__()

        # record training set mention triples
        self.instance_in_train = set([]) if instance_in_train is None else instance_in_train
        self.data = None
        self.document_max_length = 512
        self.INTRA_EDGE = 0
        self.INTER_EDGE = 1
        self.LOOP_EDGE = 2
        self.count = 0

        print('Reading data from {}.'.format(src_file))
        if os.path.exists(save_file):
            with open(file=save_file, mode='rb') as fr:
                info = pickle.load(fr)
                self.data = info['data']
                self.instance_in_train = info['intrain_set']
            print('load preprocessed data from {}.'.format(save_file))

        else:
            with open(file=src_file, mode='r', encoding='utf-8') as fr:
                ori_data = json.load(fr)
            print('loading..')
            self.data = []

            for i, doc in enumerate(ori_data):

                entity_list_temp, labels, sentences = \
                    doc['vertexSet'], doc.get('labels', []), doc['sents']

                Ls = [0]
                L = 0
                for x in sentences:
                    L += len(x)
                    Ls.append(L)


                entity_list = []
                for j in range(len(entity_list_temp)):
                    entity_list_j = []
                    for k in range(len(entity_list_temp[j])):
                        sent_id = int(entity_list_temp[j][k]['sent_id'])
                        entity_list_temp[j][k]['sent_id'] = sent_id

                        dl = Ls[sent_id]
                        pos0, pos1 = entity_list_temp[j][k]['pos']

                        if pos1 + dl < 512:
                            el_jk = entity_list_temp[j][k]
                            el_jk['global_pos'] = (pos0 + dl, pos1 + dl)
                            entity_list_j.append(el_jk)

                    if entity_list_j == []:
                        entity_list_j.append(entity_list_temp[j][0])
                        ran = random.randint(0, 511)
                        entity_list_j[0]['global_pos'] = (ran, ran + 1)
                    entity_list.append(entity_list_j)

                # generate positive examples
                train_triple = []
                new_labels = []
                for label in labels:
                    head, tail, relation = label['h'], label['t'], label['r']
                    assert (relation in rel2id), 'no such relation {} in rel2id'.format(relation)
                    # 修改
                    label['r'] = rel2id[relation]-1

                    train_triple.append((head, tail))

                    label['in_train'] = False

                    # record training set mention triples and mark it for dev and test set
                    for n1 in entity_list[head]:
                        for n2 in entity_list[tail]:
                            mention_triple = (n1['name'], n2['name'], relation)
                            if dataset_type == 'train':
                                self.instance_in_train.add(mention_triple)
                            else:
                                if mention_triple in self.instance_in_train:
                                    label['in_train'] = True
                                    break

                    new_labels.append(label)

                # generate negative examples
                na_triple = []
                for j in range(len(entity_list)):
                    for k in range(len(entity_list)):
                        if j != k and (j, k) not in train_triple:
                            na_triple.append((j, k))

                # generate document ids
                words = []
                for sentence in sentences:
                    for word in sentence:
                        words.append(word)
                if len(words) > self.document_max_length:
                    words = words[:self.document_max_length]

                pos_id_all = np.array([], dtype=np.int32)
                ner_id_all = np.array([], dtype=np.int32)
                mention_id_all = np.array([], dtype=np.int32)

                word_id = np.zeros((self.document_max_length,), dtype=np.int32)
                for iii, w in enumerate(words):
                    word = word2id.get(w.lower(), word2id['UNK'])
                    word_id[iii] = word

                entity2mention = defaultdict(list)
                mention_idx = 1
                for idx, vertex in enumerate(entity_list, 1):
                    pos_id = np.zeros((self.document_max_length,), dtype=np.int32)
                    ner_id = np.zeros((self.document_max_length,), dtype=np.int32)
                    mention_id = np.zeros((self.document_max_length,), dtype=np.int32)
                    for v in vertex:
                        sent_id, (pos0, pos1), ner_type = v['sent_id'], v['global_pos'], v['type']

                        if pos0 >= len(pos_id):
                            continue

                        pos_id[pos0:pos1] = idx
                        ner_id[pos0:pos1] = ner2id[ner_type]
                        mention_id[pos0:pos1] = mention_idx
                        entity2mention[idx].append(mention_idx)
                        mention_idx += 1

                    pos_id_all = np.append(pos_id_all, pos_id)
                    ner_id_all = np.append(ner_id_all, ner_id)
                    mention_id_all = np.append(mention_id_all, mention_id)
                pos_id_all = np.resize(pos_id_all, (idx, 512))
                ner_id_all = np.resize(ner_id_all, (idx, 512))
                mention_id_all = np.resize(mention_id_all, (idx, 512))


                idx = len(entity_list)
                for j in range(1, idx + 1):
                    replace_i = 0
                    if entity2mention[j] == []:
                        entity2mention[j].append(mention_idx)
                        while mention_id_all[j - 1][replace_i] != 0:
                            replace_i += 1
                        mention_id_all[j - 1][replace_i] = mention_idx
                        pos_id_all[j - 1][replace_i] = j
                        ner_id_all[j - 1][replace_i] = ner2id[vertex[0]['type']]
                        mention_idx += 1
                assert len(entity2mention) == len(entity_list)

                new_Ls = [0]
                for ii in range(1, len(Ls)):
                    if Ls[ii] < 512:
                        new_Ls.append(Ls[ii])
                    else:
                        break
                new_Ls.append(512)
                Ls = new_Ls

                # construct graph
                graph = self.create_graph(Ls, mention_id_all, pos_id_all, entity2mention)

                # construct entity graph & path
                entity_graph, path = self.create_entity_graph(Ls, pos_id_all, entity2mention)

                assert pos_id_all.max() == len(entity_list)
                assert mention_id_all.max() == graph.number_of_nodes() - 1

                overlap = doc.get('overlap_entity_pair', [])
                new_overlap = [tuple(item) for item in overlap]

                self.data.append({
                    'entities': entity_list,
                    'labels': new_labels,
                    'na_triple': na_triple,
                    'word_id': word_id,
                    'pos_id_all': pos_id_all,
                    'ner_id_all': ner_id_all,
                    'mention_id_all': mention_id_all,
                    'entity2mention': entity2mention,
                    'graph': graph,
                    'entity_graph': entity_graph,
                    'path': path,
                    'overlap': new_overlap
                })

            # save data
            with open(file=save_file, mode='wb') as fw:
                pickle.dump({'data': self.data, 'intrain_set': self.instance_in_train}, fw)
            print('finish reading {} and save preprocessed data to {}.'.format(src_file, save_file))

        if opt.k_fold != "none":
            k_fold = opt.k_fold.split(',')
            k, total = float(k_fold[0]), float(k_fold[1])
            a = (k - 1) / total * len(self.data)
            b = k / total * len(self.data)
            self.data = self.data[:a] + self.data[b:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

    def create_graph(self, Ls, mention_id_all, entity_id_all, entity2mention):

        d = defaultdict(list)

        # add intra-entity edges
        for _, mentions in entity2mention.items():
            for i in range(len(mentions)):
                for j in range(i + 1, len(mentions)):
                    d[('node', 'intra', 'node')].append((mentions[i], mentions[j]))
                    d[('node', 'intra', 'node')].append((mentions[j], mentions[i]))

        if d[('node', 'intra', 'node')] == []:
            d[('node', 'intra', 'node')].append((entity2mention[1][0], 0))

        for i in range(1, len(Ls)):
            tmp = dict()

            for k in range(len(entity2mention)):
                for j in range(Ls[i - 1], Ls[i]):
                    if mention_id_all[k][j] != 0:
                        tmp[mention_id_all[k][j]] = entity_id_all[k][j]
            mention_entity_info = [(k, v) for k, v in tmp.items()]

            # add self-loop & to-globle-node edges
            for m in range(len(mention_entity_info)):
                # self-loop
                # d[('node', 'loop', 'node')].append((mention_entity_info[m][0], mention_entity_info[m][0]))

                # to global node
                d[('node', 'global', 'node')].append((mention_entity_info[m][0], 0))
                d[('node', 'global', 'node')].append((0, mention_entity_info[m][0]))

            # add inter edges
            for m in range(len(mention_entity_info)):
                for n in range(m + 1, len(mention_entity_info)):
                    if mention_entity_info[m][1] != mention_entity_info[n][1]:
                        # inter edge
                        d[('node', 'inter', 'node')].append((mention_entity_info[m][0], mention_entity_info[n][0]))
                        d[('node', 'inter', 'node')].append((mention_entity_info[n][0], mention_entity_info[m][0]))

        # add self-loop for global node
        # d[('node', 'loop', 'node')].append((0, 0))
        if d[('node', 'inter', 'node')] == []:
            d[('node', 'inter', 'node')].append((entity2mention[1][0], 0))

        graph = dgl.heterograph(d)

        return graph

    def create_entity_graph(self, Ls, entity_id_all, entity2mention):

        graph = dgl.DGLGraph()
        graph.add_nodes(entity_id_all.max())

        d = defaultdict(set)

        for i in range(1, len(Ls)):
            tmp = set()

            for k in range(len(entity2mention)):
                for j in range(Ls[i - 1], Ls[i]):
                    if entity_id_all[k][j] != 0:
                        tmp.add(entity_id_all[k][j])
            tmp = list(tmp)
            for ii in range(len(tmp)):
                for jj in range(ii + 1, len(tmp)):
                    d[tmp[ii] - 1].add(tmp[jj] - 1)
                    d[tmp[jj] - 1].add(tmp[ii] - 1)
        a = []
        b = []
        for k, v in d.items():
            for vv in v:
                a.append(k)
                b.append(vv)
        graph.add_edges(a, b)

        path = dict()
        for i in range(0, graph.number_of_nodes()):
            for j in range(i + 1, graph.number_of_nodes()):
                a = set(graph.successors(i).numpy())
                b = set(graph.successors(j).numpy())
                c = [val + 1 for val in list(a & b)]
                path[(i + 1, j + 1)] = c

        return graph, path


class BERTDGLREDataset(IterableDataset):

    def __init__(self, src_file, save_file, word2id, ner2id, rel2id,
                 dataset_type='train', instance_in_train=None, opt=None):

        super(BERTDGLREDataset, self).__init__()

        # record training set mention triples
        self.instance_in_train = set([]) if instance_in_train is None else instance_in_train
        self.data = None
        self.document_max_length = 512
        self.INFRA_EDGE = 0
        self.INTER_EDGE = 1
        self.LOOP_EDGE = 2
        self.count = 0

        print('Reading data from {}.'.format(src_file))
        if os.path.exists(save_file):
            with open(file=save_file, mode='rb') as fr:
                info = pickle.load(fr)
                self.data = info['data']
                self.instance_in_train = info['intrain_set']
            print('load preprocessed data from {}.'.format(save_file))

        else:
            bert = Bert(BertModel, 'bert-base-uncased', opt.bert_path)

            with open(file=src_file, mode='r', encoding='utf-8') as fr:
                ori_data = json.load(fr)
            print('loading..')
            self.data = []

            # max_entity_num=0
            for i, doc in enumerate(ori_data):
                # if len(doc['vertexSet'])>max_entity_num:
                #     max_entity_num=len(doc['vertexSet'])
                #     print(max_entity_num)

                entity_list_temp, labels, sentences = \
                    doc['vertexSet'], doc.get('labels', []), doc['sents']

                Ls = [0]
                L = 0
                for x in sentences:
                    L += len(x)
                    Ls.append(L)

                entity_list = []
                for j in range(len(entity_list_temp)):
                    entity_list_j = []
                    for k in range(len(entity_list_temp[j])):
                        sent_id = int(entity_list_temp[j][k]['sent_id'])
                        entity_list_temp[j][k]['sent_id'] = sent_id

                        dl = Ls[sent_id]
                        pos0, pos1 = entity_list_temp[j][k]['pos']

                        if pos1 + dl < 512:
                            el_jk = entity_list_temp[j][k]
                            el_jk['global_pos'] = (pos0 + dl, pos1 + dl)
                            entity_list_j.append(el_jk)

                    if entity_list_j == []:
                        entity_list_j.append(entity_list_temp[j][0])
                        ran = random.randint(0, 511)
                        entity_list_j[0]['global_pos'] = (ran, ran + 1)
                    entity_list.append(entity_list_j)


                # generate positive examples
                train_triple = []
                new_labels = []
                for label in labels:
                    head, tail, relation = label['h'], label['t'], label['r']
                    assert (relation in rel2id), 'no such relation {} in rel2id'.format(relation)

                    label['r'] = rel2id[relation]-1

                    train_triple.append((head, tail))

                    label['in_train'] = False

                    # record training set mention triples and mark it for dev and test set
                    for n1 in entity_list[head]:
                        for n2 in entity_list[tail]:
                            mention_triple = (n1['name'], n2['name'], relation)
                            if dataset_type == 'train':
                                self.instance_in_train.add(mention_triple)
                            else:
                                if mention_triple in self.instance_in_train:
                                    label['in_train'] = True
                                    break

                    new_labels.append(label)

                # generate negative examples
                na_triple = []
                for j in range(len(entity_list)):
                    for k in range(len(entity_list)):
                        if j != k and (j, k) not in train_triple:
                            na_triple.append((j, k))

                # generate document ids
                words = []
                for sentence in sentences:
                    for word in sentence:
                        words.append(word)

                bert_token, bert_starts, bert_subwords = bert.subword_tokenize_to_ids(words)

                pos_id_all = np.array([], dtype=np.int32)
                ner_id_all = np.array([], dtype=np.int32)
                mention_id_all=np.array([], dtype=np.int32)
                word_id = np.zeros((self.document_max_length,), dtype=np.int32)
                word_id[:] = bert_token[0]

                entity2mention = defaultdict(list)
                mention_idx = 1
                for idx, vertex in enumerate(entity_list, 1):
                    pos_id = np.zeros((self.document_max_length,), dtype=np.int32)
                    ner_id = np.zeros((self.document_max_length,), dtype=np.int32)
                    mention_id = np.zeros((self.document_max_length,), dtype=np.int32)
                    for v in vertex:
                        sent_id, (pos0, pos1), ner_type = v['sent_id'], v['global_pos'], v['type']

                        pos0 = bert_starts[pos0]
                        pos1 = bert_starts[pos1] if pos1 < len(bert_starts) else 1024
                        if pos0 >= len(pos_id):
                            continue

                        pos_id[pos0:pos1] = idx
                        ner_id[pos0:pos1] = ner2id[ner_type]
                        mention_id[pos0:pos1] = mention_idx

                        entity2mention[idx].append(mention_idx)
                        mention_idx += 1
                    pos_id_all=np.append(pos_id_all,pos_id)
                    ner_id_all=np.append(ner_id_all, ner_id)
                    mention_id_all=np.append(mention_id_all, mention_id)
                pos_id_all=np.resize(pos_id_all,(idx,512))
                ner_id_all = np.resize(ner_id_all, (idx, 512))
                mention_id_all = np.resize(mention_id_all, (idx, 512))

                idx = len(entity_list)

                for j in range(1, idx + 1):
                    replace_i = 0
                    if entity2mention[j] == []:
                        entity2mention[j].append(mention_idx)
                        while mention_id_all[j-1][replace_i] != 0:
                            replace_i += 1
                        mention_id_all[j-1][replace_i] = mention_idx
                        pos_id_all[j-1][replace_i] = j
                        ner_id_all[j-1][replace_i] = ner2id[vertex[0]['type']]
                        mention_idx += 1

                assert len(entity2mention) == len(entity_list)
                for j in range(1, idx + 1):
                    if entity2mention[j] == []:
                        assert 1==2

                new_Ls = [0]
                for ii in range(1, len(Ls)):
                    new_Ls.append(bert_starts[Ls[ii]] if Ls[ii] < len(bert_starts) else len(bert_subwords))
                new_Ls.append(512)
                Ls = new_Ls

                # construct graph
                graph = self.create_graph(Ls, mention_id_all, pos_id_all, entity2mention)

                # construct entity graph & path
                entity_graph, path = self.create_entity_graph(Ls, pos_id_all, entity2mention)

                assert pos_id_all.max() == len(entity_list)
                assert mention_id_all.max() == graph.number_of_nodes() - 1

                overlap = doc.get('overlap_entity_pair', [])
                new_overlap = [tuple(item) for item in overlap]

                self.data.append({
                    'entities': entity_list,
                    'labels': new_labels,
                    'na_triple': na_triple,
                    'word_id': word_id,
                    'pos_id_all': pos_id_all,
                    'ner_id_all': ner_id_all,
                    'mention_id_all': mention_id_all,
                    'entity2mention': entity2mention,
                    'graph': graph,
                    'entity_graph': entity_graph,
                    'path': path,
                    'overlap': new_overlap
                })

            # save data
            with open(file=save_file, mode='wb') as fw:
                pickle.dump({'data': self.data, 'intrain_set': self.instance_in_train}, fw)
            print('finish reading {} and save preprocessed data to {}.'.format(src_file, save_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

    def create_graph(self, Ls, mention_id_all, entity_id_all, entity2mention):

        d = defaultdict(list)

        # add intra edges
        for _, mentions in entity2mention.items():
            for i in range(len(mentions)):
                for j in range(i + 1, len(mentions)):
                    d[('node', 'intra', 'node')].append((mentions[i], mentions[j]))
                    d[('node', 'intra', 'node')].append((mentions[j], mentions[i]))

        if d[('node', 'intra', 'node')] == []:
            d[('node', 'intra', 'node')].append((entity2mention[1][0], 0))

        for i in range(1, len(Ls)):
            tmp = dict()
            for k in range(len(entity2mention)):
                for j in range(Ls[i - 1], Ls[i]):
                    if mention_id_all[k][j] != 0:
                        tmp[mention_id_all[k][j]] = entity_id_all[k][j]
            mention_entity_info = [(k, v) for k, v in tmp.items()]

            for m in range(len(mention_entity_info)):
                # self-loop
                # d[('node', 'loop', 'node')].append((mention_entity_info[m][0], mention_entity_info[m][0]))

                # to global node
                d[('node', 'global', 'node')].append((mention_entity_info[m][0], 0))
                d[('node', 'global', 'node')].append((0, mention_entity_info[m][0]))

            # add inter edges
            for m in range(len(mention_entity_info)):
                for n in range(m + 1, len(mention_entity_info)):
                    if mention_entity_info[m][1] != mention_entity_info[n][1]:
                        # inter edge
                        d[('node', 'inter', 'node')].append((mention_entity_info[m][0], mention_entity_info[n][0]))
                        d[('node', 'inter', 'node')].append((mention_entity_info[n][0], mention_entity_info[m][0]))

        # add self-loop for global node
        # d[('node', 'loop', 'node')].append((0, 0))
        if d[('node', 'inter', 'node')] == []:
            d[('node', 'inter', 'node')].append((entity2mention[1][0], 0))

        graph = dgl.heterograph(d)

        return graph

    def create_entity_graph(self, Ls, entity_id_all, entity2mention):

        graph = dgl.DGLGraph()
        graph.add_nodes(entity_id_all.max())

        d = defaultdict(set)

        for i in range(1, len(Ls)):
            tmp = set()
            for k in range(len(entity2mention)):
                for j in range(Ls[i - 1], Ls[i]):
                    if entity_id_all[k][j] != 0:
                        tmp.add(entity_id_all[k][j])
                        break
            tmp = list(tmp)
            for ii in range(len(tmp)):
                for jj in range(ii + 1, len(tmp)):
                    d[tmp[ii] - 1].add(tmp[jj] - 1)
                    d[tmp[jj] - 1].add(tmp[ii] - 1)
        a = []
        b = []
        for k, v in d.items():
            for vv in v:
                a.append(k)
                b.append(vv)
        graph.add_edges(a, b)

        path = dict()
        for i in range(0, graph.number_of_nodes()):
            for j in range(i + 1, graph.number_of_nodes()):
                a = set(graph.successors(i).numpy())
                b = set(graph.successors(j).numpy())
                c = [val + 1 for val in list(a & b)]
                path[(i + 1, j + 1)] = c

        return graph, path


class DGLREDataloader(DataLoader):

    def __init__(self, dataset, batch_size, shuffle=False, h_t_limit_per_batch=300, h_t_limit=1722, relation_num=37,
                 max_length=512, negativa_alpha=0.0, dataset_type='train'):
        super(DGLREDataloader, self).__init__(dataset, batch_size=batch_size)
        self.shuffle = shuffle
        self.length = len(self.dataset)
        self.max_length = max_length
        self.negativa_alpha = negativa_alpha
        self.dataset_type = dataset_type

        self.h_t_limit_per_batch = h_t_limit_per_batch
        self.h_t_limit = h_t_limit
        self.relation_num = relation_num
        self.dis2idx = np.zeros((512), dtype='int64')
        self.dis2idx[1] = 1
        self.dis2idx[2:] = 2
        self.dis2idx[4:] = 3
        self.dis2idx[8:] = 4
        self.dis2idx[16:] = 5
        self.dis2idx[32:] = 6
        self.dis2idx[64:] = 7
        self.dis2idx[128:] = 8
        self.dis2idx[256:] = 9
        self.dis_size = 20

        self.order = list(range(self.length))


        self.max_entity_num=20

    def __iter__(self):
        # shuffle
        random.seed(52)
        if self.shuffle:
            random.shuffle(self.order)
            self.data = [self.dataset[idx] for idx in self.order]
        else:
            self.data = self.dataset
        batch_num = math.ceil(self.length / self.batch_size)
        self.batches = [self.data[idx * self.batch_size: min(self.length, (idx + 1) * self.batch_size)]
                        for idx in range(0, batch_num)]
        self.batches_order = [self.order[idx * self.batch_size: min(self.length, (idx + 1) * self.batch_size)]
                              for idx in range(0, batch_num)]

        # begin
        context_word_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        context_pos_ids_all = torch.LongTensor(self.batch_size, self.max_entity_num, self.max_length).cpu()
        context_ner_ids_all = torch.LongTensor(self.batch_size, self.max_entity_num, self.max_length).cpu()
        context_mention_ids_all = torch.LongTensor(self.batch_size, self.max_entity_num, self.max_length).cpu()
        context_word_mask = torch.LongTensor(self.batch_size, self.max_length).cpu()
        context_word_length = torch.LongTensor(self.batch_size).cpu()
        ht_pairs = torch.LongTensor(self.batch_size, self.h_t_limit, 2).cpu()
        relation_multi_label = torch.Tensor(self.batch_size, self.h_t_limit, self.relation_num).cpu()
        relation_mask = torch.Tensor(self.batch_size, self.h_t_limit).cpu()
        relation_label = torch.LongTensor(self.batch_size, self.h_t_limit).cpu()
        ht_pair_distance = torch.LongTensor(self.batch_size, self.h_t_limit).cpu()

        for idx, minibatch in enumerate(self.batches):
            cur_bsz = len(minibatch)
            for mapping in [context_word_ids, context_pos_ids_all, context_ner_ids_all, context_mention_ids_all,
                            context_word_mask, context_word_length,
                            ht_pairs, ht_pair_distance, relation_multi_label, relation_mask, relation_label]:
                if mapping is not None:
                    mapping.zero_()

            relation_label.fill_(IGNORE_INDEX)

            max_h_t_cnt = 0

            label_list = []
            L_vertex = []
            indexes = []
            graph_list = []
            entity_graph_list = []
            entity2mention_table = []
            path_table = []
            overlaps = []

            for i, example in enumerate(minibatch):
                entities, labels, na_triple, word_id, pos_id_all, ner_id_all, mention_id_all, entity2mention, graph, entity_graph, path = \
                    example['entities'], example['labels'], example['na_triple'], \
                    example['word_id'], example['pos_id_all'], example['ner_id_all'], example['mention_id_all'], example[
                        'entity2mention'], example['graph'], example['entity_graph'], example['path']
                graph_list.append(graph)
                entity_graph_list.append(entity_graph)
                path_table.append(path)
                overlaps.append(example['overlap'])


                entity2mention_t = get_cuda(torch.zeros((pos_id_all.max() + 1, mention_id_all.max() + 1)))
                for e, ms in entity2mention.items():
                    for m in ms:
                        entity2mention_t[e, m] = 1
                entity2mention_table.append(entity2mention_t)

                L = len(entities)
                word_num = word_id.shape[0]

                entity_num=pos_id_all.shape[0]

                context_word_ids[i, :word_num].copy_(torch.from_numpy(word_id))
                context_pos_ids_all[i, :entity_num, :word_num].copy_(torch.from_numpy(pos_id_all))
                context_ner_ids_all[i, :entity_num, :word_num].copy_(torch.from_numpy(ner_id_all))
                context_mention_ids_all[i, :entity_num, :word_num].copy_(torch.from_numpy(mention_id_all))

                idx2label = defaultdict(list)
                label_set = {}
                for label in labels:
                    head, tail, relation, intrain = \
                        label['h'], label['t'], label['r'], label['in_train']
                    idx2label[(head, tail)].append(relation)
                    label_set[(head, tail, relation)] = intrain

                label_list.append(label_set)

                if self.dataset_type == 'train':
                    train_tripe = list(idx2label.keys())
                    for j, (h_idx, t_idx) in enumerate(train_tripe):
                        hlist, tlist = entities[h_idx], entities[t_idx]
                        ht_pairs[i, j, :] = torch.Tensor([h_idx + 1, t_idx + 1])
                        label = idx2label[(h_idx, t_idx)]

                        delta_dis = hlist[0]['global_pos'][0] - tlist[0]['global_pos'][0]
                        if delta_dis < 0:
                            ht_pair_distance[i, j] = -int(self.dis2idx[-delta_dis]) + self.dis_size // 2
                        else:
                            ht_pair_distance[i, j] = int(self.dis2idx[delta_dis]) + self.dis_size // 2

                        for r in label:
                            relation_multi_label[i, j, r] = 1

                        relation_mask[i, j] = 1
                        rt = np.random.randint(len(label))
                        relation_label[i, j] = label[rt]

                    max_h_t_cnt = max(max_h_t_cnt, len(train_tripe))
                else:
                    train_tripe = list(idx2label.keys())
                    for j, (h_idx, t_idx) in enumerate(train_tripe):
                        hlist, tlist = entities[h_idx], entities[t_idx]
                        ht_pairs[i, j, :] = torch.Tensor([h_idx + 1, t_idx + 1])
                        relation_mask[i, j] = 1

                        delta_dis = hlist[0]['global_pos'][0] - tlist[0]['global_pos'][0]
                        if delta_dis < 0:
                            ht_pair_distance[i, j] = -int(self.dis2idx[-delta_dis]) + self.dis_size // 2
                        else:
                            ht_pair_distance[i, j] = int(self.dis2idx[delta_dis]) + self.dis_size // 2
                    max_h_t_cnt = max(max_h_t_cnt, len(train_tripe))
                    L_vertex.append(L)
                    indexes.append(self.batches_order[idx][i])

            context_word_mask = context_word_ids > 0
            context_word_length = context_word_mask.sum(1)
            batch_max_length = context_word_length.max()
            batch_entity_num=context_pos_ids_all.max()

            yield {'context_idxs': get_cuda(context_word_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'context_pos_all': get_cuda(context_pos_ids_all[:cur_bsz, :batch_entity_num, :batch_max_length].contiguous()),
                   'context_ner_all': get_cuda(context_ner_ids_all[:cur_bsz, :batch_entity_num, :batch_max_length].contiguous()),
                   'context_mention_all': get_cuda(context_mention_ids_all[:cur_bsz, :batch_entity_num, :batch_max_length].contiguous()),
                   'context_word_mask': get_cuda(context_word_mask[:cur_bsz, :batch_max_length].contiguous()),
                   'context_word_length': get_cuda(context_word_length[:cur_bsz].contiguous()),
                   'h_t_pairs': get_cuda(ht_pairs[:cur_bsz, :max_h_t_cnt, :2]),
                   'relation_label': get_cuda(relation_label[:cur_bsz, :max_h_t_cnt]).contiguous(),
                   'relation_multi_label': get_cuda(relation_multi_label[:cur_bsz, :max_h_t_cnt]),
                   'relation_mask': get_cuda(relation_mask[:cur_bsz, :max_h_t_cnt]),
                   'ht_pair_distance': get_cuda(ht_pair_distance[:cur_bsz, :max_h_t_cnt]),
                   'labels': label_list,
                   'L_vertex': L_vertex,
                   'indexes': indexes,
                   'graphs': graph_list,
                   'entity2mention_table': entity2mention_table,
                   'entity_graphs': entity_graph_list,
                   'path_table': path_table,
                   'overlaps': overlaps
                   }
