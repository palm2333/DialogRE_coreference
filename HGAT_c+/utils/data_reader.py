import os
import pickle
import json
import numpy as np
from collections import Counter
import spacy
import copy

from utils import constant
from utils.config import config

nlp = spacy.load("en_core_web_sm")

def build_embedding(wv_file, vocab, wv_dim):
    vocab_size = len(vocab)
    emb = np.random.randn(vocab_size, config.embed_dim) * 0.01
    emb[constant.PAD_ID] = 0 # <pad> should be all 0

    w2id = {w: i for i, w in enumerate(vocab)}
    with open(wv_file, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = ''.join(elems[0:-wv_dim])
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]
    return emb


def load_glove_vocab(file, wv_dim):
    vocab = set()
    with open(file, encoding='utf8') as f:
        for line in f:
            elems = line.split()
            token = ''.join(elems[0:-wv_dim])
            vocab.add(token)
    return vocab


class Vocab(object):
    def __init__(self, init_wordlist, word_counter):
        self.word2id = {w:i for i,w in enumerate(init_wordlist)}
        self.id2word = {i:w for i,w in enumerate(init_wordlist)}
        self.n_words = len(init_wordlist)  
        self.word_counter = word_counter

    def map(self, token_list):
        """
        Map a list of tokens to their ids.
        """
        return [self.word2id[w] if w in self.word2id else constant.UNK_ID for w in token_list]

    def unmap(self, idx_list):
        """
        Unmap ids back to tokens.
        """
        return [self.id2word[idx] for idx in idx_list]



def GetVertexSet_rule(sents, entities):
    e_data = []
    for (entity, e_type) in entities:
        ents = entity.split(' ')
        e_len = len(ents)
        data = []
        for s_id, sent in enumerate(sents):
            sent_lower = [word.lower() for word in sent]
            if ents[0].lower() in sent_lower:
                pos_1 = sent_lower.index(ents[0].lower())
                flag = True
                for e_i, pos in enumerate(range(pos_1, pos_1 + e_len)):
                    if ents[e_i].lower() in sent_lower[pos] and len(ents[e_i]) == len(sent[pos]):
                        pass
                    else:
                        flag = False
                        break
                if flag:
                    data.append({
                        "name": entity,
                        "mention": entity,
                        "pos": [pos_1, pos_1 + e_len],
                        "sent_id": s_id,
                        "type": e_type
                    })
        if len(data) == 0:
            for s_id, sent in enumerate(sents):
                pos_1 = -1
                pos_2 = -1
                for ti, token in enumerate(sent):
                    if entity.lower().startswith((token.lower())) and pos_1 == -1:
                        pos_1 = ti
                    if entity.lower().endswith((token.lower())) and pos_1 != -1:
                        pos_2 = ti
                if pos_1 != -1 and pos_2 != -1 and pos_2 - pos_1 > 0:
                    for ind in range(pos_1 + 1, pos_2 + 1):
                        try:
                            if sent[ind].lower() in entity.lower():
                                data.append({
                                    "name": entity,
                                    "mention": entity,
                                    "pos": [pos_1, pos_2 + 1],
                                    "sent_id": s_id,
                                    "type": e_type
                                })
                                break
                        except:
                            print('')
                elif pos_1 != -1 and pos_2 != -1:
                    data.append({
                        "name": entity,
                        "mention": entity,
                        "pos": [pos_1, pos_2 + 1],
                        "sent_id": s_id,
                        "type": e_type
                    })

        bianxing = ['s', 'es', 'ing', 'ed', 'ers', '.']
        bianxing_n = 0
        while len(data) == 0 and bianxing_n < len(bianxing):
            entity_1 = entity + bianxing[bianxing_n]
            ents_1 = entity_1.split(' ')
            for s_id, sent in enumerate(sents):
                sent_lower = [word.lower() for word in sent]
                if ents_1[0].lower() in sent_lower:
                    pos_1 = sent_lower.index(ents_1[0].lower())
                    flag = True
                    for e_i, pos in enumerate(range(pos_1, pos_1 + e_len)):
                        if ents_1[e_i].lower() in sent_lower[pos] and len(ents_1[e_i]) == len(sent[pos]):
                            pass
                        else:
                            flag = False
                            break
                    if flag:
                        data.append({
                            "name": entity,
                            "mention": entity_1,
                            "pos": [pos_1, pos_1 + e_len],
                            "sent_id": s_id,
                            "type": e_type
                        })
            if len(data) == 0:
                for s_id, sent in enumerate(sents):
                    pos_1 = -1
                    pos_2 = -1
                    for ti, token in enumerate(sent):
                        if entity_1.lower().startswith((token.lower())) and pos_1 == -1:
                            pos_1 = ti
                        if entity_1.lower().endswith((token.lower())) and pos_1 != -1:
                            pos_2 = ti
                    if pos_1 != -1 and pos_2 != -1 and pos_2 - pos_1 > 0:
                        for ind in range(pos_1 + 1, pos_2 + 1):
                            try:
                                if sent[ind].lower() in entity_1.lower():
                                    data.append({
                                        "name": entity,
                                        "mention": entity_1,
                                        "pos": [pos_1, pos_2 + 1],
                                        "sent_id": s_id,
                                        "type": e_type
                                    })
                                    break
                            except:
                                print('')
                    elif pos_1 != -1 and pos_2 != -1:
                        data.append({
                            "name": entity,
                            "mention": entity_1,
                            "pos": [pos_1, pos_2 + 1],
                            "sent_id": s_id,
                            "type": e_type
                        })
            bianxing_n += 1
        if len(data) == 0 and 'director' in entity and len(sents) == 9:
            data.append({
                "name": entity,
                "mention": ' '.join(sents[7][28:29]),
                "pos": [28, 29],
                "sent_id": 7,
                "type": e_type
            })
        if len(data) == 0 and 'Dr.' in entity and len(sents) == 12:
            data.append({
                "name": entity,
                "mention": ' '.join(sents[3][13:14]),
                "pos": [13, 14],
                "sent_id": 3,
                "type": e_type
            })
        if len(data) == 0 and 'big spender' in entity and len(sents) == 7:
            data.append({
                "name": entity,
                "mention": ' '.join(sents[0][5:6]),
                "pos": [5, 6],
                "sent_id": 0,
                "type": e_type
            })
            data.append({
                "name": entity,
                "mention": ' '.join(sents[2][4:5]),
                "pos": [4, 5],
                "sent_id": 2,
                "type": e_type
            })
        if len(data) == 0 and 'Don' in entity and len(sents) == 16:
            data.append({
                "name": entity,
                "mention": ' '.join(sents[8][3:5]),
                "pos": [3, 5],
                "sent_id": 8,
                "type": e_type
            })
        if len(data) == 0 and 'man' in entity and len(sents) == 28:
            data.append({
                "name": entity,
                "mention": ' '.join(sents[3][4:5]),
                "pos": [4, 5],
                "sent_id": 3,
                "type": e_type
            })
            data.append({
                "name": entity,
                "mention": ' '.join(sents[5][15:16]),
                "pos": [15, 16],
                "sent_id": 5,
                "type": e_type
            })
            data.append({
                "name": entity,
                "mention": ' '.join(sents[10][9:10]),
                "pos": [9, 10],
                "sent_id": 10,
                "type": e_type
            })
        # try:
        #     assert len(data) != 0
        # except:
        #     print('')

        e_data.append(data)
    new_e_data = copy.deepcopy(e_data)
    for i, ent in enumerate(e_data):
        if ent!=[]:
            name = ent[0]['name']
            add_sent = []
            if 'Speaker' in name:
                for ment in ent:
                    sent_id = ment['sent_id']
                    sent = sents[sent_id]
                    if 'I' in sent:
                        item = {
                            'name': name,
                            'mention': 'I',
                            'pos': [sent.index('I'), sent.index('I') + 1],
                            'sent_id': sent_id,
                            'type': ment['type']
                        }
                        new_e_data[i].append(item)

                    if sent_id - 1 >= 0:
                        lower_sent = [sen.lower() for sen in sents[sent_id - 1]]
                        prons = ['you']
                        for pron in prons:
                            if pron in lower_sent and pron + str(sent_id - 1) not in add_sent:
                                item = {
                                    'name': name,
                                    'mention': pron,
                                    'pos': [lower_sent.index(pron), lower_sent.index(pron) + 1],
                                    'sent_id': sent_id - 1,
                                    'type': ment['type']
                                }
                                new_e_data[i].append(item)
                                add_sent.append(pron + str(sent_id - 1))
                    if sent_id + 1 < len(sents):
                        lower_sent = [sen.lower() for sen in sents[sent_id + 1]]

                        prons = ['you']
                        for pron in prons:
                            if pron in lower_sent and pron + str(sent_id + 1) not in add_sent:
                                item = {
                                    'name': name,
                                    'mention': pron,
                                    'pos': [lower_sent.index(pron), lower_sent.index(pron) + 1],
                                    'sent_id': sent_id + 1,
                                    'type': ment['type']
                                }
                                new_e_data[i].append(item)
                                add_sent.append(pron + str(sent_id + 1))

    return new_e_data


def not_visit(visited,left_idx,length):
    for i in range(left_idx,left_idx+length):
        if visited[i]==1:
            return False
    return True


def GetVertexSet_all(doc_tokens,anno_coref,entity,type):
    an_coref = []
    if len(anno_coref) != 0:
        pre_turn = -1
        for i in range(len(anno_coref["turn"])):
            turn_id = anno_coref["turn"][i] - 1
            if turn_id < len(doc_tokens):
                mention = anno_coref["mention"][i]
                an_mention = {}
                an_mention["type"] = type
                an_mention["sent_id"] = int(turn_id)
                an_mention["name"] = mention

                now_turn = anno_coref["turn"][i]
                utter = doc_tokens[now_turn - 1]
                mention_l = mention.lower()
                if pre_turn != now_turn:
                    visited = [0] * len(utter)
                    pre_turn = now_turn
                for u in range(len(utter)):
                    if mention_l.startswith(utter[u].lower()) and not_visit(visited, u, 1):
                        men_len = 1
                        temp_men = utter[u].lower()
                        while mention_l != temp_men and len(temp_men) < len(mention):
                            if u + men_len < len(utter) and not_visit(visited, u, men_len):
                                men_len += 1
                                temp_men += " "
                                temp_men += utter[u + men_len - 1].lower()
                            else:
                                break
                        if mention_l == temp_men:
                            visited[u:u + men_len] = [1] * (men_len)
                            an_mention['pos'] = [u, u + men_len]
                            an_coref.append(an_mention)
                            break

    # rule-based
    v_entities = [(entity, type)]
    rule_coref = GetVertexSet_rule(doc_tokens, v_entities)
    rule_coref=rule_coref[0]

    # conbine
    all_coref=an_coref
    for i in range(len(rule_coref)):
        rule_temp = {'type': rule_coref[i]['type'], 'sent_id': rule_coref[i]['sent_id'],
                     'pos': rule_coref[i]['pos'], 'name': rule_coref[i]['mention']}
        if rule_temp not in all_coref:
            all_coref.append(rule_temp)

    sorted_coref = sorted(all_coref, key=lambda x: (x['sent_id'], x['pos'][0]))
    return sorted_coref


def get_feats(utters, word_pairs):
    ret = {'tokens': [], 'dep_head': [], 'dep_tag':[], 'pos_tag':[], 'ner_iob':[], 'ner_type':[], 'noun_chunks':[], 'noun_chunks_root':[]}
    for utter in utters:
        if config.lower:
            utter = utter.lower()
        # break_index = utter.find(':')
        # speaker, utter = utter[:break_index], utter[break_index:]
        # speaker = ''.join(speaker.split()) # remove white space
        # utter = speaker + utter
        # # DONE: 1. unsplit speaker 2. ner type -> PER 3. change x and y
        # # DONE: pass x and y through nlp
        # for k,v in word_pairs.items():
        #     utter = utter.replace(k,v)
        utter = nlp(utter)
        ret['tokens'].append([str(token) for token in utter])
        ret['dep_head'].append( [token.head.i+1 if token.i != token.head.i else 0 for token in utter ])
        ret['dep_tag'].append([token.dep_ for token in utter])
        ret['pos_tag'].append( [token.pos_ for token in utter])
        ret['ner_iob'].append([utter[i].ent_iob_ for i in range(len(utter))])
        ret['ner_type'].append([utter[i].ent_type_ if i!=0 else 'PERSON' for i in range(len(utter))]) # hard-code ner type to be 'PER' for speaker
        ret['noun_chunks'].append([str(o) for o in utter.noun_chunks])
        ret['noun_chunks_root'].append([str(o.root) for o in utter.noun_chunks])

    return ret
    

word_pairs = {"it's":"it is", "don't":"do not", "doesn't":"does not", "didn't":"did not", "you'd":"you would", "you're":"you are", "you'll":"you will", "i'm":"i am", "they're":"they are", "that's":"that is", "what's":"what is", "couldn't":"could not", "i've":"i have", "we've":"we have", "can't":"cannot", "i'd":"i would", "i'd":"i would", "aren't":"are not", "isn't":"is not", "wasn't":"was not", "weren't":"were not", "won't":"will not", "there's":"there is", "there're":"there are"}


def load_data(filename):
    tokens = []
    word_pairs = {}
    with open(filename) as infile:
        data = json.load(infile)
    D = []
    for i in range(len(data)):
        utters = data[i][0]
        spacy_feats = get_feats(utters, word_pairs)
        for j in range(len(data[i][1])):
            d = {}
            d['us'] = utters
            d['feats'] = spacy_feats                 
            d['x_type'] = data[i][1][j]["x_type"]
            d['y_type'] = data[i][1][j]["y_type"]
            d['rid'] = data[i][1][j]["rid"]
            d['r'] = data[i][1][j]["r"]
            d['t'] = data[i][1][j]["t"]

            # d['x'] = [str(token) for token in nlp(data[i][1][j]["x"])]
            # d['x'] = ''.join(d['x']) if 'Speaker' in d['x'] else ' '.join(d['x'])
            # d['y'] = [str(token) for token in nlp(data[i][1][j]["y"])]
            # d['y'] = ''.join(d['y']) if 'Speaker' in d['y'] else ' '.join(d['y'])
            d['x'] = data[i][1][j]["x"]
            d['y'] = data[i][1][j]["y"]

            d['x_coref']=GetVertexSet_all(spacy_feats["tokens"],data[i][1][j]["x_mention"],d['x'],data[i][1][j]["x_type"])
            d['y_coref']=GetVertexSet_all(spacy_feats["tokens"],data[i][1][j]["y_mention"],d['y'],data[i][1][j]["y_type"])

            D.append(d)
        
        tokens += [oo for o in d['feats']['tokens'] for oo in o]

    return tokens, D

def load_data_c(filename):
    tokens = []
    word_pairs = {}
    with open(filename) as infile:
        data = json.load(infile)
    D = []
    for i in range(len(data)):
        utters = data[i][0]
        spacy_feats = get_feats(utters, word_pairs)
        for j in range(len(data[i][1])):
            for l in range(1, len(data[i][0])+1):
                d = {}
                d['us'] = utters[:l]
                d['feats'] = {k:v[:l] for k,v in spacy_feats.items()}
                d['x_type'] = data[i][1][j]["x_type"]
                d['y_type'] = data[i][1][j]["y_type"]
                d['rid'] = data[i][1][j]["rid"]
                d['r'] = data[i][1][j]["r"]
                d['t'] = data[i][1][j]["t"]

                # d['x'] = [str(token) for token in nlp(data[i][1][j]["x"])]
                # d['x'] = ''.join(d['x']) if 'Speaker' in d['x'] else ' '.join(d['x'])
                # d['y'] = [str(token) for token in nlp(data[i][1][j]["y"])]
                # d['y'] = ''.join(d['y']) if 'Speaker' in d['y'] else ' '.join(d['y'])
                d['x'] = data[i][1][j]["x"]
                d['y'] = data[i][1][j]["y"]
                d['x_coref'] = GetVertexSet_all(d['feats']['tokens'], data[i][1][j]["x_mention"], d['x'],
                                                data[i][1][j]["x_type"])
                d['y_coref'] = GetVertexSet_all(d['feats']['tokens'], data[i][1][j]["y_mention"], d['y'],
                                                data[i][1][j]["y_type"])
                D.append(d)
        
        tokens += [oo for o in d['feats']['tokens'] for oo in o]

    return tokens, D


def build_vocab(tokens, glove_vocab, min_freq):
    """ build vocab from tokens and glove words. """
    counter = Counter(t for t in tokens)
    # if min_freq > 0, use min_freq, otherwise keep all glove words
    if config.min_freq > 0:
        v = sorted([t for t in counter if counter.get(t) >= min_freq], key=counter.get, reverse=True)
    else:
        v = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    # add special tokens and entity mask tokens
    v = constant.VOCAB_PREFIX + v
    print("vocab built with {}/{} words.".format(len(v), len(counter)))
    return v, counter

def count_oov(tokens, vocab):
    c = Counter(t for t in tokens)
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total, total-matched


def load_dataset():
    if os.path.exists(config.proce_f):
        print("LOADING dialogre dataset")
        with open(config.proce_f, "rb") as f: [train_data, dev_data, test_data, vocab] = pickle.load(f)
        return train_data, dev_data, test_data, vocab

    print("Preprocessing dialogre dataset... (This can take some time)")
    # load files
    print("loading files...")
    train_tokens, train_data = load_data(config.train_f)
    dev_tokens, dev_data = load_data(config.val_f)
    test_tokens, test_data = load_data(config.test_f)

    # load glove
    print("loading glove...")
    glove_vocab = load_glove_vocab(config.glove_f, config.embed_dim)
    print("{} words loaded from glove.".format(len(glove_vocab)))
    
    print("building vocab...")

    # TODO: The vocab should contain all 3 splits? 
    v, v_counter = build_vocab(train_tokens + dev_tokens + test_tokens, glove_vocab, config.min_freq)

    print("calculating oov...")
    datasets = {'train': train_tokens, 'dev': dev_tokens, 'test': test_tokens}
    for dname, d in datasets.items():
        total, oov = count_oov(d, v)
        print("{} oov: {}/{} ({:.2f}%)".format(dname, oov, total, oov*100.0/total))
    
    print("building embeddings...")
    embedding = build_embedding(config.glove_f, v, config.embed_dim)
    print("embedding size: {} x {}".format(*embedding.shape))

    print("dumping to files...")
    with open(config.proce_f, 'wb') as outfile:
        vocab = Vocab(v, v_counter)
        pickle.dump([train_data, dev_data, test_data, vocab], outfile)
    np.save(config.embed_f, embedding)
    print("all done.")

    return train_data, dev_data, test_data, vocab

def load_dataset_c():
    if os.path.exists(config.proce_f_c):
        print("LOADING dialogre dataset for conversation setting")
        with open(config.proce_f_c, "rb") as f: [dev_data, test_data] = pickle.load(f)
        return dev_data, test_data

    print("Preprocessing dialogre dataset... (This can take some time)")
    # load files
    print("loading files...")
    dev_tokens, dev_data = load_data_c(config.val_f)
    test_tokens, test_data = load_data_c(config.test_f)

    print("dumping to files...")
    with open(config.proce_f_c, 'wb') as outfile:
        pickle.dump([dev_data, test_data], outfile)
    print("all done.")

    return dev_data, test_data

def get_original_data(fn):
    with open(fn, "r", encoding='utf8') as f:
        data = json.load(f)    
    for i in range(len(data)):
        for j in range(len(data[i][1])):
            for k in range(len(data[i][1][j]["rid"])):
                data[i][1][j]["rid"][k] -= 1
    return data

if __name__ == "__main__":
    load_dataset()
    load_data_c()
