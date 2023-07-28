import numpy as np
import json
import copy
import tokenization

def GetVertexSet(sents, entities):
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
        try:
            assert len(data) != 0
        except:
            print('')

        e_data.append(data)
    new_e_data = copy.deepcopy(e_data)
    for i, ent in enumerate(e_data):
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



if __name__=="__main__":
    path="../DialogRE_c+"
    tokenizer = tokenization.BasicTokenizer(do_lower_case=False)
    tmp_all=0
    all=0
    for sid in range(3):
        with open(path + ["/train.json", "/dev.json", "/test.json"][sid], "r", encoding="utf8") as f:
            datas = json.load(f)
            prepro_datas=[]
            for data in datas:
                pp_data = {}
                # doc tokens
                doc = data[0]
                d_tokens = []
                for sen in doc:
                    d_tokens.append(tokenizer.tokenize(sen))
                pp_data['sents'] = d_tokens

                rel_triples = data[1]
                xy2id = {}
                id2type={}
                count = 0
                # {entity:type,mention,turn,tokenp}
                ori_entities_mention = []
                for triple in rel_triples:
                    if triple["x"] not in xy2id:
                        xy2id[triple["x"]] = count
                        id2type[count]=triple["x_type"]
                        count += 1
                        if triple["x_mention"]!=[]:
                            temp_triple=triple["x_mention"]
                            temp_triple['type']=triple["x_type"]
                            ori_entities_mention.append(temp_triple)
                        else:
                            ori_entities_mention.append({})

                    if triple["y"] not in xy2id:
                        xy2id[triple["y"]] = count
                        id2type[count] = triple["y_type"]
                        count += 1
                        if triple["y_mention"]!=[]:
                            temp_triple = triple["y_mention"]
                            temp_triple['type'] = triple["y_type"]
                            ori_entities_mention.append(temp_triple)
                        else:
                            ori_entities_mention.append({})

                # Converted entity information
                convert_entities_info=[]
                for entity_info in ori_entities_mention:
                    an_entity=[]
                    if len(entity_info)!=0:
                        pre_turn = -1
                        for i in range(len(entity_info["turn"])):
                            an_mention={}
                            an_mention["type"] = entity_info["type"]
                            turn_id = entity_info["turn"][i] - 1
                            mention=entity_info["mention"][i]
                            an_mention["sent_id"] = int(turn_id)
                            an_mention["name"] = mention

                            # Find the position of mention in sentence
                            now_turn = entity_info["turn"][i]
                            utter=d_tokens[now_turn - 1]
                            mention_l=mention.lower()
                            if pre_turn != now_turn:
                                visited = [0] * len(utter)
                                pre_turn = now_turn
                            for u in range(len(utter)):
                                if mention_l.startswith(utter[u].lower()) and not_visit(visited, u, 1):
                                    men_len = 1
                                    temp_men = utter[u].lower()
                                    while mention_l != temp_men and len(temp_men)<len(mention):
                                        if u + men_len < len(utter) and not_visit(visited, u, men_len):
                                            men_len += 1
                                            temp_men += " "
                                            temp_men += utter[u + men_len - 1].lower()
                                        else:
                                            break
                                    if mention_l == temp_men:
                                        visited[u:u + men_len] = [1] * (men_len)
                                        an_mention['pos'] =[u,u+men_len]
                                        an_entity.append(an_mention)
                                        break
                    convert_entities_info.append(an_entity)

                # rule-based
                xys = list(xy2id.keys())
                v_entities=[(xys[i],id2type[i]) for i in range(len(xy2id)) ]
                rule_vert=GetVertexSet(d_tokens, v_entities)

                # combine
                for i in range(len(rule_vert)):
                    for j in range(len(rule_vert[i])):
                        rule_temp={'type':rule_vert[i][j]['type'],'sent_id':rule_vert[i][j]['sent_id'],'pos':rule_vert[i][j]['pos'],'name':rule_vert[i][j]['mention']}
                        if rule_temp not in convert_entities_info[i]:
                            convert_entities_info[i].append(rule_temp)
                sort_entities_info=[]
                for i in range(len(convert_entities_info)):
                    an_entity_info=convert_entities_info[i]
                    st=sorted(an_entity_info,key=lambda x: (x['sent_id'], x['pos'][0]))
                    sort_entities_info.append(st)
                pp_data['vertexSet'] = sort_entities_info

                labels=[]
                for triple in rel_triples:
                    x_id = xy2id[triple["x"]]
                    y_id = xy2id[triple["y"]]
                    
                    r=triple["r"][0]
                    labels.append({'r':r,'h':x_id,'t':y_id})
                pp_data['labels']=labels
                prepro_datas.append(pp_data)
            dis_path = "../c+_conv" + ["/train.json", "/dev.json", "/test.json"][sid]
            with open(dis_path, 'w') as f:
                json.dump(prepro_datas, f)
