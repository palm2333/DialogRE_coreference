import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from models.encoder import Encoder
from models.attention import SelfAttention, MultiHeadedAttention
from models.gcn import GraphConvLayer,MultiGraphConvLayer
from pytorch_transformers import *


path = "./modeling_bert"

class DIALOGRE(nn.Module):
    def __init__(self, config):
        super(DIALOGRE, self).__init__()
        self.config = config
        modelConfig = BertConfig.from_pretrained(path + "/" + "bert_config.json")
        self.bert1 = BertModel.from_pretrained(
            path + "/" + 'pytorch_model.bin', config=modelConfig)

        hidden_size = config.rnn_hidden
        bert_hidden_size = 768
        speaker_hidden_size = 16
        if self.config.use_spemb:
            self.speaker_emb = nn.Embedding(10, speaker_hidden_size)
            self.rel_emb = nn.Embedding(config.relation_num-1, bert_hidden_size + speaker_hidden_size)
            self.linear_bert_re = nn.Linear(bert_hidden_size + speaker_hidden_size, hidden_size)
            self.linear_context = nn.Linear(bert_hidden_size + speaker_hidden_size, hidden_size)
            self.multi_att = MultiHeadedAttention(16, bert_hidden_size + speaker_hidden_size)
        else:
            self.linear_bert_re = nn.Linear(bert_hidden_size, hidden_size)
            self.linear_context = nn.Linear(bert_hidden_size, hidden_size)
            self.rel_emb = nn.Embedding(config.relation_num-1, bert_hidden_size)
            self.multi_att = MultiHeadedAttention(16, bert_hidden_size)

        self.self_att = SelfAttention(hidden_size)

        self.bili = torch.nn.Bilinear(hidden_size,  hidden_size, hidden_size)

        self.linear_output = nn.Linear(2 * hidden_size, config.relation_num-1)

        self.relu = nn.ReLU()
        self.dropout_rate = nn.Dropout(config.dropout_rate)
        self.hidden_size = hidden_size

        self.dropout_gcn = nn.Dropout(config.dropout_gcn)
        if config.use_gcn:
            self.gcn_head = 16
            self.gcn_layer = MultiGraphConvLayer(hidden_size, 1, self.gcn_head, self.dropout_gcn)

    def forward(self, context_idxs, h_mapping, t_mapping,
                relation_mask, mention_node_position, entity_position,
                mention_node_sent_num, entity_num_list, sdp_pos, sdp_num_list,
                context_masks, context_starts, attention_label_mask, speaker_label):
        """
        均针对一个三元组, context_xx下标针对piece token，其余均为token，若有长度等于context_xx仅为pad
        context_idxs: doc Token IDs, batch*real_len,token被分割
        h_mapping: Head，根据x共指的数量、长度，为x共指位置分配的初始权重，batch*1*real_len
        t_mapping: Tail，根据y共指的数量、长度，为y共指位置分配的初始权重
        relation_mask: There are multiple relations for each instance so we need a mask in a batch
        mention_node_position: batch*node_num*real_len，第i个结点在i-1行对应位置=1
        entity_position:batch*2*real_len，x‘mention在第一行对应位置=1，y第二行。mention_node_position合并表示
        mention_node_sent_num:  number of mention nodes in each sentences of a document,batch*sen_num，每行mention结点数
        entity_num_list: the number of entity nodes in each document，batch,xy数量(2)
        sdp_pos: MDP node position,batch*sdp_num*real_len, 对于第i个sdp结点，第i-1行对应位置=1
        sdp_num_list: the number of MDP node in each document，batch，sdp结点数量
        context_masks:batch*real_len, pad=0，其余=1,token被分割
        context_starts:batch*real_len,原token开始的位置=1，其余=0，（因为token被分割）
        attention_label_mask:batch*token_len*36,原token对应36维=1，pad=0
        speaker_label:batch*real_len, 对应位置的说话者id，from 1
        """
        # batch*real_len*encode_size，piece token encode
        context_output1 = self.bert1(context_idxs, attention_mask=context_masks)[0]
        # 取出原token开头对应表示
        context_output = [layer[starts.nonzero().squeeze(1)]for layer, starts in zip(context_output1, context_starts)]
        del context_output1
        # context_output: batch*token_len*encode_size
        context_output = pad_sequence(context_output, batch_first=True, padding_value=-1)
        max_doc_len = context_output.shape[1]

        # rel_embedding: batch*36*rel_size
        rel_embedding = self.rel_emb(torch.tensor([i for i in range(36)]).cuda())
        rel_embedding = rel_embedding.unsqueeze(0).expand(context_output.shape[0],-1,-1)

        # batch*token_len，token对应speaker id
        speaker_label = speaker_label[:, :max_doc_len]
        if self.config.use_spemb:
            # 拼接speaker embedding至embedding
            speaker_emb = self.speaker_emb(speaker_label)
            context_output = torch.cat([context_output, speaker_emb], dim=-1)
        if self.config.use_wratt:
            # 输入顺序：query, key, value,输出attention后value、attention score
            # h_t_query:batch*token_len*hidden_size, attention(bert编码+speaker_emb,rel_emb)
            # attn: batch*head(16)*hidden_size*36
            h_t_query, attn = self.multi_att(context_output, rel_embedding, rel_embedding, mask=attention_label_mask)
            # lsr_input：rel_embedding和embedding attention并linear后
            lsr_input = self.linear_bert_re(h_t_query)
            # context_output:batch*token_len*hidden_size，bert编码+speaker_emb+linear
            context_output = self.linear_context(context_output)
        else:
            context_output = self.linear_context(context_output)
            lsr_input = context_output
            attn = torch.zeros(context_output.shape[0],16,context_output.shape[1],36)

        if self.config.use_gcn:
            '''extract Mention node representations'''
            # batch，每个三元组mention数量
            mention_num_list = torch.sum(mention_node_sent_num, dim=1).long().tolist()
            max_mention_num = max(mention_num_list)
            # batch*mention_num*hidden_size, mention结点表示
            mentions_rep = torch.bmm(mention_node_position[:, :max_mention_num, :max_doc_len], lsr_input)

            '''extract MDP(meta dependency paths) node representations'''
            # sdp结点数量
            sdp_num_list = sdp_num_list.long().tolist()
            max_sdp_num = max(sdp_num_list)
            # batch*sdp_num*hidden_size, sdp结点表示
            sdp_rep = torch.bmm(sdp_pos[:,:max_sdp_num, :max_doc_len], lsr_input)

            '''extract Entity node representations'''
            # batch*entity_num(2)*hidden_size，entity结点表示，mention结点合集
            entity_rep = torch.bmm(entity_position[:,:,:max_doc_len], lsr_input)

            '''concatenate all nodes of an instance'''
            # batch*all_node_num*hidden_size，所有结点表示
            gcn_inputs = []
            # batch，结点数量
            all_node_num_batch = []
            for batch_no, (m_n, e_n, s_n) in enumerate(zip(mention_num_list, entity_num_list.long().tolist(), sdp_num_list)):
                # 出去pad
                m_rep = mentions_rep[batch_no][:m_n]
                e_rep = entity_rep[batch_no][:e_n]
                s_rep = sdp_rep[batch_no][:s_n]
                gcn_inputs.append(torch.cat((m_rep, e_rep, s_rep),dim=0))
                node_num = m_n + e_n + s_n
                all_node_num_batch.append(node_num)

            # batch*node_num*hidden_size, pad(all_node)
            gcn_inputs = pad_sequence(gcn_inputs).permute(1, 0, 2)
            output = gcn_inputs

            # 一个doc所有结点全连接，除pad
            adj_matrix = torch.zeros(output.shape[0], output.shape[1], output.shape[1], self.gcn_head).cuda()
            for ni, node_num in enumerate(all_node_num_batch):
                adj_matrix[ni, :node_num, :node_num, :] = 1
            # 卷积，batch*node_num*hidden_size
            output = self.gcn_layer(adj_matrix, output)

            mention_node_position = mention_node_position.permute(0, 2, 1)
            #batch*token_len *hidden_size, mention结点对应位置=gcn后mention表示，其余=0
            output = torch.bmm(mention_node_position[:, :max_doc_len, :max_mention_num], output[:, :max_mention_num])
            # bert+linear 与 gcn(attention(bert+linear,label_emb))相加
            context_output = torch.add(context_output, output)

        # batch*1*hidden_size，(x_mention_node*权重）相加
        start_re_output = torch.matmul(h_mapping[:, :, :max_doc_len], context_output) # aggregation
        # batch*1*hidden_size，(y_mention_node*权重）相加
        end_re_output = torch.matmul(t_mapping[:, :, :max_doc_len], context_output) # aggregation
        # batch*1*2hidden_size，xy表示融合
        re_rep = self.dropout_rate(self.relu(self.bili(start_re_output, end_re_output)))

        re_rep = self.self_att(re_rep, re_rep, relation_mask)

        # return1: batch*1*36

        return self.linear_output(re_rep), attn