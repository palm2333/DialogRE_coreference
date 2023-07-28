import sklearn.metrics
import torch

from config import *
from data import DGLREDataset, DGLREDataloader, BERTDGLREDataset
from models.GAIN import GAIN_GloVe, GAIN_BERT
from utils import get_cuda, logging, print_params


# for ablation
# from models.GCNRE_nomention import GAIN_GloVe, GAIN_BERT


def test(model, dataloader):
    # ours: inter-sentence F1 in LSR

    total_recall_ignore = 0

    test_result = []
    total_recall = 0
    total_steps = len(dataloader)
    for cur_i, d in enumerate(dataloader):
        print('step: {}/{}'.format(cur_i, total_steps))

        with torch.no_grad():
            labels = d['labels']
            predictions = model(words=d['context_idxs'],
                                src_lengths=d['context_word_length'],
                                mask=d['context_word_mask'],
                                entity_type_all=d['context_ner_all'],
                                entity_id_all=d['context_pos_all'],
                                mention_id_all=d['context_mention_all'],
                                distance=None,
                                entity2mention_table=d['entity2mention_table'],
                                graphs=d['graphs'],
                                h_t_pairs=d['h_t_pairs'],
                                relation_mask=None,
                                path_table=d['path_table'],
                                entity_graphs=d['entity_graphs'],
                                ht_pair_distance=d['ht_pair_distance']
                                )

            predict_re = torch.sigmoid(predictions)

        predict_re = predict_re.data.cpu().numpy()

        for i in range(len(labels)):
            label_temp = labels[i]
            label = {}
            ht_set = set()
            for l in list(label_temp.keys()):
                h, t, r = l
                if (h, t) not in ht_set:
                    ht_set.add((h, t))
                    label[(h, t)] = [r]
                else:
                    label[(h, t)].append(r)

            assert len(label) <= predict_re[i].shape[0]

            j = 0
            for ht_pair in list(label.keys()):
                rid_s = label[ht_pair]
                sort_idx = np.argsort(predict_re[i, j])
                for k in range(len(rid_s)):
                    test_result.append((sort_idx[-k - 1], rid_s[k]))
                j += 1

    correct_sys, all_sys = 0, 0
    correct_gt = 0
    for result in test_result:
        pred_y=result[0]
        true_y=result[1]
        if true_y != 36:
            correct_gt += 1
            if pred_y==true_y:
                correct_sys += 1
        if pred_y!=36:
            all_sys+=1
    precision = correct_sys / all_sys if all_sys != 0 else 1
    recall = correct_sys / correct_gt if correct_gt != 0 else 0
    f_1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    logging(
        'f1 {:3.4f} | precision {:3.4f} | recall {:3.4f}'.format(f_1,precision,recall))
    
    return f_1


if __name__ == '__main__':
    print('processId:', os.getpid())
    print('prarent processId:', os.getppid())
    opt = get_opt()
    print(json.dumps(opt.__dict__, indent=4))
    opt.data_word_vec = word2vec

    if opt.use_model == 'bert':
        # datasets
        train_set = BERTDGLREDataset(opt.train_set, opt.train_set_save, word2id, ner2id, rel2id, dataset_type='train',
                                     opt=opt)
        test_set = BERTDGLREDataset(opt.test_set, opt.test_set_save, word2id, ner2id, rel2id, dataset_type='test',
                                    instance_in_train=train_set.instance_in_train, opt=opt)

        test_loader = DGLREDataloader(test_set, batch_size=opt.test_batch_size, dataset_type='test')

        model = GAIN_BERT(opt)
    elif opt.use_model == 'bilstm':
        # datasets
        train_set = DGLREDataset(opt.train_set, opt.train_set_save, word2id, ner2id, rel2id, dataset_type='train',
                                 opt=opt)
        test_set = DGLREDataset(opt.test_set, opt.test_set_save, word2id, ner2id, rel2id, dataset_type='test',
                                instance_in_train=train_set.instance_in_train, opt=opt)

        test_loader = DGLREDataloader(test_set, batch_size=opt.test_batch_size, dataset_type='test')

        model = GAIN_GloVe(opt)
    else:
        assert 1 == 2, 'please choose a model from [bert, bilstm].'

    import gc

    del train_set
    gc.collect()

    # print(model.parameters)
    print_params(model)

    start_epoch = 1
    pretrain_model = opt.pretrain_model
    lr = opt.lr
    model_name = opt.model_name

    if pretrain_model != '':
        chkpt = torch.load(pretrain_model, map_location=torch.device('cpu'))
        model.load_state_dict(chkpt['checkpoint'])
        logging('load checkpoint from {}'.format(pretrain_model))
    else:
        assert 1 == 2, 'please provide checkpoint to evaluate.'

    model = get_cuda(model)
    model.eval()

    f1 = test(model, test_loader)
    print('finished')