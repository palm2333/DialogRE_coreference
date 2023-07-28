## 0. Package Description
```
GAIN_c+/
├─ code/
    ├── checkpoint/: save model checkpoints
    ├── logs/: save training / evaluation logs
    ├── models/:
        ├── GAIN.py: GAIN model for GloVe or BERT version
    ├── config.py: process command arguments
    ├── convert_data_coref.py: convert the format of DialogRE_c+ to the required format for GAIN 
    ├── data.py: define Datasets / Dataloader for GAIN-GloVe or GAIN-BERT
    ├── test.py: evaluation code
    ├── train.py: training code
    ├── utils.py: some tools for training / evaluation
    ├── *.sh: training / evaluation shell scripts
├─ data/: data in the required format for GAIN and preprocessed data
    ├── prepro_data/
    ├── README.md
├─ PLM/: save pre-trained language models such as BERT_base / BERT_lagrge
    ├── bert-base-uncased/
    ├── bert-large-uncased/
├─ README.md
```

## 1. Environments

- python         (3.7.4)
- cuda           (10.2)
- Ubuntu-18.0.4  (4.15.0-65-generic)

## 2. Dependencies

- numpy          (1.19.2)
- matplotlib     (3.3.2)
- torch          (1.6.0)
- transformers   (3.1.0)
- dgl-cu102      (0.4.3)
- scikit-learn   (0.23.2)

PS: dgl >= 0.5 is not compatible with our code, we will fix this compatibility problem in the future.

## 3. Preparation

### 3.1. Dataset
- Obtain our `DialogRE_c+` dataset 

- Put `word2id.json`, `ner2id.json`, `rel2id.json`, `vec.npy` into the directory `data/`

- Run the `convert_data_coref.py` to convert `DialogRE_c+` to the format required by GAIN and place the converted data in `data/`

### 3.2. (Optional) Pre-trained Language Models
Following the hint in this [link](http://viewsetting.xyz/2019/10/17/pytorch_transformers/?nsukey=v0sWRSl5BbNLDI3eWyUvd1HlPVJiEOiV%2Fk8adAy5VryF9JNLUt1TidZkzaDANBUG6yb6ZGywa9Qa7qiP3KssXrGXeNC1S21IyT6HZq6%2BZ71K1ADF1jKBTGkgRHaarcXIA5%2B1cUq%2BdM%2FhoJVzgDoM7lcmJg9%2Be6NarwsZzpwAbAwjHTLv5b2uQzsSrYwJEdPl7q9O70SmzCJ1VF511vwxKA%3D%3D), download possible required files (`pytorch_model.bin`, `config.json`, `vocab.txt`, etc.) into the directory `PLM/bert-????-uncased` such as `PLM/bert-base-uncased`.

## 4. Training

```bash
>> cd code
>> ./runXXX.sh gpu_id   # like ./run_GAIN_BERT.sh 2
>> tail -f -n 2000 logs/train_xxx.log
```

## 5. Evaluation

```bash
>> cd code
>> ./evalXXX.sh gpu_id threshold(optional)  # like ./eval_GAIN_BERT.sh 0 0.5521
>> tail -f -n 2000 logs/test_xxx.log
```

PS: we recommend to use threshold = -1 (which is the default, you can omit this arguments at this time) for dev set, 
the log will print the optimal threshold in dev set, and you can use this optimal value as threshold to evaluate test set.