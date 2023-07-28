# Requirement
```
python==3.6.10 
torch==1.5.1
tqdm==4.29.1
numpy==1.15.4
spacy==2.1.0
transformers==4.2.2
```

# Dataset
- Obtain our `DialogRE_c+` dataset 
- Put `word2id.json`, `ner2id.json`, `rel2id.json`, `char2id.json` into the data directory

# Data Proprocessing
If you want to use manually annotated coreferences in `DialogRE_c+`, run:

```
# cd code
# python3 gen_data.py --in_path ./DialogRE_c+  --out_path  prepro_data_c+  --use_corefchain True
```

If you want to remove the first or second person reference and don't need to use the manually annotated coreferences, run: 

```
# cd code
# python3 gen_data_noanycoref.py --in_path ./DialogRE_c+  --out_path  prepro_data_noanycoref
```

# Training
In order to train the model, run:

```
# cd code
# CUDA_VISIBLE_DEVICES=0 python3 train.py --save_name dialogre --use_spemb True --use_wratt True --use_gcn True
```

# Test
After the training process, we can test the model by:

```
 CUDA_VISIBLE_DEVICES=0 python3 test.py --save_name dialogre --use_spemb True --use_wratt True --use_gcn True
```