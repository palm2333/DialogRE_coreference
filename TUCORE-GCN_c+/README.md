## Environments

- python		(3.8.3)
- cuda			(11.0)
- Ubuntu-18.04.5 

## Requirements

- dgl-cu110			   (0.5.3)
- torch					   (1.7.0)
- numpy					(1.19.2)
- sklearn
- regex
- packaging
- tqdm


## Usage

- run_classifier.py : Code to train and evaluate the model
- data.py : Code to define Datasets / Dataloader
- evaluate.py : Code to evaluate the model on DialogRE_c+
- models/BERT : The directory containing the GeGCN for BERT version

## Preparation

### Dataset

- put `train.json`, `dev.json`, `test.json` from ```DialogRE_c+``` into the directory `dataset`


### Pre-trained Language Models

- Download and unzip BERT-Base Uncased from [here](https://github.com/google-research/bert), and copy the files into the directory `pre-trained_model/BERT/`
- Set up the environment variable for BERT by ```export BERT_BASE_DIR=/PATH/TO/BERT/DIR```. 
- In `pre-trained_model`, execute ```python convert_tf_checkpoint_to_pytorch_BERT.py --tf_checkpoint_path=$BERT_BASE_DIR/bert_model.ckpt --bert_config_file=$BERT_BASE_DIR/bert_config.json --pytorch_dump_path=$BERT_BASE_DIR/pytorch_model.bin```.

## Training & Evaluation

### BERT

- Execute the following commands:
```
python run_classifier.py --do_train --do_eval --encoder_type BERT  --data_dir dataset  --vocab_file $BERT_BASE_DIR/vocab.txt   --config_file $BERT_BASE_DIR/bert_config.json   --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 12   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir GeGCN_BERT  --gradient_accumulation_steps 2

rm GeGCN_BERT/model_best.pt

python evaluate.py --dev dataset/dev.json --test dataset/test.json --f1dev GeGCN_BERT/logits_dev.txt --f1test GeGCN_BERT/logits_test.txt --f1cdev GeGCN_BERT/logits_devc.txt --f1ctest GeGCN_BERT/logits_testc.txt --result_path GeGCN_BERT/result.txt
```