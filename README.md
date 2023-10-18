# DialogRE^C+: An Extension of DialogRE to Investigate How Much Coreference Helps Relation Extraction in Dialogs

This subject contains the DialogRE^C+ dataset, script, and the benchmark models.

ArXiv: https://arxiv.org/abs/2308.04498

## script

add_coref.py: is used to add coreference information to the DialogRE dataset.  
1. download the **DialogRE** dataset and **coref_data**, decompress;    
2. place the data_v2/en/data/{dev/test/train}.json in data/DialogRE/;  
3. run add_coref.py.

For more details of the models, please refer to the README.md file in the  lower level directory.
