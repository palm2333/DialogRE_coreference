## Setup
Download GloVe vectors from [here](https://www.kaggle.com/thanakomsn/glove6b300dtxt/data) and put it into `dataset/` folder

Next Install the required libraries:
1. Assume you have installed Pytorch >= 1.5
2. Install dgl library according to your cuda version using the commands below
```sh
pip install --pre dgl-cu100     # For CUDA 10.0 Build
pip install --pre dgl-cu101     # For CUDA 10.1 Build
pip install --pre dgl-cu102     # For CUDA 10.2 Build
```
3. Install PytorchLightning [github](https://github.com/PyTorchLightning/pytorch-lightning)
4. Install from requirements.txt by `pip install -r requirements.txt` and run `python -m spacy download en_core_web_sm`
5. Gain `train.json`, `dev.json`, `test.json` from ```DialogRE_c+```


## Run code

### Training
```sh
python main.py --data_dir DialogRE_c+/
```

### Testing
```sh
python main.py --data_dir DialogRE_c+/ --mode test --ckpt_path [your_ckpt_file_path]
```