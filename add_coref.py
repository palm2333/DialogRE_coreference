import json
import os

def add_coref():
    sets = ["dev.json", "train.json", "test.json"]
    DialogRE_path = "./data/DialogRE/"
    coref_path="./data/coref_data/"
    cplus_path="./data/DialogRE_c+/"
    if not os.path.exists(cplus_path):
        os.makedirs(cplus_path)

    for set in sets:

        Dia_file = DialogRE_path + set
        with open(Dia_file, 'r', encoding="utf-8") as f:
            Dia_data = json.load(f)

        coref_file = coref_path + set + "l"
        with open(coref_file, 'r', encoding="utf-8") as f:
            coref_data=[json.loads(line) for line in f]

        assert len(coref_data)==len(Dia_data)
        for i in range(len(coref_data)):
            assert len(coref_data[i])==len(Dia_data[i][1])
            for j in range(len(coref_data[i])):
                Dia_data[i][1][j]['x_mention']=coref_data[i][j]['x_mention']
                Dia_data[i][1][j]['y_mention'] = coref_data[i][j]['y_mention']

        cplus_file=cplus_path+set
        with open(cplus_file, 'w') as f:
            json.dump(Dia_data,f, indent=4)


if __name__=="__main__":
    add_coref()