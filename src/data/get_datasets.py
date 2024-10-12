import os.path
from datasets import load_dataset

def get_sst2(splits=["train", "validation"]):
    # load SST2 and store as text file
    dataset = load_dataset("stanfordnlp/sst2")
    labels_map = {0: "negative", 1: "positive"}
    ss2_data = []
    for s in splits:
        for idx in range(len(dataset[s])):
            sentence = dataset[s][idx]["sentence"].strip()
            label = labels_map[dataset[s][idx]["label"]]
            ss2_data.append(f"{sentence},{label}\n")
    # remove newline from last sample
    ss2_data[-1] = ss2_data[-1].rstrip()

    with open('Data/sst2-data.txt', 'w') as fp:
        for sample in ss2_data:
            # write to file
            fp.write(sample)

def get_imdb(splits=["train", "test"]):
    # load IMDB and store as text file
    dataset = load_dataset("stanfordnlp/imdb")
    labels_map = {0: "negative", 1: "positive"}
    imdb_data = []
    for s in splits:
        for idx in range(len(dataset[s])):
            text = dataset[s][idx]["text"].strip()
            label = labels_map[dataset[s][idx]["label"]]
            imdb_data.append(f"{text},{label}\n")
    # remove newline from last sample
    imdb_data[-1] = imdb_data[-1].rstrip()

    with open('Data/imdb-data.txt', 'w') as fp:
        for sample in imdb_data:
            # write to file
            fp.write(sample)

def get_clinc150(splits=["train", "valid", "test"]):
    data_dir = "Data/CLINC-OOD"
    clinc150_data = []
    for s in splits:
        with open(f'{data_dir}/{s}.seq.in', 'r', encoding="utf-8") as fp_seq, open(f'{data_dir}/{s}.label', 'r', encoding="utf-8") as fp_label:
            for text, label in zip(fp_seq, fp_label):
                clinc150_data.append(f'{text.strip()},{label.strip()}\n')
    # remove newline from last sample
    clinc150_data[-1] = clinc150_data[-1].rstrip()

    with open('Data/clinc-data.txt', 'w') as fp:
        for sample in clinc150_data:
            # write to file
            fp.write(sample)

def get_rostd(splits=["train", "validation", "test"]):
    # TODO
    pass

def get_banking77_oos(ind_splits=["train", "valid", "test"]):
    data_dir = "Data/BANKING77-OOS"
    banking77_oos_data = []
    # Get IND train/val/test samples
    for s in ind_splits:
        with open(f'{data_dir}/{s}/seq.in', 'r', encoding="utf-8") as fp_seq, open(f'{data_dir}/{s}/label', 'r', encoding="utf-8") as fp_label:
            for text, label in zip(fp_seq, fp_label):
                banking77_oos_data.append(f'{text.strip()},{label.strip()}\n')
    
    # Get IND-OOS and OOD-OOS
    ood_splits = ['id-oos', 'ood-oos']
    for s in ood_splits:
        sub_dirs = os.listdir(f'{data_dir}/{s}/')
        for sd in sub_dirs:
            with open(f'{data_dir}/{s}/{sd}/seq.in', 'r', encoding="utf-8") as fp_seq, open(f'{data_dir}/{s}/{sd}/label', 'r', encoding="utf-8") as fp_label:
                for text, label in zip(fp_seq, fp_label):
                    banking77_oos_data.append(f'{text.strip()},{label.strip()}\n')
    # remove newline from last sample
    banking77_oos_data[-1] = banking77_oos_data[-1].rstrip()

    with open('Data/banking77-oos-data.txt', 'w') as fp:
        for sample in banking77_oos_data:
            # write to file
            fp.write(sample)


def main():
    if not os.path.isfile('Data/sst2-data.txt'):
        print("Loading SST2 dataset to text file...")
        get_sst2()
    if not os.path.isfile('Data/imdb-data.txt'):
        print("Loading IMDB dataset to text file...")
        get_imdb()
    if not os.path.isfile('Data/clinc-data.txt'):
        print("Loading CLINC-150 dataset to text file...")
        get_clinc150()
    if not os.path.isfile('Data/rostd-data.txt'):
        print("Loading ROSTD dataset to text file...")
        get_rostd()
    if not os.path.isfile('Data/banking77-oos-data.txt'):
        print("Loading BANKING77-OOS dataset to text file...")
        get_banking77_oos()


if __name__ == '__main__':
    main()