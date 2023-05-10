import os
import json
import argparse
import yaml
import csv
from tqdm import tqdm
from collections import defaultdict
import codecs
from bs4 import BeautifulSoup


def piextract(dirpath):
    BIO_tags = {
        'CollectUse': {
            'true': ["O", "B-COLLECT", "I-COLLECT"],
            'false': ["O", "B-NOT_COLLECT", "I-NOT_COLLECT"]
        },
        'Share': {
            'true': ["O", "B-SHARE", "I-SHARE"],
            'false': ["O", "B-NOT_SHARE", "I-NOT_SHARE"]
        }
    }

    def process_split(split):
        for practice, value in BIO_tags.items():
            examples = {}
            exceptions = {}
            for v, tags in value.items():
                line_no = 0
                with open(os.path.join(dirpath, f"{practice}_{v}", f"{split}.conll03"), "r") as f:
                    tokens, labels = [], []
                    for line in f.readlines():
                        sp = line.strip().split()
                        if len(sp) == 4:
                            if sp[0] == '-DOCSTART-':
                                # first line in the file
                                continue
                            assert sp[3] in tags
                            tokens.append(sp[0])
                            labels.append(sp[3])
                        elif len(tokens) > 0:
                            if line_no in examples:
                                assert examples[line_no]['tokens'] == tokens
                                ex = examples[line_no]
                                inconsistent_label = False
                                for idx, l in enumerate(labels):
                                    if ex['labels'][idx] != 'O' and l != 'O':
                                        inconsistent_label = True
                                    if l != 'O':
                                        ex['labels'][idx] == l
                                if inconsistent_label:
                                    if line_no not in exceptions:
                                        exceptions[line_no] = [ex['tokens']]
                                    for w, v1, v2 in zip(ex['tokens'], ex['labels'], labels):
                                        if v1 != 'O' and v2 != 'O' and v1 != v2:
                                            exceptions[line_no].append((w, v1, v2))
                            else:
                                examples[line_no] = {'tokens': tokens, 'labels': labels}
                            tokens, labels = [], []
                            line_no += 1

            print("[Kept] Examples with consistent labels: ", len(examples))
            print("[Discarded] Examples with inconsistent labels: ", len(exceptions))
            with open(os.path.join(dirpath, f"{practice}_{split}.json"), "w") as f:
                for _, ex in examples.items():
                    f.write(json.dumps(ex) + "\n")

    process_split('train')
    # NOTE: validation and test datasets are exactly the same
    # process_split('validation')
    process_split('test')


def app350(dirpath, only_data_type=True):
    annotation_path = os.path.join(dirpath, "APP-350_v1.1", "annotations")
    annotation_filenames = sorted([os.path.join(annotation_path, fn) for fn in os.listdir(annotation_path)])

    label_list = set()

    def process_file(fn):
        with open(fn, 'r') as fr:
            raw_data_dict = yaml.safe_load(fr)

        segments = raw_data_dict['segments']
        examples = []

        no_data_party_practice = ["Facebook_SSO", "SSO"]
        for segment in segments:
            segment_text = segment['segment_text']
            if only_data_type:
                label = [ann['practice'] if ann['practice'] in no_data_party_practice else "_".join(
                    ann['practice'].split("_")[:-1]) for ann in segment['annotations']]
            else:
                label = [ann['practice'] for ann in segment['annotations']]
            label_list.update(label)
            if not label:
                label.append("No_Mentioned")
            examples.append(
                {
                    'text': segment_text,
                    'label': label,
                }
            )
        return examples

    data = []
    for idx, fn in enumerate(tqdm(annotation_filenames, total=len(annotation_filenames))):
        policy_data = process_file(fn)
        data.extend(policy_data)

        if idx + 1 == 250:
            train_fn = os.path.join(dirpath, 'train.json')
            with open(train_fn, 'w') as fw:
                for d in data:
                    fw.write(json.dumps(d) + "\n")
            print(f"Saving APP-350 training data to {train_fn}")
            data = []
        elif idx + 1 == 300:
            valid_fn = os.path.join(dirpath, 'valid.json')
            with open(valid_fn, 'w') as fw:
                for d in data:
                    fw.write(json.dumps(d) + "\n")
            print(f"Saving APP-350 training data to {valid_fn}")
            data = []
        elif idx + 1 == 350:
            test_fn = os.path.join(dirpath, 'test.json')
            with open(test_fn, 'w') as fw:
                for d in data:
                    fw.write(json.dumps(d) + "\n")
            print(f"Saving APP-350 training data to {test_fn}")
            data = []

    # save label_list
    label_list = sorted(label_list)
    label_list.append("No_Mentioned")
    label_list_fn = os.path.join(dirpath, 'label_list.txt')
    with open(label_list_fn, 'w') as fw:
        fw.write('\n'.join(label for label in label_list))


def opp115(dirpath):
    src_dir = os.path.join(dirpath, "OPP-115")

    def get_company_names(src_dir):
        filenames = os.listdir(os.path.join(src_dir, "consolidation", "threshold-1.0-overlap-similarity"))
        com_names = []
        for f in filenames:
            com_names.append(os.path.splitext(f)[0])
        return com_names

    company_names = get_company_names(src_dir)

    def read_policy(filename):
        filepath = os.path.join(src_dir, "sanitized_policies/", filename)
        f = codecs.open(filepath, 'r', 'utf-8')
        document = BeautifulSoup(f.read(), features="lxml").get_text()
        segments = document.split('|||')
        segments = [s.strip() for s in segments]
        return segments

    def read_annotations(com_name, segments):
        filename = "%s.csv" % com_name
        cname = ('_'.join(com_name.split('_')[1:]))
        filepath = os.path.join(src_dir, "consolidation/threshold-1.0-overlap-similarity/", filename)
        segid2labels = defaultdict(list)
        with open(filepath, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                segment_id = int(row[4])
                category_name = row[5]
                if category_name in segid2labels[segment_id]:
                    pass
                else:
                    segid2labels[segment_id].append(category_name)

        data = []
        for seg_id, labels in segid2labels.items():
            data_item = {
                'text': segments[seg_id],
                'label': labels,
            }
            data.append(data_item)

        return data

    data = []
    for idx, cname in enumerate(tqdm(company_names, total=len(company_names))):
        segments = read_policy("%s.html" % cname)
        policy_data = read_annotations(cname, segments)
        data.extend(policy_data)
        if idx + 1 == 85:
            train_fn = os.path.join(dirpath, 'train.json')
            with open(train_fn, 'w') as fw:
                for d in data:
                    fw.write(json.dumps(d) + "\n")
            data = []
        elif idx + 1 == 95:
            valid_fn = os.path.join(dirpath, 'valid.json')
            with open(valid_fn, 'w') as fw:
                for d in data:
                    fw.write(json.dumps(d) + "\n")
            data = []
        elif idx + 1 == 115:
            test_fn = os.path.join(dirpath, 'test.json')
            with open(test_fn, 'w') as fw:
                for d in data:
                    fw.write(json.dumps(d) + "\n")
            data = []

    # save label list
    label_list = [
        'First Party Collection/Use',
        'Third Party Sharing/Collection',
        'User Choice/Control',
        'User Access, Edit and Deletion',
        'Data Retention',
        'Data Security',
        'Policy Change',
        'Do Not Track',
        'International and Specific Audiences',
        'Other',
    ]
    label_list_fn = os.path.join(dirpath, 'label_list.txt')
    with open(label_list_fn, 'w') as fw:
        fw.write('\n'.join(label for label in label_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, help='dataset name', choices=['piextract', 'app350', 'opp115']
    )
    parser.add_argument(
        "--source_dir", type=str, help='source data directory path'
    )
    args = parser.parse_args()
    if args.dataset == 'piextract':
        piextract(args.source_dir)
    elif args.dataset == 'app350':
        app350(args.source_dir)
    elif args.dataset == 'opp115':
        opp115(args.source_dir)
    else:
        raise NotImplementedError
