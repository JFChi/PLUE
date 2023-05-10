import argparse
import os
import glob

from tqdm import tqdm
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--data_dir",
        type=str,
        default='data/pretraining_corpus',
        help="The input path of the dataset to use",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default='data/processed_pretraining_corpus',
        help="The output path of the preproecessed corpus",
    )
    parser.add_argument(
        "--test_split_percentage",
        type=float,
        default=0.05,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    args = parser.parse_args()

    assert args.data_dir is not None
    assert args.test_split_percentage > 0 and args.test_split_percentage < 1
    return args

def combine_text_corpus(text_files, out_fn, args):

    with open(os.path.join(args.out_dir, out_fn), 'w') as fw:
        for fn in tqdm(text_files, total=len(text_files)):
            with open(fn, 'r') as fr:
                text_lines = fr.readlines()
            text_str = "".join([line for line in text_lines if len(line) > 0 and not line.isspace()])
            
            fw.write(text_str.rstrip()+"\n\n")

    return

def main():
    args = parse_args()
    txt_files = glob.glob(os.path.join(args.data_dir, "**/*.txt"), recursive=True)

    assert len(txt_files) != 0

    # train/test split    
    train_txt_files, test_txt_files = train_test_split(
        txt_files,
        test_size=args.test_split_percentage,
        random_state=args.seed,
    )

    print(len(txt_files), len(train_txt_files), len(test_txt_files))
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    combine_text_corpus(train_txt_files, 'train.txt', args)
    combine_text_corpus(test_txt_files, 'test.txt', args)


if __name__ == "__main__":
    main()
