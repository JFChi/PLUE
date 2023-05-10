# Steps to crawl privacy policy history dataset

1. Download the dataset from github

```
DATA_PATH=<dataset_directory_name>
git clone https://github.com/citp/privacy-policy-historical.git $DATA_PATH
```

2. convert md document to raw text document

```
python convert_md_to_raw_text.py $DATA_PATH/privacy-policy-historical
```

3. filter policies with noise ("privacy" "policy" or "legal"  are in documents), remove empty documents, etc.

```
python filter_policies.py
```

## Output files structure

* maps_txt_policies folder contains all text privacy policies extracted.
