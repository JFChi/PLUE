# Steps to crawl MAPS privacy policies

1. Download maps data and unzip it

```
bash download_maps_dataset.sh
``` 

2. crawl/download html policies
    * Different apps might contain the a list of same privacy policy urls, so we need to filter them out;
    * One app might span multiple rows, we need to distinguish them, so we need to use urls first, and use "line_id +
      app_id + time"  as the final file names

```
python crawl_html_policies.py
```

which results in

* Number of unique urls for html policies: 144525
* Number of failed downloaded privacy policies: 43337 (the corresponding urls and apps id are shown in logs)

3. crawl/download pdf policies
    * Different apps might contain the a list of same privacy policy urls, so we need to filter them out;
    * One app might span multiple rows, we need to distinguish them, so we need to use urls first, and use "line_id +
      app_id + time"  as the final file names

```
python crawl_pdf_policies.py
```

which results in

* Number of unique urls for pdf policies: 3675
* Number of failed downloaded privacy policies: 1719 (the corresponding urls and apps id are shown in logs)

4. convert html policies & pdf policies to txt file

```
python convert_pdf_to_text.py
python convert_html_to_md_text.py && python convert_md_to_raw_text.py
```

which results in

* 101187 text documents with html source left
* 1772 text documents with pdf source left

5. filter policies with noise ("privacy" "policy" or "legal"  are in documents), remove empty documents, etc.

```
python filter_policies.py
```

## Output files structure

* privacy_policy_history_txt_policies folder contains all text privacy policies extracted. It contains 64717 documents.