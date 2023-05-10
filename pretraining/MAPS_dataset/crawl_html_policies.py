import os
import csv
import json
import multiprocessing
import mmap

from tqdm import tqdm
import newspaper
from newspaper import Article


IN_DIR = 'data'
OUT_DIR = "html_policies"
LOG_DIR = "logs"
csv_fn = 'april_2018_policies.csv'

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def read_maps_csv():
    # read data from csv
    url2policyID = {}
    with open(os.path.join(IN_DIR, csv_fn)) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        next(readCSV)
        for row in tqdm(readCSV, total=get_num_lines(file_path=os.path.join(IN_DIR, csv_fn))-1):
            # print(row)
            _id = row[0]
            package_id = row[1]
            document_type = row[2]
            # skip pdf privacy policies
            if document_type != 'HTML':
                continue
            source = row[4]
            source = eval(source)
            final_url = source[0]["Final URL"]
            # print(_id, package_id, final_url)
            
            url2policyID[final_url] = f"{_id}_{package_id}"

    return url2policyID

def get_url_content(url_str):
    # scrap
    article = Article(url_str, keep_article_html=True, language='en')
    try:
        article.download()
        article.parse()
    except newspaper.article.ArticleException:
        return
    return article

def save_html_file(html_str, fn):
    with open(os.path.join(OUT_DIR, fn), 'w') as fw:
        fw.write(html_str)

def crawl_and_save(url_str, policy_id):
    try:
        article = get_url_content(url_str)
    except: # other error except that newspaper.article.ArticleException
        article = None
    if article is None:
        return False
    else:
        html_str = article.article_html
        save_html_file(html_str=html_str, fn=f'{policy_id}.html')
        return True

def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
            
    url2policyID = read_maps_csv()
    print(f"num of unique urls for html policies: {len(url2policyID)}")
    # url2policyID = dict(list(url2policyID.items())[:5]) # TODO delete: sample 5 items from url2policyID for fast testing
    
    # parallel downloading html files
    num_worker = 18
    url2policyID = list(url2policyID.items())
    with multiprocessing.Pool(processes=num_worker) as pool:
        successes = pool.starmap(crawl_and_save, url2policyID)
    
    # write failed parsed policies info to log
    failed_parsed_policies = [fail_item for success_bool, fail_item in zip(successes, url2policyID) if success_bool == False]
    print(f"Number of failed downloaded privacy policies {len(failed_parsed_policies)}")
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    with open(os.path.join(LOG_DIR, "failed_downloaded_html_policies.txt"), "w") as fw:
        for fpp in failed_parsed_policies:
            fw.write('\t'.join(str(s) for s in fpp) + '\n')
    

    '''
    # sequential downloading html files
    num_parse_err = 0
    for url_str, policy_id in tqdm(url2policyID.items(), total=len(url2policyID)):
        article = get_url_content(url_str)
        print("url_str", url_str)
        if article is None:
            num_parse_err += 1
            continue
        html_str = article.article_html
        with open(os.path.join(OUT_DIR, f"{policy_id}.html"), 'w') as fw:
            fw.write(html_str)
    '''
    
if __name__ == '__main__':
    main()