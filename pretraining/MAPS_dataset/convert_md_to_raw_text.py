import os
import glob
import re

from bs4 import BeautifulSoup
from markdown import markdown

from tqdm import tqdm

IN_DIR = 'md_html_policies'
OUT_DIR = 'text_html_policies'
LOG_DIR = "logs"

def markdown_to_text(markdown_string):
    """ Converts a markdown string to plaintext """

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)

    # remove code snippets
    html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
    html = re.sub(r'<code>(.*?)</code >', ' ', html)

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.findAll(text=True))

    return text

def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    md_files = glob.glob(os.path.join(IN_DIR, '*.md'))
    assert len(md_files) != 0
    
    failed_fns = []     
    for fn in tqdm(md_files, total=len(md_files)):
            
        # try:
        with open(fn, 'r') as fr:
            md_str = fr.read()
        # extract plain text
        plain_text = markdown_to_text(md_str)

        # save to output
        fn_basename = os.path.basename(fn)
        fn_basename = os.path.splitext(fn_basename)[0] + ".txt"
        output_fn = os.path.join(OUT_DIR, fn_basename)

        with open(output_fn, 'w') as fw:
            fw.write(plain_text)

        
        # except:
        #     failed_fns.append(os.path.basename(fn))

    # log failed examples
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    with open(os.path.join(LOG_DIR, "failed_parsed_md2txt_policies.txt"), "w") as fw:
        for fn in failed_fns:
            fw.write(fn + '\n')

if __name__ == '__main__':
    main()