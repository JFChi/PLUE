import os
import glob
import re
import sys

from bs4 import BeautifulSoup
from markdown import markdown

from tqdm import tqdm

IN_DIR = sys.argv[1]
OUT_DIR = 'text_policies'
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

def remove_first_line(string):
    """ Remove the first line of the string """
    assert string.startswith(">")
    ind1 = string.find('\n')
    return string[ind1+1:]

def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    md_files = list(glob.glob(f'{IN_DIR}/**/*.md', recursive=True))
    assert len(md_files) >= 1
    readme_fn = md_files.pop(0) # remove the readme file
    assert readme_fn.endswith("README.md")

    failed_fns = []     
    for fn in tqdm(md_files, total=len(md_files)):

        with open(fn, 'r', errors="ignore") as fr:
            md_str = fr.read()
        
        # remove the first line of md file
        md_str = remove_first_line(md_str)

        try:
            # extract plain text
            plain_text = markdown_to_text(md_str)

            # save to output
            fn_basename = os.path.basename(fn)
            fn_basename = os.path.splitext(fn_basename)[0] + ".txt"
            output_fn = os.path.join(OUT_DIR, fn_basename)

            with open(output_fn, 'w') as fw:
                fw.write(plain_text)
            
        except:
            failed_fns.append(os.path.basename(fn))

    # log failed examples
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    with open(os.path.join(LOG_DIR, "failed_parsed_md2txt_policies.txt"), "w") as fw:
        for fn in failed_fns:
            fw.write(fn + '\n')

if __name__ == '__main__':
    main()