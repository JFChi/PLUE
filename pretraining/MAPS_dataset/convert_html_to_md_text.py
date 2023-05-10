import os
import glob
import html2text

from tqdm import tqdm

IN_DIR = 'html_policies'
OUT_DIR = 'md_html_policies'
LOG_DIR = "logs"

def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    html_files = glob.glob(os.path.join(IN_DIR, '*.html'))
    assert len(html_files) != 0

    failed_fns = []     
    for fn in tqdm(html_files, total=len(html_files)):
        with open(fn, 'r') as fr:
            html_str = fr.read()

        try:
            h = html2text.HTML2Text()
            # set configuration here
            h.ignore_links = True
            h.ignore_emphasis = True
            
            plain_text = h.handle(html_str)
            
            # save to output
            fn_basename = os.path.basename(fn)
            fn_basename = os.path.splitext(fn_basename)[0] + ".md"
            output_fn = os.path.join(OUT_DIR, fn_basename)

            # print(output_fn)
            with open(output_fn, 'w') as fw:
                fw.write(plain_text)
        except:
            failed_fns.append(os.path.basename(fn))

    # log failed examples
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    with open(os.path.join(LOG_DIR, "failed_parsed_html2md_policies.txt"), "w") as fw:
        for fn in failed_fns:
            fw.write(fn + '\n')

if __name__ == '__main__':
    main()