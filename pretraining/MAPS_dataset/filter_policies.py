import os
import glob

from tqdm import tqdm

IN_HTML_DIR = 'text_html_policies'
IN_PDF_DIR = 'text_pdf_policies'
OUT_DIR = 'maps_txt_policies'
LOG_DIR = "logs"

def keep_criteria(text_str):
    text_str = text_str.lower()
    if 'privacy' in text_str \
        or 'policy' in text_str \
        or 'legal' in text_str:
        return True
    else:
        return False

def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    html_text_files = glob.glob(os.path.join(IN_HTML_DIR, '*.txt'))
    assert len(html_text_files) != 0

    pdf_text_files = glob.glob(os.path.join(IN_PDF_DIR, '*.txt'))
    assert len(pdf_text_files) != 0

    filtered_text_fns = []
    # filter html text 
    for fn in tqdm(html_text_files, total=len(html_text_files)):
        with open(fn, 'r') as fr:
            text_str = fr.read()
        
        if keep_criteria(text_str):
            fn_basename = os.path.basename(fn)
            output_fn = os.path.join(OUT_DIR, fn_basename)
            with open(output_fn, 'w') as fw:
                fw.write(text_str)
        else:
            filtered_text_fns.append(os.path.basename(fn))

    # filter pdf text 
    for fn in tqdm(pdf_text_files, total=len(pdf_text_files)):
        with open(fn, 'r') as fr:
            text_str = fr.read()
        
        if keep_criteria(text_str):
            fn_basename = os.path.basename(fn)
            output_fn = os.path.join(OUT_DIR, fn_basename)
            with open(output_fn, 'w') as fw:
                fw.write(text_str)
        else:
            filtered_text_fns.append(os.path.basename(fn))

    # log failed examples
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    with open(os.path.join(LOG_DIR, "filtered_txt_policies.txt"), "w") as fw:
        for fn in filtered_text_fns:
            fw.write(fn + '\n')


if __name__ == '__main__':
    main()


