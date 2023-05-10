import os
import glob

from tqdm import tqdm
import PyPDF2

IN_DIR = 'pdf_policies'
OUT_DIR = 'text_pdf_policies'
LOG_DIR = "logs"


def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    pdf_files = glob.glob(os.path.join(IN_DIR, '*.pdf'))
    assert len(pdf_files) != 0
    
    failed_fns = []
    for fn in tqdm(pdf_files, total=len(pdf_files)):
        try:
            with open(fn, 'rb') as pdfFileObj:
                pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
                # print(" No. Of Pages :", pdfReader.numPages)

                policy_text_str = ''
                for page_idx in range(pdfReader.numPages):
                    pageObj = pdfReader.getPage(page_idx)
                    # print(f"In page {page_idx}{pageObj.extractText()}")
                    page_text = pageObj.extractText()
                    policy_text_str += page_text

            # save to output
            fn_basename = os.path.basename(fn)
            fn_basename = os. path. splitext(fn_basename)[0] + ".txt"
            output_fn = os.path.join(OUT_DIR, fn_basename)

            with open(output_fn, 'w') as fw:
                fw.write(policy_text_str)
        except:
            failed_fns.append(os.path.basename(fn))

    # log failed examples
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    with open(os.path.join(LOG_DIR, "failed_parsed_pdf2txt_policies.txt"), "w") as fw:
        for fn in failed_fns:
            fw.write(fn + '\n')


if __name__ == '__main__':
    main()