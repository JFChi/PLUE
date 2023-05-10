# borrowed from https://github.com/huggingface/datasets/blob/1.18.X/datasets/squad/squad.py

from __future__ import absolute_import, division, print_function

import json
import logging

import datasets

_CITATION = """\
@inproceedings{ahmad-etal-2020-policyqa,
    title = "{P}olicy{QA}: A Reading Comprehension Dataset for Privacy Policies",
    author = "Ahmad, Wasi  and
      Chi, Jianfeng  and
      Tian, Yuan  and
      Chang, Kai-Wei",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.66",
    pages = "743--749"
}
"""

_DESCRIPTION = """\
Privacy policy documents are long and verbose. A question answering (QA) system can assist users in finding the 
information that is relevant and important to them. Prior studies in this domain frame the QA task as retrieving the 
most relevant text segment or a list of sentences from the policy document given a question. On the contrary, we argue 
that providing users with a short text span from policy documents reduces the burden of searching the target 
information from a lengthy text segment. In this paper, we present PolicyQA, a dataset that contains 25,017 reading 
comprehension style examples curated from an existing corpus of 115 website privacy policies. PolicyQA provides 714 
human-annotated questions written for a wide range of privacy practices. We evaluate two existing neural QA models 
and perform rigorous analysis to reveal the advantages and challenges offered by PolicyQA.
"""


class PolicyQAConfig(datasets.BuilderConfig):
    """BuilderConfig for PolicyQA"""

    def __init__(self, **kwargs):
        """BuilderConfig for SQUAD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(PolicyQAConfig, self).__init__(**kwargs)


class PolicyQA(datasets.GeneratorBasedBuilder):
    """PolicyQA dataset."""

    BUILDER_CONFIGS = [
        PolicyQAConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://aclanthology.org/2020.findings-emnlp.66/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        train_file = self.config.data_files.get("train", [""])[0]
        dev_file = self.config.data_files.get("validation", [""])[0]
        test_file = self.config.data_files.get("test", [""])[0]

        splits = []
        if train_file:
            splits.append(datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_file}))
        if dev_file:
            splits.append(datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dev_file}))
        if test_file:
            splits.append(datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_file}))

        return splits

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logging.info("generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            squad = json.load(f)
            for article in squad["data"]:
                title = article.get("title", "").strip()
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"].strip()
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        id_ = qa["id"]

                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"].strip() for answer in qa["answers"]]

                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield id_, {
                            "title": title,
                            "context": context,
                            "question": question,
                            "id": id_,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }
