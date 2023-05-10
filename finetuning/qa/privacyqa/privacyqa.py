# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import os

import datasets

_CITATION = """\
@inproceedings{ravichander-etal-2019-question,
    title = "Question Answering for Privacy Policies: Combining Computational and Legal Perspectives",
    author = "Ravichander, Abhilasha  and
      Black, Alan W  and
      Wilson, Shomir  and
      Norton, Thomas  and
      Sadeh, Norman",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1500",
    doi = "10.18653/v1/D19-1500",
    pages = "4947--4958"
}
"""

_DESCRIPTION = """\
PrivacyQA, a corpus consisting of 1750 questions about the privacy policies of mobile applications, 
and over 3500 expert annotations of relevant answers. 
"""


class PrivacyQA(datasets.GeneratorBasedBuilder):
    """The PrivacyQA Corpus."""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text import of PrivacyQA",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "_id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=["Irrelevant", "Relevant"]),
                }
            ),
            # No default supervised_keys (as we have to pass both premise
            # and hypothesis as input).
            supervised_keys=None,
            homepage="https://aclanthology.org/D19-1500/",
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
        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for idx, row in enumerate(reader):
                if "Label" in row:
                    label = row["Label"]
                else:
                    labels = []
                    for ann in ["Ann1", "Ann2", "Ann3", "Ann4", "Ann5", "Ann6"]:
                        if row[ann] != "None":
                            labels.append(row[ann])

                    label = None
                    counter = 0
                    for l in labels:
                        curr_frequency = labels.count(l)
                        if curr_frequency > counter:
                            counter = curr_frequency
                            label = l
                    assert label in ["Irrelevant", "Relevant"]

                _id = f'{row["DocID"]}.{row["QueryID"]}.{row["SentID"]}'
                yield _id, {
                    "_id": _id,
                    "question": row["Query"],
                    "sentence": row["Segment"],
                    "label": label
                }
