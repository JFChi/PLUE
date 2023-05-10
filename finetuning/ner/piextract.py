# borrowed from https://github.com/huggingface/datasets/blob/1.18.X/datasets/conll2003/conll2003.py

import os
import json
import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
article{piextract_pets21,
    author = {Duc Bui and Kang G. Shin and Jong-Min Choi and Junbum Shin},
    doi = {doi:10.2478/popets-2021-0019},
    url = {https://doi.org/10.2478/popets-2021-0019},
    title = {Automated Extraction and Presentation of Data Practices in Privacy Policies},
    journal = {Proceedings on Privacy Enhancing Technologies},
    number = {2},
    volume = {2021},
    year = {2021},
    pages = {88--110}
}
"""

_DESCRIPTION = """\
The dataset contains privacy policy sentences with annotated data practices. There are four types of data actions:
- COLLECT
- SHARE
- NOT_COLLECT
- NOT_SHARE
See details from https://github.com/um-rtcl/piextract_dataset/blob/master/dataset/annotation_guidelines.pdf.
"""


class PIExtractConfig(datasets.BuilderConfig):
    """BuilderConfig for PiExtract"""

    def __init__(self, **kwargs):
        """BuilderConfig PiExtract.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(PIExtractConfig, self).__init__(**kwargs)


class PIExtract(datasets.GeneratorBasedBuilder):
    """PiExtract dataset."""

    BUILDER_CONFIGS = [
        PIExtractConfig(name="piextract", version=datasets.Version("1.0.0"), description="PI-Extract dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-COLLECT",
                                "I-COLLECT",
                                "B-NOT_COLLECT",
                                "I-NOT_COLLECT",
                                "B-SHARE",
                                "I-SHARE",
                                "B-NOT_SHARE",
                                "I-NOT_SHARE",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://rtcl.eecs.umich.edu/rtclweb/assets/publications/2021/pets21-duc.pdf",
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
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                ex = json.loads(line.strip())
                yield idx, {
                    "id": str(idx),
                    "tokens": ex["tokens"],
                    "ner_tags": ex["labels"],
                }
