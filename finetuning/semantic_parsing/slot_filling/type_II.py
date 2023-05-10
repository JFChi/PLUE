import os
import json
import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@inproceedings{ahmad-etal-2021-intent,
    title = "Intent Classification and Slot Filling for Privacy Policies",
    author = "Ahmad, Wasi  and
      Chi, Jianfeng  and
      Le, Tu  and
      Norton, Thomas  and
      Tian, Yuan  and
      Chang, Kai-Wei",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.340",
    doi = "10.18653/v1/2021.acl-long.340",
    pages = "4402--4417",
}
"""

_DESCRIPTION = """\
PolicyIE is an English corpus consisting of 5,250 intent and 11,788 slot annotations spanning 31 privacy policies of 
websites and mobile applications. PolicyIE corpus is a challenging real-world benchmark with limited labeled examples 
reflecting the cost of collecting large-scale annotations from domain experts.
"""


class PolicyIEConfig(datasets.BuilderConfig):
    """BuilderConfig for PolicyIE"""

    def __init__(self, **kwargs):
        """BuilderConfig PolicyIE.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(PolicyIEConfig, self).__init__(**kwargs)


class PolicyIE(datasets.GeneratorBasedBuilder):
    """PolicyIE dataset."""

    BUILDER_CONFIGS = [
        PolicyIEConfig(name="policyie", version=datasets.Version("1.0.0"), description="PolicyIE dataset"),
    ]

    def _info(self):
        class_labels = [c.strip() for c in open("type_II_tags.txt").readlines()]
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(names=class_labels)
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://aclanthology.org/2021.acl-long.340/",
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
