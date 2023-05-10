import json


def convert_to_intent_classification(split):
    writer = open(f"intent_classification_{split}.json", "w", encoding="utf8")
    with open(f'bio_format/{split}/seq.in', encoding="utf8") as f1, \
            open(f'bio_format/{split}/label', encoding="utf8") as f2:
        for sentence, label in zip(f1, f2):
            ex = {"sentence": sentence.strip(), "label": label.strip()}
            writer.write(json.dumps(ex) + "\n")
    writer.close()


def convert_to_slot_filling(split, type):
    writer = open(f"{type}_slot_filling_{split}.json", "w", encoding="utf8")
    with open(f'bio_format/{split}/seq.in', encoding="utf8") as f1, \
            open(f'bio_format/{split}/seq_{type}.out', encoding="utf8") as f2:
        for sentence, label in zip(f1, f2):
            tokens = sentence.strip().split()
            labels = label.strip().split()
            assert len(tokens) == len(labels)
            ex = {"tokens": tokens, "labels": labels}
            writer.write(json.dumps(ex) + "\n")
    writer.close()


def generate_slot_tags(type):
    tags = set()
    for split in ["train", "valid", "test"]:
        with open(f'bio_format/{split}/seq_{type}.out', encoding="utf8") as f:
            for label in f:
                labels = label.strip().split()
                tags.update(labels)

    tags = list(tags)
    tags.sort()
    assert tags[-1] == "O"
    tags = tags[:-1]
    tags = ["PAD", "UNK", "O"] + tags
    with open(f"{type}_tags.txt", "w", encoding="utf8") as writer:
        writer.write("\n".join(tags))


convert_to_intent_classification("train")
convert_to_intent_classification("valid")
convert_to_intent_classification("test")
convert_to_slot_filling("train", "type_I")
convert_to_slot_filling("valid", "type_I")
convert_to_slot_filling("test", "type_I")
convert_to_slot_filling("train", "type_II")
convert_to_slot_filling("valid", "type_II")
convert_to_slot_filling("test", "type_II")
generate_slot_tags("type_I")
generate_slot_tags("type_II")
