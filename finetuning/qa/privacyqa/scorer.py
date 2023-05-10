import os
import csv
import json
import argparse
from statistics import mean
from collections import Counter, OrderedDict


def compute_f1(gold_sent_ids, pred_sent_ids):
    # gold_sent_ids = a list of str (SentID)
    # pred_sent_ids = a list of str (SentID)
    if len(gold_sent_ids) == 0 or len(pred_sent_ids) == 0:
        # If either gold or prediction is no-answer,
        # then F1 is 1 if they agree, 0 otherwise.
        agree = float(len(gold_sent_ids) == len(pred_sent_ids))
        return agree, agree, agree

    common = Counter(gold_sent_ids) & Counter(pred_sent_ids)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_sent_ids)
    recall = 1.0 * num_same / len(gold_sent_ids)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def human_performance(results):
    total_prec, total_rec, total_f1 = [], [], []
    for eid, ex in results.items():
        annotator_ids = list(ex['gold'].keys())
        assert len(annotator_ids) >= 2
        query_prec, query_rec, query_f1 = [], [], []
        for pred_id in annotator_ids:
            predictions = ex['gold'][pred_id]
            pred_prec, pred_rec, pred_f1 = 0.0, 0.0, 0.0
            for anno_id, judgements in ex['gold'].items():
                if anno_id != pred_id:
                    prec, rec, f1 = compute_f1(judgements, predictions)
                    if f1 > pred_f1:
                        pred_prec, pred_rec, pred_f1 = prec, rec, f1

            query_prec.append(pred_prec)
            query_rec.append(pred_rec)
            query_f1.append(pred_f1)

        total_prec.append(mean(query_prec))
        total_rec.append(mean(query_rec))
        total_f1.append(mean(query_f1))

    human_prec, human_rec = mean(total_prec), mean(total_rec)
    human_f1 = (2 * human_prec * human_rec) / (human_prec + human_rec)
    print(f"[Human performance] Precision: {round(human_prec * 100, 2)}, "
          f"Recall: {round(human_rec * 100, 2)}, "
          f"F1: {round(human_f1 * 100, 2)}")


def score(results, score_file=None):
    total_prec, total_rec, total_f1 = [], [], []
    scorer_out = []
    for eid, ex in results.items():
        # ex = {
        #     "pred": [],
        #     "gold": {"Ann1": [], "Ann2": [], "Ann3": [], "Ann4": [], "Ann5": [], "Ann6": []}
        # }
        # eid = company_id.question_id
        precision, recall, f1_score = 0.0, 0.0, 0.0
        for anno_id, judgements in ex['gold'].items():
            # judgements is a list of relevant SentID
            # ex['pred'] is the list of predicted SentID as relevant
            prec, rec, f1 = compute_f1(judgements, ex['pred'])
            if f1 > f1_score:
                precision, recall, f1_score = prec, rec, f1

        total_prec.append(precision)
        total_rec.append(recall)
        total_f1.append(f1_score)
        scorer_out.append({
            "id": eid,
            **ex,
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
        })

    print(f"[Model performance] Precision: {round(mean(total_prec) * 100, 2)}, "
          f"Recall: {round(mean(total_rec) * 100, 2)}, "
          f"F1: {round(mean(total_f1) * 100, 2)}")
    if score_file:
        with open(score_file, 'w') as writer:
            json.dump(scorer_out, writer, indent=4)


def main(args):
    results = {}
    anno_ids = ["Ann1", "Ann2", "Ann3", "Ann4", "Ann5", "Ann6"]
    with open(args.ground_truth, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for idx, row in enumerate(reader):
            assert list(row.keys()) == [
                'Folder', 'DocID', 'QueryID', 'SentID', 'Split', 'Query', 'Segment',
                'Any_Relevant', 'Ann1', 'Ann2', 'Ann3', 'Ann4', 'Ann5', 'Ann6'
            ]
            _id = row["QueryID"]
            if _id not in results:
                results[_id] = {
                    "question": row["Query"],
                    "pred": [],
                    "gold": {}}

            for anno_id in anno_ids:
                anno = row[anno_id].lower()
                assert anno in ["relevant", "irrelevant", "none"]
                if anno != 'none' and anno_id not in results[_id]["gold"]:
                    results[_id]["gold"][anno_id] = []
                if anno == "relevant":
                    sent_id = row["SentID"].split("_")[3]
                    results[_id]["gold"][anno_id].append(sent_id)

    with open(args.predictions, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for idx, row in enumerate(reader):
            id_parts = row["id"].split(".")  # company_id.question_id.sentence_id
            _id = id_parts[1]
            if int(row["prediction"]) == 1:
                sent_id = id_parts[2].split("_")[3]
                results[_id]["pred"].append(sent_id)

    unans_groundtruth = sum([all([len(v2) == 0 for _, v2 in v["gold"].items()]) for k, v in results.items()])
    unans_predictions = sum([len(v["pred"]) == 0 for k, v in results.items()])
    score_file = os.path.join(args.output_dir, "scorer_out.json")
    print("[predictions] unanswerable questions = ", unans_predictions)
    print("[groundtruth] unanswerable questions = ", unans_groundtruth)
    score(results, score_file)
    human_performance(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True, help='path to predictions')
    parser.add_argument('--ground_truth', required=True, help='path to ground-truth')
    parser.add_argument('--output_dir', required=True, help='path to output directory')
    args = parser.parse_args()

    main(args)
