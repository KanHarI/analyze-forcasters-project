import argparse
import math
import os
import shutil
import sys
from typing import Optional, Callable, List, Tuple, Dict

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import beta

results_1st = dict()
results_2nd = dict()
results_total = dict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions-csv', type=str, required=True)
    parser.add_argument('--truth-csv', type=str, required=True)
    parser.add_argument('--end-only-1st-pred', type=str, required=False)
    parser.add_argument('--start-only-2nd-pred', type=str, required=False)
    parser.add_argument('--size-1st-questions', type=int, required=False)
    parser.add_argument('--size-2nd-questions', type=int, required=False)
    return parser.parse_args()


def get_stats(n_success, n_tries, confidence=0.7) -> Tuple[float, float, Tuple[float, float]]:
    """Returns: percent_observed, mean, [10th percentile, 90th percentile]"""
    dist = beta(n_success + 1, n_tries - n_success + 1)
    if n_tries == 0:
        return math.nan, math.nan, (math.nan, math.nan)
    return n_success / n_tries, dist.mean(), (dist.isf(1 - (1 - confidence) / 2), dist.isf((1 - confidence) / 2))


def parse_gt(gt_path: str) -> pd.DataFrame:
    gt = pd.read_csv(gt_path, index_col=0, header=None)
    return gt.T['Score']


def parse_predictions(predictions_path: str) -> pd.DataFrame:
    preds = pd.read_csv(predictions_path, index_col=0)
    return preds.T


def calc_brier_score(zippped_list: List[Tuple[float, float]]):
    count = 0
    brier_score = 0
    for pred_item, gt_item in zippped_list:
        if math.isnan(gt_item):
            continue
        count += 1
        if math.isnan(pred_item):
            pred_item = 0.5  # Assume 50% for nan predictions for scoring
        brier_score += (pred_item - gt_item) ** 2
    if count == 0:
        return math.nan
    return brier_score / count


def create_image(zipped_list, name, suffix, score):
    pred_buckets = {i / 10: (0, 0) for i in range(11)}
    for pred_item, gt_item in zipped_list:
        if math.isnan(gt_item) or math.isnan(pred_item):
            continue
        pred_item = math.floor(pred_item * 10 + 0.5) / 10.
        m, n = pred_buckets[pred_item]
        n += 1
        m += gt_item
        pred_buckets[pred_item] = (m, n)
    percents = []
    means = []
    min_ps = []
    max_ps = []
    xs = []
    for i in range(11):
        pred_x = i / 10
        xs.append(pred_x)
        _percent, _mean, (_min_p, _max_p) = get_stats(*pred_buckets[pred_x])
        percents.append(_percent)
        means.append(_mean)
        min_ps.append(_min_p)
        max_ps.append(_max_p)

    fig, ax = plt.subplots(1, 1)
    ax.plot(xs, percents, 'b-', lw=5, alpha=0.9, label='Percent Obs.')
    ax.fill_between(xs, min_ps, max_ps, color='r', alpha=0.3, label='70% Confidence')
    ax.plot(xs, xs, 'k-', lw=2, alpha=0.2, label='Reference')
    ax.plot(xs, means, 'r-', lw=2, alpha=0.2, label='Expected')
    plt.title(f"Brier score: {score}")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.savefig(os.path.join("out", name + suffix + ".png"))


def find_score_and_graph(gt: pd.DataFrame,
                         end_first: Optional[str] = None,
                         start_second: Optional[str] = None,
                         size_of_first: Optional[int] = None,
                         size_of_second: Optional[int] = None) -> Callable:
    def _find_score_and_graph(pred: pd.DataFrame):
        global results
        responder_name = pred.name
        is_first = responder_name < start_second
        is_second = responder_name > end_first
        if responder_name[:2] == 'M_':  # Aggregates
            is_first = True
            is_second = True
        pred_with_gt = list(zip(pred, gt))
        if is_first:
            first_pred_with_gt = pred_with_gt[:size_of_first + 1]
            first_score = calc_brier_score(first_pred_with_gt)
            create_image(first_pred_with_gt, responder_name, "_first", first_score)
            results_1st[responder_name] = first_score
        if is_second:
            second_pred_with_gt = pred_with_gt[size_of_first + 1:]
            second_score = calc_brier_score(second_pred_with_gt)
            create_image(second_pred_with_gt, responder_name, "_second", second_score)
            results_2nd[responder_name] = second_score
        if is_first and is_second:
            total_score = calc_brier_score(pred_with_gt)
            create_image(pred_with_gt, responder_name, "_total", total_score)
            results_total[responder_name] = total_score

    return _find_score_and_graph


def translate_percents_to_float(pred: pd.DataFrame) -> pd.DataFrame:
    def translate_single_item(item) -> float:
        if type(item) == str:
            return float(item[:-1]) / 100
        return math.nan

    return pred.apply(translate_single_item)


def translate_gt_item_to_float(item) -> float:
    if type(item) == str:
        return float(item)
    return math.nan


def print_sorted(results_dict: Dict[str, float], challenge: str):
    print(challenge + ":")
    names = list(results_dict.keys())
    names.sort(key=lambda x: results_dict[x])
    for i, name in enumerate(names):
        print(i + 1, name, results_dict[name])
    print("\n\n")


def print_sorted_forecasts():
    print_sorted(results_1st, "First challenge")
    print_sorted(results_2nd, "Second challenge")
    print_sorted(results_total, "Total 2020 predictions score")


def main() -> int:
    args = parse_args()
    gt = parse_gt(args.truth_csv)
    gt = gt.apply(translate_gt_item_to_float)
    predictions = parse_predictions(args.predictions_csv)
    predictions = predictions.apply(translate_percents_to_float)
    predictions['M_mean'] = predictions.mean(numeric_only=True, axis=1)
    predictions['M_median'] = predictions.median(numeric_only=True, axis=1)
    gt = list(gt)  # Who has time to remember how to zip 2 dataframes...

    if os.path.exists("out"):
        shutil.rmtree("out")

    os.makedirs("out")

    predictions.apply(find_score_and_graph(gt,
                                           args.end_only_1st_pred,
                                           args.start_only_2nd_pred,
                                           args.size_1st_questions,
                                           args.size_2nd_questions)
                      )
    print_sorted_forecasts()
    return 0


if __name__ == "__main__":
    sys.exit(main())
