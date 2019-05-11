#!/usr/bin/env python3

from collections import namedtuple
import json
import numpy as np
import os
import scipy.stats

Result = namedtuple("Result", ("score", "err"))

def jia_data(experiment_path, experiment_field, n_runs):
    scores = []
    for i_run in range(n_runs):
        with open(os.path.join("exp", experiment_path, "stats.%d.json" % i_run)) as data_f:
            data = json.load(data_f)
            scores.append(data[experiment_field]["sentence"]["accuracy"])
    return Result(
        np.mean(scores),
        np.std(scores, ddof=1)# / np.sqrt(len(scores)),
    )

def jia_sig(exp1_path, exp2_path, n_runs):
    def load_log(exp_path, i_run):
        seq = []
        with open(os.path.join("exp", exp_path, "train.%d.out" % i_run)) as log_f:
            found_dev = False
            for line in log_f:
                if line.startswith("Dev data:"):
                    found_dev = True
                if not found_dev:
                    continue
                if line.startswith("  sequence correct"):
                    result = line.strip().split()[-1]
                    seq.append(result == "True")
        return seq

    wins = []
    for i_run in range(n_runs):
        log1 = load_log(exp1_path, i_run)
        log2 = load_log(exp2_path, i_run)
        for r1, r2 in zip(log1, log2):
            if r1 == r2:
                continue
            wins.append(r2)

    x = np.sum(wins)
    n = len(wins)
    p = scipy.stats.binom_test(x, n)
    return p, x, n

def ingest(log_file, trim=4, ignore={"#"}):
    out = {}

    def insert(key, value):
        key_parts = list(reversed(key.split("/")))
        active = out
        while len(key_parts) > 1:
            part = key_parts.pop()
            if part in active and not isinstance(active[part], dict):
                active[part] = {"_value": active[part]}
            if part not in active:
                active[part] = {}
            active = active[part]
        part = key_parts.pop()
        if part in active:
            if not isinstance(active[part], list):
                active[part] = [active[part]]
            active[part].append(value)
        else:
            active[part] = value

    for line in log_file:
        parts = line.strip().split(" ", trim+1)[trim:]
        if len(parts) != 2:
            continue
        key, value = parts
        try:
            value = float(value)
        except ValueError as e:
            pass
        if key in ignore:
            continue
        insert(key, value)

    return out

def jda_data(experiment_path, key, n_runs):
    scores = []
    for i_run in range(n_runs):
        with open(os.path.join("exp", experiment_path, "eval.%d.err" % i_run)) as data_f:
            data = ingest(data_f)
            for part in key.split("/"):
                data = data[part]
            scores.append(data)
    return Result(
        np.mean(scores),
        np.std(scores, ddof=1)# / np.sqrt(len(scores))
    )

def jda_sig(exp1_path, exp2_path, n_runs):
    # TODO just a Z test
    pass

def jda_lm_data(experiment_path, val_key, test_key, aug):
    with open(os.path.join("exp", experiment_path, "eval.0.err")) as data_f:
        data = ingest(data_f, trim=0, ignore={"#", "Reading", "Loading"})
        if aug:
            best_val = 1+np.argmin(data[val_key]["ppl"][1:])
        else:
            best_val = 0
        val = data[val_key]["ppl"][best_val]
        test = data[test_key]["ppl"][best_val]
        return Result(test, None)

def format(headers, results):
    print(r"\begin{tabular}{l%s}" % ("c" * len(headers)))
    print(r"\toprule")
    print("&", " & ".join(headers), r"\\")
    out_table = []
    for row_name, row in results:
        out_row = [row_name]
        for result in row:
            if result.err is None:
                out_row.append(r"%0.2f" % result.score)
            else:
                out_row.append(r"%0.2f \pm %0.2f" % result)
        out_table.append(" & ".join(out_row))
    print((r" \\" + "\n").join(out_table))
    print(r"\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print()

def main():
    print("% SEMANTIC PARSING: GEOQUERY / LFS")
    format(["query", "question"],
    [
        ("baseline", [
            jia_data("semparse_geo_logic/query_baseline_jia", "dev", 10),
            jia_data("semparse_geo_logic/question_baseline_jia", "dev", 10),
        ]),
        (r"JL", [
            jia_data("semparse_geo_logic/query_recomb_jia", "dev", 10),
            jia_data("semparse_geo_logic/question_recomb_jia", "dev", 10),
        ]),
        ("subst", [
            jia_data("semparse_geo_logic/query_retrieval_jia", "dev", 10),
            jia_data("semparse_geo_logic/question_retrieval_jia", "dev", 10),
        ]),
        ("subst+concat", [
            jia_data("semparse_geo_logic/query_retrieval2_jia", "dev", 10),
            jia_data("semparse_geo_logic/question_retrieval2_jia", "dev", 10),
        ]),
    ])
    for split in ["query", "question"]:
        for model1 in ["baseline", "recomb"]:
            for model2 in ["retrieval", "retrieval2"]:
                sig, x, n = jia_sig(
                    "semparse_geo_logic/%s_%s_jia" % (split, model1),
                    "semparse_geo_logic/%s_%s_jia" % (split, model2),
                    10,
                )
                print("%% %s: %s > %s? %0.3f (%d / %d)" % (split, model2, model1, sig, x, n))

    print("% SEMANTIC PARSING: GEOQUERY / SQL")
    format(["query", "question"],
    [
        ("baseline", [
            jda_data("semparse_geo_sql/query_eval_baseline", "train/00149/eval_test/acc", 10),
            jda_data("semparse_geo_sql/question_eval_baseline", "train/00149/eval_test/acc", 10),
        ]),
        ("subst", [
            jda_data("semparse_geo_sql/query_eval_retrieval", "train/00149/eval_test/acc", 10),
            jda_data("semparse_geo_sql/question_eval_retrieval", "train/00149/eval_test/acc", 10),
        ]),
    ])
    #for split in ["query", "question"]:
    #    for model1 in ["baseline"]:
    #        for model2 in ["retrieval"]:
    #            sig, x, n = jda_sig(
    #                "semparse_geo_sql/%s_eval_%s" % (split, model1),
    #                "semparse_geo_sql/%s_eval_%s" % (split, model2),
    #                10,
    #            )
    #            print("%% %s: %s > %s? %0.3f (%d / %d)" % (split, model2, model1, sig, x, n))
    #print()

    print("% SEMANTIC PARSING: SCHOLAR / SQL")
    format(["query", "question"],
    [
        ("baseline", [
            jda_data("semparse_scholar_sql/query_eval_baseline", "train/00149/eval_test/acc", 10),
            jda_data("semparse_scholar_sql/question_eval_baseline", "train/00149/eval_test/acc", 10),
        ]),
        ("subst", [
            jda_data("semparse_scholar_sql/query_eval_retrieval", "train/00149/eval_test/acc", 10),
            jda_data("semparse_scholar_sql/question_eval_retrieval", "train/00149/eval_test/acc", 10),
        ]),
    ])

    print("% SCAN")
    format(["jump/scan", "jump/nacs", "turn/scan", "turn/nacs"],
    [
        ("baseline", [
            jda_data("scan_jump/baseline", "train/00149/eval_test/acc", 10),
            jda_data("nacs_jump/baseline", "train/00149/eval_test/acc", 10),
            jda_data("scan_turn/baseline", "train/00149/eval_test/acc", 10),
            jda_data("nacs_turn/baseline", "train/00149/eval_test/acc", 10),
        ]),
        ("subst", [
            jda_data("scan_jump/retrieval", "train/00149/eval_test/acc", 10),
            jda_data("nacs_jump/retrieval", "train/00149/eval_test/acc", 10),
            jda_data("scan_turn/retrieval", "train/00149/eval_test/acc", 10),
            jda_data("nacs_turn/retrieval", "train/00149/eval_test/acc", 10),
        ]),
    ])
    

    print("% LM")
    format(["eng", "kin", "lao", "na", "pus", "tok"],
    [
        ("baseline", [
            jda_lm_data("lm_eng/mkn", "eval_val", "eval_test", aug=False),
            jda_lm_data("lm_kin/mkn", "eval_val", "eval_test", aug=False),
            jda_lm_data("lm_lao/mkn", "eval_val", "eval_test", aug=False),
            jda_lm_data("lm_na/mkn",  "eval_val", "eval_test", aug=False),
            jda_lm_data("lm_pus/mkn", "eval_val", "eval_test", aug=False),
            jda_lm_data("lm_tok/mkn", "eval_val", "eval_test", aug=False),
        ]),

        ("subst", [
            jda_lm_data("lm_eng/mkn", "eval_val", "eval_test", aug=True),
            jda_lm_data("lm_kin/mkn", "eval_val", "eval_test", aug=True),
            jda_lm_data("lm_lao/mkn", "eval_val", "eval_test", aug=True),
            jda_lm_data("lm_na/mkn",  "eval_val", "eval_test", aug=True),
            jda_lm_data("lm_pus/mkn", "eval_val", "eval_test", aug=True),
            jda_lm_data("lm_tok/mkn", "eval_val", "eval_test", aug=True),
        ]),
    ])

if __name__ == "__main__":
    main()
