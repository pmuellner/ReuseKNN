import numpy as np
from algorithms import metrics
from collections import defaultdict
from scipy.stats import mannwhitneyu

"""def evaluate_user_groups(models, **kwargs):
    results = dict()
    for group, users in kwargs.items():
        results[group] = evaluate(models, users=users)

    return results"""


def evaluate(models, models_q=None):
    results, results_samples = _evaluate(models)
    if models_q is not None:
        results_q, results_q_samples = _evaluate_q(models_q)
        return {**results, **results_q}, {**results_samples, **results_q_samples}
    else:
        return {**results, **results_samples}

def _evaluate_q(models):
    results = defaultdict(list)
    results_samples = defaultdict(list)
    for k_idx, m in enumerate(models):
        avg, sample = metrics.avg_neighborhood_size_q(m, n_queries=100)
        results["avg_neighborhood_size_q"].append(avg)
        results_samples["avg_neighborhood_size_q"].append(sample)
        avg, sample = metrics.avg_rating_overlap_q(m, n_queries=100)
        results["avg_rating_overlap_q"].append(avg)
        results_samples["avg_rating_overlap_q"].append(sample)
    return results, results_samples


def _evaluate(models, users=None):
    results = defaultdict(list)
    results_samples = defaultdict(list)
    for k_idx, m in enumerate(models):
        avg, sample = metrics.mean_absolute_error(m, users=users)
        results["mean_absolute_error"].append(avg)
        results_samples["mean_absolute_error"].append(sample)
        avg, sample = metrics.avg_privacy_risk_dp(m, users=users)
        results["avg_privacy_risk_dp"].append(avg)
        results_samples["avg_privacy_risk_dp"].append(sample)
        avg, sample = metrics.avg_privacy_risk(m, users=users)
        results["avg_privacy_risk"].append(avg)
        results_samples["avg_privacy_risk"].append(sample)
        avg, sample = metrics.ndcg(m, users=users)
        results["avg_ndcg"].append(avg)
        results_samples["avg_ndcg"].append(sample)

        frac_vulnerables = metrics.fraction_vulnerables(m, users=users)
        results["fraction_vulnerables"].append(frac_vulnerables)
        if 1 - frac_vulnerables > 0:
            below_threshold_pr = (m.privacy_risk[m.privacy_risk < m.threshold])

        else:
            # if there are no secures
            below_threshold_pr = [0]
        results["avg_privacy_risk_dp_secures"].append(np.mean(below_threshold_pr))
        results_samples["avg_privacy_risk_dp_secures"].append(below_threshold_pr)

        results["recommendation_frequency"].append(metrics.recommendation_frequency(m, n=10, users=users))

    return results, results_samples

def significance_tests(results1, results2):
    significance = dict()

    mae_significance = dict()
    mae_significance["<"] = _significance_test(results1["mean_absolute_error"], results2["mean_absolute_error"], h0="<")
    mae_significance[">"] = _significance_test(results1["mean_absolute_error"], results2["mean_absolute_error"], h0=">")
    mae_significance["=="] = _significance_test(results1["mean_absolute_error"], results2["mean_absolute_error"], h0="==")
    significance["mean_absolute_error"] = mae_significance

    pr_significance = dict()
    pr_significance["<"] = _significance_test(results1["avg_privacy_risk_dp"], results2["avg_privacy_risk_dp"], h0="<")
    pr_significance[">"] = _significance_test(results1["avg_privacy_risk_dp"], results2["avg_privacy_risk_dp"], h0=">")
    pr_significance["=="] = _significance_test(results1["avg_privacy_risk_dp"], results2["avg_privacy_risk_dp"], h0="==")
    significance["privacy_risk_dp"] = pr_significance

    pr_secure_significance = dict()
    pr_secure_significance["<"] = _significance_test(results1["avg_privacy_risk_dp_secures"], results2["avg_privacy_risk_dp_secures"], h0="<")
    pr_secure_significance[">"] = _significance_test(results1["avg_privacy_risk_dp_secures"], results2["avg_privacy_risk_dp_secures"], h0=">")
    pr_secure_significance["=="] = _significance_test(results1["avg_privacy_risk_dp_secures"], results2["avg_privacy_risk_dp_secures"], h0="==")
    significance["privacy_risk_dp_secures"] = pr_secure_significance

    neighborhood_size_significance = dict()
    neighborhood_size_significance["<"] = _significance_test_q(results1["avg_neighborhood_size_q"], results2["avg_neighborhood_size_q"], h0="<")
    neighborhood_size_significance[">"] = _significance_test_q(results1["avg_neighborhood_size_q"], results2["avg_neighborhood_size_q"], h0=">")
    neighborhood_size_significance["=="] = _significance_test_q(results1["avg_neighborhood_size_q"], results2["avg_neighborhood_size_q"], h0="==")
    significance["avg_neighborhood_size_q"] = neighborhood_size_significance

    rating_overlap_significance = dict()
    rating_overlap_significance["<"] = _significance_test_q(results1["avg_rating_overlap_q"], results2["avg_rating_overlap_q"], h0="<")
    rating_overlap_significance[">"] = _significance_test_q(results1["avg_rating_overlap_q"], results2["avg_rating_overlap_q"], h0=">")
    rating_overlap_significance["=="] = _significance_test_q(results1["avg_rating_overlap_q"], results2["avg_rating_overlap_q"], h0="==")
    significance["avg_rating_overlap_q"] = rating_overlap_significance

    return significance

def _mann_whitney_u_test(x, y, alternative="less"):
    u, p = mannwhitneyu(x, y, alternative=alternative)
    nx = len(x)
    ny = len(y)
    mu_u = (nx * ny) / 2
    sigma_u = np.sqrt(nx * ny * (nx + ny + 1) / 12)
    z = (u - mu_u) / sigma_u
    n = len(x) + len(y)
    r = z / np.sqrt(n)

    return u, p, r

def _significance_test_q(x, y, h0="<"):
    n_ks = len(x)
    results = []
    for k_idx in range(n_ks):
        n_queries = len(x[k_idx])
        results_k = []
        for q in range(n_queries):
            n_samples = len(x[k_idx][q])
            if np.array_equal(x[k_idx][q], y[k_idx][q]):
                results_k.append({"p": np.inf, "sample_size": n_samples, "r": np.nan, "U": np.nan})
            elif h0 == "<":
                u, p, r = _mann_whitney_u_test(x[k_idx][q], y[k_idx][q], alternative="greater")
                results_k.append({"p": p, "sample_size": n_samples, "r": r, "U": u})
            elif h0 == ">":
                u, p, r = _mann_whitney_u_test(x[k_idx][q], y[k_idx][q], alternative="less")
                results_k.append({"p": p, "sample_size": n_samples, "r": r, "U": u})
            elif h0 == "==":
                u, p, r = _mann_whitney_u_test(x[k_idx][q], y[k_idx][q], alternative="two-sided")
                results_k.append({"p": p, "sample_size": n_samples, "r": r, "U": u})
            else:
                print("Unknown h0!")
        results.append(results_k)

    return results


def _significance_test(x, y, h0="<"):
    n_ks = len(x)
    n_samples = len(x[0])

    #n_ks, n_samples = np.array(x).shape
    results = []
    for k_idx in range(n_ks):
        if np.array_equal(x[k_idx], y[k_idx]):
            print("x equal y")
            results.append({"p": np.inf, "sample_size": n_samples, "r": np.nan, "U": np.nan})
        elif h0 == "<":
            u, p, r = _mann_whitney_u_test(x[k_idx], y[k_idx], alternative="greater")
            results.append({"p": p, "sample_size": n_samples, "r": r, "U": u})
        elif h0 == ">":
            u, p, r = _mann_whitney_u_test(x[k_idx], y[k_idx], alternative="less")
            results.append({"p": p, "sample_size": n_samples, "r": r, "U": u})
        elif h0 == "==":
            u, p, r = _mann_whitney_u_test(x[k_idx], y[k_idx], alternative="two-sided")
            results.append({"p": p, "sample_size": n_samples, "r": r, "U": u})
        else:
            print("Unknown h0!")

    return results