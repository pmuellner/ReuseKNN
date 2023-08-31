import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=True)
from algorithms.knn_neighborhood import UserKNN
import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import KFold
from datetime import datetime as dt
from collections import defaultdict
import pickle as pl
from algorithms import evaluation, utils
import argparse

def run(trainset, testset, K, configuration=None):
    """
    :param trainset: surprise trainset
    :param testset: surprise testset
    :param K: list containing the values for the parameter k (no. of neighbors)
    :param configuration: dict containing the experiment settings
        - reuse: use naive reuse (UserKNN+Reuse)
        - sim: user-user similarity matrix
        - expect_scores: reusability scores according to Expect
        - gain_scores: reusability scores according to Gain
        - overlap: matrix containing the rating overlap between users
        - rated_items: dict of rated items per user
    :return: models and predictions for every k
    """
    if configuration is None:
        configuration = {}

    reuse = configuration.get("reuse", False)
    sim = configuration.get("sim", None)
    expect_scores = configuration.get("expect_scores", None)
    gain_scores = configuration.get("gain_scores", None)
    overlap = configuration.get("overlap", None)
    rated_items = configuration.get("rated_items", None)

    thresholds = configuration.get("thresholds", None)
    use_dp = configuration.get("use_dp", False)

    config_str = str({"reuse": reuse, "precomputed_sim": sim is not None,
                      "precomputed_expect": expect_scores is not None, "precomputed_gain": gain_scores is not None,
                      "use_dp": use_dp, "precomputed_overlap": overlap is not None,
                      "rated_items": rated_items is not None})

    # generate recomendations with the given configuration
    t0 = dt.now()
    print("Started training model with K: " + str(K) + " and " + config_str)
    results = defaultdict(list)
    for idx, k in enumerate(K):
        if thresholds is not None:
            th = thresholds[idx]
        else:
            th = 0
        model = UserKNN(k=k, reuse=reuse, sim=sim, expect_scores=expect_scores, gain_scores=gain_scores, threshold=th,
                        use_dp=use_dp, overlap=overlap, rated_items=rated_items)
        model.fit(trainset)
        predictions = model.test(testset)
        results["models"].append(model)
        results["predictions"].append(predictions)

    print("Training finished after " + str(dt.now() - t0))
    return results["models"], results["predictions"]


if __name__ == "__main__":
    DATASET_PATH = "../../datasets/"
    RESULT_PATH = "results/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="ml-100k")
    parser.add_argument("--use_dp", default=True)
    args = parser.parse_args()

    if isinstance(args.use_dp, str):
        if args.use_dp.lower() == "true":
            args.use_dp = True
        else:
            args.use_dp = False

    if args.dataset_name == "ml-100k":
        data_df = pd.read_csv(DATASET_PATH + "ml-100k/u.data", sep="\t",
                              names=["user_id", "item_id", "rating", "timestamp"],
                              usecols=["user_id", "item_id", "rating"])
        reader = Reader(rating_scale=(1, 5))
    elif args.dataset_name == "ml-1m":
        data_df = pd.read_csv(DATASET_PATH + "ml-1m/ratings.dat", sep="::",
                              names=["user_id", "item_id", "rating", "timestamp"],
                              usecols=["user_id", "item_id", "rating"])
        reader = Reader(rating_scale=(1, 5))
    elif args.dataset_name == "goodreads":
        data_df = pd.read_csv(DATASET_PATH + "goodreads/sample.csv", sep=";", names=["user_id", "item_id", "rating"])
        reader = Reader(rating_scale=(1, 5))
    elif args.dataset_name == "lfm":
        data_df = pd.read_csv(DATASET_PATH + "lfm/artist_ratings.csv", sep=";", names=["user_id", "item_id", "rating"])
        reader = Reader(rating_scale=(1, 1000))
    elif args.dataset_name == "ciao":
        data_df = pd.read_csv(DATASET_PATH + "ciao/ciao.csv", sep=";", names=["user_id", "item_id", "rating"])
        reader = Reader(rating_scale=(1, 5))
    elif args.dataset_name == "douban":
        data_df = pd.read_csv(DATASET_PATH + "douban/douban.csv", sep=";", names=["user_id", "item_id", "rating"])
        reader = Reader(rating_scale=(1, 5))
    else:
        print("error")
        data_df = pd.DataFrame()
        reader = Reader()
    dataset = Dataset.load_from_df(data_df, reader=reader)

    if args.use_dp:
        RESULT_PATH = "dp/" + args.dataset_name
    else:
        RESULT_PATH = "nodp/" + args.dataset_name

    # use 5-fold cross-validation and 5-30 neighbors
    n_processed_folds = 0
    folds = KFold(n_splits=5, random_state=42)

    K = [5, 10, 15, 20, 25, 30]
    K_q_idx = 1
    data_usage = defaultdict(list)
    mean_absolute_error = defaultdict(list)
    ndcg = defaultdict(list)
    recommendation_frequency = defaultdict(list)
    fraction_vulnerables = defaultdict(list)
    privacy_risk = defaultdict(list)
    neighborhood_size_q = defaultdict(list)
    rating_overlap_q = defaultdict(list)
    privacy_risk_secures = defaultdict(list)
    significance_test_results = defaultdict(list)
    significance_test_results_full = defaultdict(list)
    thresholds = []
    for trainset, testset in folds.split(dataset):
        starttime = dt.now()

        sim = UserKNN.compute_similarities(trainset, min_support=1, kind="cosine")
        expect_scores = UserKNN.compute_expect_scores(trainset)
        gain_scores = UserKNN.compute_gain_scores(trainset)
        overlap = UserKNN.compute_overlap(trainset)
        rated_items = UserKNN.compute_rated_items(trainset)

        # Determine threshold
        models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "sim": sim, "overlap": overlap,
                                                               "rated_items": rated_items, "use_dp": False})
        threshs = [m.get_privacy_threshold() for m in models]
        thresholds.append(threshs)

        # UserKNN (in case use_dp=True, DP is applied to all vulnerable users)
        models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "sim": sim, "overlap": overlap,
                                                               "rated_items": rated_items, "thresholds": threshs,
                                                               "use_dp": args.use_dp})
        results, userknn_results_samples = evaluation.evaluate(models, [models[K_q_idx]])
        mean_absolute_error["userknn"].append(results["mean_absolute_error"])
        ndcg["userknn"].append(results["avg_ndcg"])
        recommendation_frequency["userknn"].append(results["recommendation_frequency"])
        fraction_vulnerables["userknn"].append(results["fraction_vulnerables"])
        privacy_risk["userknn"].append(results["avg_privacy_risk"])
        neighborhood_size_q["userknn"].append(results["avg_neighborhood_size_q"])
        rating_overlap_q["userknn"].append(results["avg_rating_overlap_q"])
        privacy_risk_secures["userknn"].append(results["avg_privacy_risk_secures"])
        data_usage["userknn"].append(userknn_results_samples["avg_data_usage"])

        # Baseline: UserKNN without DP
        models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "sim": sim, "overlap": overlap,
                                                               "rated_items": rated_items, "use_dp": False,
                                                               "thresholds": [np.inf for _ in range(len(K))]})
        results, results_samples = evaluation.evaluate(models, [models[K_q_idx]])
        mean_absolute_error["userknn_no"].append(results["mean_absolute_error"])
        ndcg["userknn_no"].append(results["avg_ndcg"])
        recommendation_frequency["userknn_no"].append(results["recommendation_frequency"])
        fraction_vulnerables["userknn_no"].append(results["fraction_vulnerables"])
        privacy_risk["userknn_no"].append(results["avg_privacy_risk"])
        neighborhood_size_q["userknn_no"].append(results["avg_neighborhood_size_q"])
        rating_overlap_q["userknn_no"].append(results["avg_rating_overlap_q"])
        privacy_risk_secures["userknn_no"].append(results["avg_privacy_risk_secures"])
        data_usage["userknn_no"].append(results_samples["avg_data_usage"])
        significance_test_results["userknn_no"].append(evaluation.significance_tests(userknn_results_samples, results_samples))

        # Baseline: UserKNN with DP applied to all users
        models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "sim": sim, "overlap": overlap,
                                                               "rated_items": rated_items, "use_dp": True,
                                                               "thresholds": [0 for _ in range(len(K))]})
        results, userknn_full_results_samples = evaluation.evaluate(models, [models[K_q_idx]])
        mean_absolute_error["userknn_full"].append(results["mean_absolute_error"])
        ndcg["userknn_full"].append(results["avg_ndcg"])
        recommendation_frequency["userknn_full"].append(results["recommendation_frequency"])
        fraction_vulnerables["userknn_full"].append(results["fraction_vulnerables"])
        privacy_risk["userknn_full"].append(results["avg_privacy_risk"])
        neighborhood_size_q["userknn_full"].append(results["avg_neighborhood_size_q"])
        rating_overlap_q["userknn_full"].append(results["avg_rating_overlap_q"])
        privacy_risk_secures["userknn_full"].append(results["avg_privacy_risk_secures"])
        data_usage["userknn_full"].append(userknn_full_results_samples["avg_data_usage"])
        significance_test_results["userknn_full"].append(evaluation.significance_tests(userknn_results_samples, userknn_full_results_samples))
        significance_test_results_full["userknn_full"].append(evaluation.significance_tests(userknn_full_results_samples, userknn_results_samples))

        # UserKNN+Reuse
        models, _ = run(trainset, testset, K=K, configuration={"reuse": True, "sim": sim, "overlap": overlap,
                                                               "rated_items": rated_items, "thresholds": threshs,
                                                               "use_dp": args.use_dp})
        results, results_samples = evaluation.evaluate(models, [models[K_q_idx]])
        mean_absolute_error["userknn_reuse"].append(results["mean_absolute_error"])
        ndcg["userknn_reuse"].append(results["avg_ndcg"])
        recommendation_frequency["userknn_reuse"].append(results["recommendation_frequency"])
        fraction_vulnerables["userknn_reuse"].append(results["fraction_vulnerables"])
        privacy_risk["userknn_reuse"].append(results["avg_privacy_risk"])
        neighborhood_size_q["userknn_reuse"].append(results["avg_neighborhood_size_q"])
        rating_overlap_q["userknn_reuse"].append(results["avg_rating_overlap_q"])
        privacy_risk_secures["userknn_reuse"].append(results["avg_privacy_risk_secures"])
        data_usage["userknn_reuse"].append(results_samples["avg_data_usage"])
        significance_test_results["userknn_reuse"].append(evaluation.significance_tests(userknn_results_samples, results_samples))
        significance_test_results_full["userknn_reuse"].append(evaluation.significance_tests(userknn_full_results_samples, results_samples))

        # Expect
        models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "sim": sim, "use_dp": args.use_dp,
                                                               "expect_scores": expect_scores, "overlap": overlap,
                                                               "rated_items": rated_items, "thresholds": threshs})
        results, results_samples = evaluation.evaluate(models, [models[K_q_idx]])
        mean_absolute_error["expect"].append(results["mean_absolute_error"])
        ndcg["expect"].append(results["avg_ndcg"])
        recommendation_frequency["expect"].append(results["recommendation_frequency"])
        fraction_vulnerables["expect"].append(results["fraction_vulnerables"])
        privacy_risk["expect"].append(results["avg_privacy_risk"])
        neighborhood_size_q["expect"].append(results["avg_neighborhood_size_q"])
        rating_overlap_q["expect"].append(results["avg_rating_overlap_q"])
        privacy_risk_secures["expect"].append(results["avg_privacy_risk_secures"])
        data_usage["expect"].append(results_samples["avg_data_usage"])
        significance_test_results["expect"].append(evaluation.significance_tests(userknn_results_samples, results_samples))
        significance_test_results_full["expect"].append(evaluation.significance_tests(userknn_full_results_samples, results_samples))

        # Gain
        models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "sim": sim, "gain_scores": gain_scores,
                                                               "overlap": overlap, "rated_items": rated_items,
                                                               "thresholds": threshs, "use_dp": args.use_dp})
        results, results_samples = evaluation.evaluate(models, [models[K_q_idx]])
        mean_absolute_error["gain"].append(results["mean_absolute_error"])
        ndcg["gain"].append(results["avg_ndcg"])
        recommendation_frequency["gain"].append(results["recommendation_frequency"])
        fraction_vulnerables["gain"].append(results["fraction_vulnerables"])
        privacy_risk["gain"].append(results["avg_privacy_risk"])
        neighborhood_size_q["gain"].append(results["avg_neighborhood_size_q"])
        rating_overlap_q["gain"].append(results["avg_rating_overlap_q"])
        privacy_risk_secures["gain"].append(results["avg_privacy_risk_secures"])
        data_usage["gain"].append(results_samples["avg_data_usage"])
        significance_test_results["gain"].append(evaluation.significance_tests(userknn_results_samples, results_samples))
        significance_test_results_full["gain"].append(evaluation.significance_tests(userknn_full_results_samples, results_samples))

        n_processed_folds += 1

    # save all results to disk (RESULT_PATH)
    f = open(RESULT_PATH + "/data_usage_distribution.pkl", "wb")
    pl.dump(data_usage, f)
    f.close()

    f = open(RESULT_PATH + "/significance_test_results.pkl", "wb")
    pl.dump(significance_test_results, f)
    f.close()

    f = open(RESULT_PATH + "/significance_test_results_full.pkl", "wb")
    pl.dump(significance_test_results_full, f)
    f.close()

    avg_neighborhood_size_q_userknn = utils.avg_over_q(neighborhood_size_q["userknn"], n_folds=n_processed_folds, n_ks=1)
    avg_neighborhood_size_q_userknn_reuse = utils.avg_over_q(neighborhood_size_q["userknn_reuse"], n_folds=n_processed_folds, n_ks=1)
    avg_neighborhood_size_q_expect = utils.avg_over_q(neighborhood_size_q["expect"], n_folds=n_processed_folds, n_ks=1)
    avg_neighborhood_size_q_gain = utils.avg_over_q(neighborhood_size_q["gain"], n_folds=n_processed_folds, n_ks=1)

    avg_rating_overlap_q_userknn = utils.avg_over_q(rating_overlap_q["userknn"], n_folds=n_processed_folds, n_ks=1)
    avg_rating_overlap_q_userknn_reuse = utils.avg_over_q(rating_overlap_q["userknn_reuse"], n_folds=n_processed_folds, n_ks=1)
    avg_rating_overlap_q_expect = utils.avg_over_q(rating_overlap_q["expect"], n_folds=n_processed_folds, n_ks=1)
    avg_rating_overlap_q_gain = utils.avg_over_q(rating_overlap_q["gain"], n_folds=n_processed_folds, n_ks=1)

    np.save(RESULT_PATH + "/K.npy", K)
    np.save(RESULT_PATH + "/thresholds.npy", np.mean(thresholds, axis=0))

    np.save(RESULT_PATH + "/neighborhood_size_q_userknn.npy", avg_neighborhood_size_q_userknn)
    np.save(RESULT_PATH + "/neighborhood_size_q_userknn_reuse.npy", avg_neighborhood_size_q_userknn_reuse)
    np.save(RESULT_PATH + "/neighborhood_size_q_expect.npy", avg_neighborhood_size_q_expect)
    np.save(RESULT_PATH + "/neighborhood_size_q_gain.npy", avg_neighborhood_size_q_gain)

    np.save(RESULT_PATH + "/rating_overlap_q_userknn.npy", avg_rating_overlap_q_userknn)
    np.save(RESULT_PATH + "/rating_overlap_q_userknn_reuse.npy", avg_rating_overlap_q_userknn_reuse)
    np.save(RESULT_PATH + "/rating_overlap_q_expect.npy", avg_rating_overlap_q_expect)
    np.save(RESULT_PATH + "/rating_overlap_q_gain.npy", avg_rating_overlap_q_gain)

    np.save(RESULT_PATH + "/mae_userknn_no.npy", np.mean(mean_absolute_error["userknn_no"], axis=0))
    np.save(RESULT_PATH + "/mae_userknn_full.npy", np.mean(mean_absolute_error["userknn_full"], axis=0))
    np.save(RESULT_PATH + "/mae_userknn.npy", np.mean(mean_absolute_error["userknn"], axis=0))
    np.save(RESULT_PATH + "/mae_userknn_reuse.npy", np.mean(mean_absolute_error["userknn_reuse"], axis=0))
    np.save(RESULT_PATH + "/mae_expect.npy", np.mean(mean_absolute_error["expect"], axis=0))
    np.save(RESULT_PATH + "/mae_gain.npy", np.mean(mean_absolute_error["gain"], axis=0))

    np.save(RESULT_PATH + "/ndcg_userknn_no.npy", np.mean(ndcg["userknn_no"], axis=0))
    np.save(RESULT_PATH + "/ndcg_userknn_full.npy", np.mean(ndcg["userknn_full"], axis=0))
    np.save(RESULT_PATH + "/ndcg_userknn.npy", np.mean(ndcg["userknn"], axis=0))
    np.save(RESULT_PATH + "/ndcg_userknn_reuse.npy", np.mean(ndcg["userknn_reuse"], axis=0))
    np.save(RESULT_PATH + "/ndcg_expect.npy", np.mean(ndcg["expect"], axis=0))
    np.save(RESULT_PATH + "/ndcg_expect_reuse.npy", np.mean(ndcg["expect_reuse"], axis=0))
    np.save(RESULT_PATH + "/ndcg_gain.npy", np.mean(ndcg["gain"], axis=0))
    np.save(RESULT_PATH + "/ndcg_gain_reuse.npy", np.mean(ndcg["gain_reuse"], axis=0))

    np.save(RESULT_PATH + "/privacy_risk_userknn_no.npy", np.mean(privacy_risk["userknn_no"], axis=0))
    np.save(RESULT_PATH + "/privacy_risk_userknn_full.npy", np.mean(privacy_risk["userknn_full"], axis=0))
    np.save(RESULT_PATH + "/privacy_risk_userknn.npy", np.mean(privacy_risk["userknn"], axis=0))
    np.save(RESULT_PATH + "/privacy_risk_userknn_reuse.npy", np.mean(privacy_risk["userknn_reuse"], axis=0))
    np.save(RESULT_PATH + "/privacy_risk_expect.npy", np.mean(privacy_risk["expect"], axis=0))
    np.save(RESULT_PATH + "/privacy_risk_gain.npy", np.mean(privacy_risk["gain"], axis=0))

    np.save(RESULT_PATH + "/privacy_risk_secures_userknn_no.npy", np.mean(privacy_risk_secures["userknn_no"], axis=0))
    np.save(RESULT_PATH + "/privacy_risk_secures_userknn_full.npy", np.mean(privacy_risk_secures["userknn_full"], axis=0))
    np.save(RESULT_PATH + "/privacy_risk_secures_userknn.npy", np.mean(privacy_risk_secures["userknn"], axis=0))
    np.save(RESULT_PATH + "/privacy_risk_secures_userknn_reuse.npy", np.mean(privacy_risk_secures["userknn_reuse"], axis=0))
    np.save(RESULT_PATH + "/privacy_risk_secures_expect.npy", np.mean(privacy_risk_secures["expect"], axis=0))
    np.save(RESULT_PATH + "/privacy_risk_secures_gain.npy", np.mean(privacy_risk_secures["gain"], axis=0))

    np.save(RESULT_PATH + "/fraction_vulnerables_userknn_no.npy", np.mean(fraction_vulnerables["userknn_no"], axis=0))
    np.save(RESULT_PATH + "/fraction_vulnerables_userknn_full.npy", np.mean(fraction_vulnerables["userknn_full"], axis=0))
    np.save(RESULT_PATH + "/fraction_vulnerables_userknn.npy", np.mean(fraction_vulnerables["userknn"], axis=0))
    np.save(RESULT_PATH + "/fraction_vulnerables_userknn_reuse.npy", np.mean(fraction_vulnerables["userknn_reuse"], axis=0))
    np.save(RESULT_PATH + "/fraction_vulnerables_expect.npy", np.mean(fraction_vulnerables["expect"], axis=0))
    np.save(RESULT_PATH + "/fraction_vulnerables_gain.npy", np.mean(fraction_vulnerables["gain"], axis=0))

    f = open(RESULT_PATH + "/recommendation_frequency_userknn_no.pkl", "wb")
    pl.dump(utils.dict3d_avg(recommendation_frequency["userknn_no"], n_folds=n_processed_folds, K=K), f)
    f = open(RESULT_PATH + "/recommendation_frequency_userknn_full.pkl", "wb")
    pl.dump(utils.dict3d_avg(recommendation_frequency["userknn_full"], n_folds=n_processed_folds, K=K), f)
    f = open(RESULT_PATH + "/recommendation_frequency_userknn.pkl", "wb")
    pl.dump(utils.dict3d_avg(recommendation_frequency["userknn"], n_folds=n_processed_folds, K=K), f)
    f = open(RESULT_PATH + "/recommendation_frequency_userknn_reuse.pkl", "wb")
    pl.dump(utils.dict3d_avg(recommendation_frequency["userknn_reuse"], n_folds=n_processed_folds, K=K), f)
    f = open(RESULT_PATH + "/recommendation_frequency_expect.pkl", "wb")
    pl.dump(utils.dict3d_avg(recommendation_frequency["expect"], n_folds=n_processed_folds, K=K), f)
    f = open(RESULT_PATH + "/recommendation_frequency_gain.pkl", "wb")
    pl.dump(utils.dict3d_avg(recommendation_frequency["gain"], n_folds=n_processed_folds, K=K), f)
    f.close()


