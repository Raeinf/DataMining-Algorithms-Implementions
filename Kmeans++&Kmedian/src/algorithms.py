import numpy as np
import pandas as pd


def run_kmeans_multiple_times(data,k, num_of_iterations = 10):
    result = k_means(data, k)
    purity = calculate_purity(data, result[0])
    for _ in range(num_of_iterations - 1):
        new_result = k_means(data, k)
        new_purity = calculate_purity(data, result[0])
        if new_purity > purity:
            result = new_result
            purity = new_purity

    return result,purity

def run_kmedian_multiple_times(data,k, num_of_iterations = 10):
    result = k_median(data, k)
    purity = calculate_purity(data, result[0])
    for _ in range(num_of_iterations - 1):
        new_result = k_median(data, k)
        new_purity = calculate_purity(data, result[0])
        if new_purity > purity:
            result = new_result
            purity = new_purity

    return result,purity


def calculate_purity(df_true, df_pred):
    true_labels = df_true.iloc[:, -1]
    pred_labels = df_pred.iloc[:, -1]

    contingency_table = pd.crosstab(true_labels, pred_labels)

    dominant_counts = contingency_table.max(axis=0)

    purity = dominant_counts.sum() / df_true.shape[0]

    return purity

def has_converged(old_centroids, new_centroids, threshold=0.00001):
    old_centroids_array = old_centroids.to_numpy()
    new_centroids_array = new_centroids.to_numpy()
    max_diff = np.max(np.abs(old_centroids_array - new_centroids_array))
    return max_diff < threshold

def compute_new_centroids_mean(labeled_data: pd.DataFrame) -> pd.DataFrame:
    return labeled_data.groupby('cluster').mean().reset_index(drop=True)

def compute_new_centroids_median(labeled_data: pd.DataFrame) -> pd.DataFrame:
    return labeled_data.groupby('cluster').median().reset_index(drop=True)

def calculate_sse_and_label(data: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    data_array = data.to_numpy()
    centroids_array = centroids.to_numpy()
    distances = np.sum((data_array[:, np.newaxis, :] - centroids_array[np.newaxis, :, :]) ** 2, axis=2)
    closest_centroids = np.argmin(distances, axis=1)
    data['cluster'] = closest_centroids
    return data

def centroid_calculator_plus(data: pd.DataFrame, k: int) -> pd.DataFrame:
    if k <= 0 or k > len(data):
        raise ValueError("Invalid number of centroids (k)")

    result = data.sample(n=1).copy()
    remaining_data = data.drop(result.index)

    for _ in range(k - 1):
        distances = np.inf * np.ones(len(remaining_data))
        result_array = result.to_numpy()
        remaining_data_array = remaining_data.to_numpy()

        for centroid in result_array:
            current_distances = np.sum((remaining_data_array - centroid) ** 2, axis=1)
            distances = np.minimum(distances, current_distances)

        new_centroid_idx = np.argmax(distances)
        new_centroid = remaining_data.iloc[[new_centroid_idx]]
        result = pd.concat([result, new_centroid], ignore_index=True)
        remaining_data = remaining_data.drop(new_centroid.index)

    return result

def k_means(data: pd.DataFrame, k: int,centroids=None, threshold=0.00001, max_iterations=1000, lable = True):
    data = data.copy()
    if lable:
        data = data.drop(columns=[data.columns[-1]])
    if centroids is None:
        centroids = centroid_calculator_plus(data, k)
    labeled_data = calculate_sse_and_label(data, centroids)

    for iter in range(max_iterations):
        new_centroids = compute_new_centroids_mean(labeled_data)
        labeled_data = labeled_data.drop(columns=["cluster"])
        labeled_data = calculate_sse_and_label(labeled_data, new_centroids)

        if has_converged(centroids, new_centroids, threshold):
            return labeled_data, new_centroids, iter

        centroids = new_centroids

    return labeled_data, centroids, max_iterations

def k_median(data: pd.DataFrame, k: int, centroids=None, threshold=0.00001, max_iterations=1000, lable = True):
    data = data.copy()
    if lable:
        data = data.drop(columns=[data.columns[-1]])
    if centroids is None:
        centroids = centroid_calculator_plus(data, k)
    labeled_data = calculate_sse_and_label(data, centroids)

    for iter in range(max_iterations):
        new_centroids = compute_new_centroids_median(labeled_data)
        labeled_data = labeled_data.drop(columns=["cluster"])
        labeled_data = calculate_sse_and_label(labeled_data, new_centroids)

        if has_converged(centroids, new_centroids, threshold):
            return labeled_data, new_centroids, iter

        centroids = new_centroids

    return labeled_data, centroids, max_iterations

