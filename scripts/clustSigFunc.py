import numpy as np, pandas as pd, scipy, ray
from sklearn.cluster import KMeans
try: 
    import cupy as cp, cupyx
except ImportError as impErr:
    print("[Error]: {}.".format(impErr.args[0]))


def getEv_cp(data, covariance_method):
    """
    Get eigenvalues from data.
    """

    nRows, nCols = data.shape

    data = cp.asarray(data)
    eigenvalues, eigenvectors = cp.linalg.eigh(cp.cov(data.T))
    assert len(eigenvalues) == nCols

    if covariance_method == 'sample_covariance':
        eigenvalues = cp.asnumpy(eigenvalues)
        del data, eigenvectors
        gpuClear()
        return eigenvalues

    elif covariance_method == 'soft_thresholding':
        sig2b = est_bg_noise_cp(data)
        eigenvalues = cp.asnumpy(eigenvalues)
        sig2b = cp.asnumpy(sig2b)
        soft_thresholded_eigenvalues = soft_threshold_hanwen_huang(eigenvalues, sig2b)
        del data, eigenvalues, eigenvectors
        gpuClear()
        return soft_thresholded_eigenvalues

    else:
        raise ValueError("Covariance method must be 'sample_covariance' or 'soft_thresholding'")


def est_bg_noise_cp(data):
    data_median = cp.median(data)
    MAD = cp.median(cp.abs(data-data_median))
    scaled_MAD = MAD / scipy.stats.norm.ppf(.75)
    sig2b = scaled_MAD**2
    return sig2b


def sim_ci_cp(i):
    simulated_matrix = cp.random.standard_normal(size=(n, d)) * cp.sqrt(eigenvalues)  
    kmeans = KMeans(n_clusters=2, n_init='auto')
    kmeans.fit(simulated_matrix)
    total_sum_squares = compute_sum_of_square_distances_to_mean(simulated_matrix)
    cluster_index = kmeans.inertia_ / total_sum_squares
    return cluster_index


@ray.remote
def simulate_cluster_indexPar(i, nRows, nCols, eigenvalues):
    """
    Create simulated Gaussian distribution based on data from 2 clusters.
    Try to separate the clusters and calculate cluster index for p value.
    """
    simulated_matrix = np.random.standard_normal(size=(nRows, nCols)) * np.sqrt(eigenvalues)  
    
    kmeans = KMeans(n_clusters=2, n_init='auto')
    kmeans.fit(simulated_matrix)
    total_sum_squares = compute_sum_of_square_distances_to_mean(simulated_matrix)

    cluster_index = kmeans.inertia_ / total_sum_squares

    print('{} done'.format(i))

    return cluster_index


def compute_cluster_index(data, labels):
    """
    Compute the cluster index for the two-class clustering
    given by `labels`.
    """
    class_1, class_2 = split_data(data, labels)
    ci = compute_cluster_index_given_classes(class_1, class_2)
    return ci


def compute_cluster_index_given_classes(class_1, class_2):
    """
    Compute the cluster index for two given classes.
    """
    class_1_sum_squares = compute_sum_of_square_distances_to_mean(class_1)
    class_2_sum_squares = compute_sum_of_square_distances_to_mean(class_2)
    total_sum_squares = compute_sum_of_square_distances_to_mean(np.concatenate([class_1, class_2]))

    cluster_index = (class_1_sum_squares + class_2_sum_squares) / total_sum_squares
    return cluster_index


def compute_sum_of_square_distances_to_mean(X):
    mean = np.mean(X, axis=0)
    return compute_sum_of_square_distances_to_point(X, mean)


def compute_sum_of_square_distances_to_point(X, y):
    displacements = X - y
    distances = np.linalg.norm(displacements, axis=1)
    return np.sum(distances**2)


def soft_threshold_hanwen_huang(eigenvalues, sig2b):
    "Soft threshold eigenvalues to background noise level sig2b according to Hanwen Huang's scheme"
    optimal_tau = _compute_tau(eigenvalues, sig2b)
    soft_thresholded_eigenvalues = _shift_and_threshold_eigenvalues(eigenvalues, optimal_tau, sig2b)
    return soft_thresholded_eigenvalues


def _compute_tau(eigenvalues, sig2b):
    """Compute the tau that gives Hanwen Huang's soft thresholded eigenvalues, which
    maximizes the relative size of the largest eigenvalue"""

    # NOTE: tau is found by searching between 0 and Ming Yuan's tilde_tau.
    tilde_tau = _compute_tilde_tau(eigenvalues, sig2b)
    tau_candidates = np.linspace(0, tilde_tau, 100, endpoint=False)  # using endpoint=False to match Matlab behavior

    criteria = [_relative_size_of_largest_eigenvalue(
                   _shift_and_threshold_eigenvalues(eigenvalues, tau, sig2b)
                ) for tau in tau_candidates]

    optimal_tau = tau_candidates[np.argmax(criteria)]
    return optimal_tau


def _compute_tilde_tau(eigenvalues, sig2b):
    """Computes tilde_tau, the value of tau that gives Ming Yuan's soft
    thresholded eigenvalues, which maintain total power"""
    # NOTE: we compute Ming Yuan's soft thresholding estimates iteratively
    # and then back out what tilde_tau was.
    thresholded_eigenvalues = soft_threshold_ming_yuan(eigenvalues, sig2b)
    tilde_tau = max(0, eigenvalues.max() - thresholded_eigenvalues.max())
    return tilde_tau


def soft_threshold_ming_yuan(eigenvalues, sig2b):
    """Soft thresholds eigenvalues to background noise level sig2b using Ming Yuan's
    scheme, which maintains total power. Results in an anti-conservative SigClust
    when the relative size of the first eigenvalue is small."""

    # Starting with the smallest eigenvalue, sequentially bring eigenvalues up to
    # sig2b and distribute the difference equally over the larger eigenvalues
    # (which maintains the total power).
    d = len(eigenvalues)
    eigenvalues_asc = np.sort(eigenvalues)  # produces a copy
    for i in range(d-1):
        lambda_ = eigenvalues_asc[i]
        if lambda_ < sig2b:
            eigenvalues_asc[i] += (sig2b-lambda_)
            eigenvalues_asc[i+1:] -= (sig2b-lambda_) / (d-i-1)
        else:
            break

    # If this process has brought the largest eigenvalue below sig2b, then it is
    # impossible to threshold to sig2b while maintaining total power. In this
    # case the need to threshold to sig2b overrides the need to maintain total
    # power.
    eigenvalues_asc[d-1] = np.maximum(eigenvalues_asc[d-1], sig2b)

    thresholded_eigenvalues_desc = eigenvalues_asc[::-1]  # reverses order
    return thresholded_eigenvalues_desc


def _shift_and_threshold_eigenvalues(eigenvalues, tau, sig2b):
    """Decrease the eigenvalues by the given tau, and threshold them at sig2b"""
    shifted_eigenvalues = eigenvalues - tau
    return np.maximum(shifted_eigenvalues, sig2b)


def _relative_size_of_largest_eigenvalue(eigenvalues):
    return eigenvalues.max() / eigenvalues.sum()


def split_data(data, labels):
    labels = np.array(labels)

    if len(labels) != data.shape[0]:
        raise ValueError("Number of labels must match number of observations")

    if np.any(pd.isna(labels)):
        raise ValueError("Labels must not contain nan or None")

    if len(np.unique(labels)) != 2:
        raise ValueError("Labels must have exactly 2 unique members")

    class_names = np.unique(labels)

    X1 = data[labels==class_names[0]]
    X2 = data[labels==class_names[1]]

    return (X1, X2)


def gpuClear():
    try: cp._default_memory_pool.free_all_blocks()
    except: pass
