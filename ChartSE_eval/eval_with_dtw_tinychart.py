# Script to evaluate the predictions of the model using the DTW (Dynamic Time Warping) metric.

import json
import numpy as np
from scipy import optimize
import itertools
import dataclasses
from typing import Optional, List, Tuple, Dict, Any
import editdistance


from typing import Callable, Optional, Union
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


Number = Union[int, float]
ArrayLike = Union[np.ndarray, list, tuple]


def _make_dist_fn(
    dist: Union[str, Callable[[np.ndarray, np.ndarray], float]],
) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Build a pointwise distance function that accepts two numpy arrays of the same shape
    (scalars or vectors) and returns a scalar.
    """
    if callable(dist):
        return dist

    name = str(dist).lower()
    if name in ("l1", "manhattan"):
        return lambda a, b: float(np.sum(np.abs(a - b)))
    if name in ("l2", "euclidean"):
        # True Euclidean (with sqrt)
        return lambda a, b: float(np.sqrt(np.sum((a - b) ** 2)))
    if name in ("sqeuclidean", "l2sq", "squared_euclidean"):
        # Squared Euclidean (no sqrt)
        return lambda a, b: float(np.sum((a - b) ** 2))
    raise ValueError(
        "Unknown dist spec. Use a callable or one of: " "'l1', 'l2', 'sqeuclidean'."
    )


def _as_2d_series(x: ArrayLike) -> np.ndarray:
    """
    Ensure the series is a 2D numpy array of shape (m, d):
    - 1D sequences become (m, 1)
    - 2D sequences (m, d) are kept as is
    """
    x = np.asarray(x)
    if x.ndim == 0:
        # Single scalar -> treat as length-1 series with 1 feature
        return x.reshape(1, 1)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim == 2:
        return x
    raise ValueError("Input series must be 1D or 2D (got shape {}).".format(x.shape))


def erp_distance(
    x: ArrayLike,
    y: ArrayLike,
    g: Union[Number, ArrayLike] = 0.0,
    window: Optional[int] = None,
    dist: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "l1",
    visualize: bool = True,
    return_ops: bool = False,
):
    """
    Compute ERP between two time series and (optionally) visualize the chosen path.

    ERP aligns using 3 operations:
      - match:      cost = dist(x_i, y_j)
      - deletion:   cost = dist(x_i, g)   (gap in Y)
      - insertion:  cost = dist(y_j, g)   (gap in X)

    Parameters
    ----------
    x, y : array-like
        1D (m,) / (n,) or 2D (m,d) / (n,d).
    g : float or array-like
        Scalar or 1D vector of length d.
    window : int or None
        Sakoe–Chiba window. Ensures |i - j| <= window; auto-widens to |m-n|.
    dist : {"l1","l2","sqeuclidean"} or callable
        Local point distance.
    visualize : bool, default True
        If True, draw a heatmap of C[1:,1:] and overlay the optimal path with
        colored markers: match(●), deletion(■), insertion(▲).
    return_ops : bool, default False
        If True, also return a list of step dictionaries with detailed info.

    Returns
    -------
    total : float
        ERP total cost.
    path_len : float
        Length (number of steps) of the chosen path.
    ops : list of dict (only if return_ops=True)
        Forward-ordered alignment steps with keys:
        {"op": "M"/"D"/"I", "i": int or None, "j": int or None, "cost": float}
        where i/j are zero-based indices into x/y for that step, or None for gaps.
    """
    # -- Prepare inputs --
    X, Y = _as_2d_series(x), _as_2d_series(y)
    m, dx = X.shape
    n, dy = Y.shape
    if dx != dy:
        raise ValueError(f"d mismatch: {dx} vs {dy}")

    dist_fn = _make_dist_fn(dist)
    gap_cost = 1.0  # Fixed cost for a gap

    # -- DP state: cost (C), path length (L), backpointers (PTR) --
    # PTR codes: 0 = match (↖), 1 = deletion (↑), 2 = insertion (←), 255 = unset
    C = np.full((m + 1, n + 1), np.inf)
    L = np.full((m + 1, n + 1), np.inf)
    PTR = np.full((m + 1, n + 1), 255, dtype=np.uint8)

    C[0, 0] = 0.0
    L[0, 0] = 0.0

    # Base row/col (aligning prefixes to gaps)
    for i in range(1, m + 1):
        C[i, 0] = C[i - 1, 0] + gap_cost
        L[i, 0] = L[i - 1, 0] + 1
        PTR[i, 0] = 1  # deletion
    for j in range(1, n + 1):
        C[0, j] = C[0, j - 1] + gap_cost
        L[0, j] = L[0, j - 1] + 1
        PTR[0, j] = 2  # insertion

    # Window handling
    if window is not None:
        if window < 0:
            raise ValueError("window >= 0 or None")
        if window < abs(m - n):
            window = abs(m - n)

    # Main DP
    for i in range(1, m + 1):
        j_lo, j_hi = 1, n
        if window is not None:
            j_lo = max(1, i - window)
            j_hi = min(n, i + window)

        for j in range(j_lo, j_hi + 1):
            # Candidates: match (↖), deletion (↑), insertion (←)
            cand = [
                (C[i - 1, j - 1] + dist_fn(X[i - 1], Y[j - 1]), L[i - 1, j - 1] + 1, 0),
                (C[i - 1, j] + gap_cost, L[i - 1, j] + 1, 1),
                (C[i, j - 1] + gap_cost, L[i, j - 1] + 1, 2),
            ]
            min_cost = min(c[0] for c in cand)
            tied = [c for c in cand if np.isclose(c[0], min_cost)]
            # tie-break: shortest path, then prefer order match > deletion > insertion
            best = min(tied, key=lambda x: (x[1], x[2]))

            C[i, j] = float(best[0])
            L[i, j] = float(best[1])
            PTR[i, j] = best[2]

    total = float(C[m, n])
    path_len = float(L[m, n])

    # -- Backtrack optimal path of DP states (from (m,n) to (0,0)) --
    # We collect states (i,j) and turn them into forward-ordered steps.
    states: List[Tuple[int, int]] = []
    i, j = m, n
    while i > 0 or j > 0:
        states.append((i, j))
        op = PTR[i, j]
        if op == 0:  # match: came from (i-1, j-1)
            i, j = i - 1, j - 1
        elif op == 1:  # deletion: came from (i-1, j)
            i = i - 1
        elif op == 2:  # insertion: came from (i, j-1)
            j = j - 1
        else:
            # Shouldn't happen if a path exists within the window
            raise RuntimeError(f"Backpointer missing at {(i, j)}; window too tight?")
    states.append((0, 0))
    states.reverse()  # now forward from (0,0) -> (m,n)

    # -- Build forward-ordered op list with local costs and (x_i, y_j) indices --
    # Step k goes from states[k-1] -> states[k]. The target indices are:
    #  match:     (i, j) -> op 'M' uses x[i-1] & y[j-1]
    #  deletion:  (i, j) with delta (1,0) -> op 'D' uses x[i-1], gap
    #  insertion: (i, j) with delta (0,1) -> op 'I' uses gap, y[j-1]
    ops: List[Dict] = []
    for k in range(1, len(states)):
        i_prev, j_prev = states[k - 1]
        i_cur, j_cur = states[k]
        di, dj = i_cur - i_prev, j_cur - j_prev
        if di == 1 and dj == 1:
            cost = dist_fn(X[i_cur - 1], Y[j_cur - 1])
            ops.append({"op": "M", "i": i_cur - 1, "j": j_cur - 1, "cost": float(cost)})
        elif di == 1 and dj == 0:
            cost = gap_cost
            ops.append({"op": "D", "i": i_cur - 1, "j": None, "cost": float(cost)})
        elif di == 0 and dj == 1:
            cost = gap_cost
            ops.append({"op": "I", "i": None, "j": j_cur - 1, "cost": float(cost)})
        else:
            # Should never occur in a proper monotone path
            raise RuntimeError(f"Non-monotone step {states[k-1]} -> {states[k]}")

    # -- Optional visualization --
    if visualize:
        # Print a concise step log to console for debugging, including actual values
        print("=== ERP alignment steps ===")
        for idx, s in enumerate(ops, 1):
            if s["op"] == "M":
                x_val = X[s["i"]] if s["i"] is not None else None
                y_val = Y[s["j"]] if s["j"] is not None else None
                print(
                    f"{idx:3d}. M  x[{s['i']}] ↔ y[{s['j']}],  cost = {s['cost']:.6f} | x = {x_val}, y = {y_val}"
                )
            elif s["op"] == "D":
                x_val = X[s["i"]] if s["i"] is not None else None
                print(
                    f"{idx:3d}. D  x[{s['i']}] ↔ GAP,     cost = {s['cost']:.6f} | x = {x_val}, y = GAP"
                )
            else:
                y_val = Y[s["j"]] if s["j"] is not None else None
                print(
                    f"{idx:3d}. I  GAP ↔ y[{s['j']}],     cost = {s['cost']:.6f} | x = GAP, y = {y_val}"
                )

    if return_ops:
        return total, path_len, ops
    return total, path_len


def _get_relative_distance(target, prediction, theta=0.1):
    """Returns min(1, |target-prediction|/|target|)."""
    if target == 0:
        return 0 if prediction == 0 else 1
    distance = min(abs(target - prediction) / target, 1)
    return distance if distance < theta else 1


def _get_distance_axis_normalized(
    target, prediction, series_axis_max, series_axis_min, theta=0.1
):
    """Returns min(1, |target-prediction|/|series_axis_max - series_axis_min|)."""
    distance = min(abs(target - prediction) / (series_axis_max - series_axis_min), 1)
    return distance if distance < theta else 1


def _get_distance_series_normalized(
    target, prediction, series_max, series_min, theta=0.1
):
    """Returns min(1, |target-prediction|/|series_max - series_min|)."""
    if series_max == series_min:
        return 0 if prediction == series_max else 1
    distance = min(abs(target - prediction) / (series_max - series_min), 1)
    return distance if distance < theta else 1


def anls_metric(target: str, prediction: str, theta: float = 0.5):
    """Calculates ANLS for DocVQA.

    There does not seem to be an official evaluation script.
    Public implementation on which this implementation is based:
    https://github.com/herobd/layoutlmv2/blob/main/eval_docvqa.py#L92

    Original paper (see Eq 1): https://arxiv.org/pdf/1907.00490.pdf

    Args:
      target: Target string.
      prediction: Predicted string.
      theta: Filter threshold set to 0.5 for DocVQA.

    Returns:
      ANLS score.
    """

    edit_distance = editdistance.eval(target, prediction)
    normalized_ld = edit_distance / max(len(target), len(prediction))
    return 1 - normalized_ld if normalized_ld < theta else 0


def _to_float(text):
    """Convert text to float, handling percentages."""
    try:
        if text.endswith("%"):
            # Convert percentages to floats.
            return float(text.rstrip("%"))
        else:
            return float(text)
    except ValueError:
        return 0


def _dtw_similarity(
    series1: List[float],
    series2: List[float],
    series_axis_max: float,
    series_axis_min: float,
) -> float:
    """Calculate DTW similarity between two time series using the formula:
    DTW_score = 1 - (DTW(P, T) / max(DTW(P, P), DTW(T, T)))

    Args:
        series1: First time series as list of floats
        series2: Second time series as list of floats

    Returns:
        Similarity score (higher is better, 0-1 range)
    """
    if not series1 or not series2:
        return 0.0

    s1 = np.array(series1)
    s2 = np.array(series2)

    series_max = max(series1)
    series_min = min(series1)

    dist_fn_relative = lambda a, b: _get_relative_distance(a, b, theta=0.1)
    dist_fn_axis_normalized = lambda a, b: _get_distance_axis_normalized(
        a, b, series_axis_max, series_axis_min, theta=0.1
    )
    dist_fn_series_normalized = lambda a, b: _get_distance_series_normalized(
        a, b, series_max, series_min, theta=0.1
    )

    dist_relative, path_len_relative = erp_distance(
        s1, s2, g=0.0, window=7, dist=dist_fn_relative, visualize=True
    )
    dist_axis_normalized, path_len_axis_normalized = erp_distance(
        s1, s2, g=0.0, window=7, dist=dist_fn_axis_normalized, visualize=True
    )
    dist_series_normalized, path_len_series_normalized = erp_distance(
        s1, s2, g=0.0, window=7, dist=dist_fn_series_normalized, visualize=True
    )

    score_relative = 1.0 - dist_relative / path_len_relative
    score_axis_normalized = 1.0 - dist_axis_normalized / path_len_axis_normalized
    score_series_normalized = 1.0 - dist_series_normalized / path_len_series_normalized
    print(f"Series DTW Similarity Score (relative): {score_relative * 100:.2f}%")
    print(
        f"Series DTW Similarity Score (axis normalized): {score_axis_normalized * 100:.2f}%"
    )
    print(
        f"Series DTW Similarity Score (series normalized): {score_series_normalized * 100:.2f}%"
    )

    return score_relative, score_axis_normalized, score_series_normalized


def _extract_series_from_table(table) -> Dict[str, List[float]]:
    """Extract time series from a table by matching series labels.

    Args:
        table: Table object with headers and rows

    Returns:
        Dictionary mapping series labels to their values
    """
    series_dict = {}

    if not table.rows or len(table.headers) <= 1:
        return series_dict

    # First column contains the x-axis labels (time points)
    # Remaining columns contain different series
    for col_idx, header in enumerate(table.headers[1:], 1):
        series_values = []
        for row in table.rows:
            if col_idx < len(row):
                value = _to_float(row[col_idx])
                if value is not None:
                    series_values.append(value)

        if series_values:  # Only add non-empty series
            series_dict[header] = series_values

    return series_dict


def _match_series_by_label(
    target_series: Dict[str, List[float]],
    pred_series: Dict[str, List[float]],
    text_theta: float = 0.5,
) -> List[Tuple[str, str, float]]:
    """Match series between target and prediction by label similarity.

    Args:
        target_series: Dictionary of target series
        pred_series: Dictionary of predicted series
        text_theta: Threshold for label matching

    Returns:
        List of tuples (target_label, pred_label, similarity_score)
    """
    if not target_series or not pred_series:
        return []

    # Create cost matrix for label matching
    target_labels = list(target_series.keys())
    pred_labels = list(pred_series.keys())

    # if one of GT/pred only has 1 series and the other has >1, then label matching will not work because the one with 1 series has no labels ==> match first series in that case
    if len(target_labels) == 1 or len(pred_labels) == 1:
        return [(target_labels[0], pred_labels[0], 1.0)]

    distance_matrix = []
    for target_label in target_labels:
        row = []
        for pred_label in pred_labels:
            # Calculate label similarity using ANLS
            similarity = anls_metric(target_label, pred_label, text_theta)
            distance = 1 - similarity
            row.append(distance)
        distance_matrix.append(row)

    # Use Hungarian algorithm to find optimal matching
    cost_matrix = np.array(distance_matrix)
    row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)

    matches = []
    for r, c in zip(row_ind, col_ind):
        target_label = target_labels[r]
        pred_label = pred_labels[c]
        similarity = 1 - cost_matrix[r, c]
        matches.append((target_label, pred_label, similarity))

    print(f"Matched series: {matches}")
    return matches


def _table_dtw_similarity(target_table, prediction_table, axes_bounds, text_theta=0.5):
    """Calculate DTW-based similarity between two tables.

    Args:
        target_table: Target table
        prediction_table: Prediction table
        text_theta: Threshold for label matching

    Returns:
        DTW similarity score (higher is better)
    """
    # Extract series from both tables
    target_series = _extract_series_from_table(target_table)
    pred_series = _extract_series_from_table(prediction_table)

    if not target_series and not pred_series:
        return 1.0, 1.0, 1.0
    if not target_series or not pred_series:
        return 0.0, 0.0, 0.0

    # Match series by label
    matches = _match_series_by_label(target_series, pred_series, text_theta)
    if not matches:
        return 0.0, 0.0, 0.0

    # Calculate DTW similarity for each matched pair (no weighting)
    matched_similarities_relative = []
    matched_similarities_axis_normalized = []
    matched_similarities_series_normalized = []
    matched_target_labels = set()

    for target_label, pred_label, label_similarity in matches:
        if label_similarity > 0:  # Only consider matches above threshold
            target_values = target_series[target_label]
            pred_values = pred_series[pred_label]
            series_axis_min, series_axis_max = axes_bounds[target_label]

            # Calculate DTW similarity
            dtw_sim_relative, dtw_sim_axis_normalized, dtw_sim_series_normalized = (
                _dtw_similarity(
                    target_values, pred_values, series_axis_max, series_axis_min
                )
            )
            matched_similarities_relative.append(dtw_sim_relative)
            matched_similarities_axis_normalized.append(dtw_sim_axis_normalized)
            matched_similarities_series_normalized.append(dtw_sim_series_normalized)
            matched_target_labels.add(target_label)

    # Count unmatched ground truth series as 0
    unmatched_count = len(target_series) - len(matched_target_labels)

    # Add zeros for unmatched ground truth series
    all_similarities = matched_similarities_relative + [0.0] * unmatched_count
    all_similarities_axis_normalized = (
        matched_similarities_axis_normalized + [0.0] * unmatched_count
    )
    all_similarities_series_normalized = (
        matched_similarities_series_normalized + [0.0] * unmatched_count
    )

    assert (
        len(all_similarities)
        == len(all_similarities_axis_normalized)
        == len(all_similarities_series_normalized)
    )

    if not all_similarities:
        return 0.0, 0.0, 0.0

    return (
        sum(all_similarities) / len(all_similarities),
        sum(all_similarities_axis_normalized) / len(all_similarities_axis_normalized),
        sum(all_similarities_series_normalized)
        / len(all_similarities_series_normalized),
    )


@dataclasses.dataclass(frozen=True)
class Table:
    """Helper class for the content of a markdown table."""

    title: Optional[str] = None
    headers: tuple[str, Ellipsis] = dataclasses.field(default_factory=tuple)
    rows: tuple[tuple[str, Ellipsis], Ellipsis] = dataclasses.field(
        default_factory=tuple
    )

    def permuted(self, indexes):
        """Builds a version of the table changing the column order."""
        return Table(
            title=self.title,
            headers=_permute(self.headers, indexes),
            rows=tuple(_permute(row, indexes) for row in self.rows),
        )

    def aligned(self, headers, text_theta=0.5):
        """Builds a column permutation with headers in the most correct order."""
        if len(headers) != len(self.headers):
            raise ValueError(f"Header length {headers} must match {self.headers}.")
        distance = []
        for h2 in self.headers:
            distance.append([1 - anls_metric(h1, h2, text_theta) for h1 in headers])
        cost_matrix = np.array(distance)
        row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
        permutation = [idx for _, idx in sorted(zip(col_ind, row_ind))]
        score = (1 - cost_matrix)[permutation[1:], range(1, len(row_ind))].prod()
        return self.permuted(permutation), score


def _permute(values, indexes):
    return tuple(values[i] if i < len(values) else "" for i in indexes)


def _parse_table(text, target_has_more_than_two_headers=None):
    """Builds a table from a markdown representation."""
    lines = text.lower().splitlines()
    if not lines:
        return Table()
    if lines[0].startswith("title |"):
        title = lines[0][len("title |") :].strip()
        offset = 1
    else:
        title = None
        offset = 0
    if len(lines) < offset + 1:
        return Table(title=title)
    rows = []
    for line in lines[offset:]:
        rows.append(tuple(v.strip() for v in line.split("|")))

    if len(rows[0]) == 2 and not target_has_more_than_two_headers:
        return Table(title=title, headers=("A", "B"), rows=tuple(rows))
    else:
        return Table(title=title, headers=rows[0], rows=tuple(rows[1:]))


def table_dtw_similarity_per_point(
    targets,
    predictions,
    reversed_predictions,
    axes_bounds,
    text_theta=0.5,
):
    """Computes DTW similarity metrics given two flattened tables.

    Parses each string into a table and extracts time series.
    Matches series by label similarity and calculates DTW similarity.

    Args:
      targets: list of list of strings.
      predictions: list of strings.
      text_theta: relative edit distance above this is set to the maximum of 1.

    Returns:
      Dictionary with per-point DTW similarity scores
    """
    assert len(targets) == len(predictions)
    per_point_scores = {
        "dtw_similarity_relative": [],
        "dtw_similarity_axis_normalized": [],
        "dtw_similarity_series_normalized": [],
    }

    for i, (pred, target) in enumerate(zip(predictions, targets)):
        print(f"Processing item {i+1} of {len(predictions)}")
        target_table = _parse_table(target, None)
        target_has_more_than_two_headers = len(target_table.headers) > 2
        pred_table = _parse_table(pred, target_has_more_than_two_headers)
        reversed_pred = reversed_predictions[i]
        reversed_pred_table = _parse_table(
            reversed_pred, target_has_more_than_two_headers
        )
        # Try all target variations
        (
            similarity_relative,
            similarity_axis_normalized,
            similarity_series_normalized,
        ) = _table_dtw_similarity(
            target_table,
            pred_table,
            axes_bounds[i],
            text_theta,
        )

        (
            reversed_similarity_relative,
            reversed_similarity_axis_normalized,
            reversed_similarity_series_normalized,
        ) = _table_dtw_similarity(
            target_table,
            reversed_pred_table,
            axes_bounds[i],
            text_theta,
        )

        # Return simple average (no weighting)
        print(
            f"Image DTW Similarity Score (relative): {100 * max(similarity_relative, reversed_similarity_relative):.2f}%\n"
            f"Image DTW Similarity Score (axis normalized): {100 * max(similarity_axis_normalized, reversed_similarity_axis_normalized):.2f}%\n"
            f"Image DTW Similarity Score (series normalized): {100 * max(similarity_series_normalized, reversed_similarity_series_normalized):.2f}%"
        )
        per_point_scores["dtw_similarity_relative"].append(
            max(similarity_relative, reversed_similarity_relative)
        )
        per_point_scores["dtw_similarity_axis_normalized"].append(
            max(similarity_axis_normalized, reversed_similarity_axis_normalized)
        )
        per_point_scores["dtw_similarity_series_normalized"].append(
            max(similarity_series_normalized, reversed_similarity_series_normalized)
        )
    return per_point_scores


def table_dtw_similarity(
    targets,
    predictions,
    reversed_predictions,
    axes_bounds,
    text_theta=0.5,
):
    """Aggregated version of table_dtw_similarity_per_point().

    Same as table_dtw_similarity_per_point() but returning aggregated
    scores instead of per-point scores.

    Args:
      targets: list of list of strings.
      predictions: list of strings.
      text_theta: relative edit distance above this is set to the maximum of 1.

    Returns:
      Dictionary with aggregated DTW similarity
    """
    score_dict = table_dtw_similarity_per_point(
        targets, predictions, reversed_predictions, axes_bounds, text_theta
    )
    return {
        "table_dtw_similarity": (
            100.0 * sum(score_dict["dtw_similarity_relative"]) / len(targets)
        ),
        "table_dtw_similarity_axis_normalized": (
            100.0 * sum(score_dict["dtw_similarity_axis_normalized"]) / len(targets)
        ),
        "table_dtw_similarity_series_normalized": (
            100.0 * sum(score_dict["dtw_similarity_series_normalized"]) / len(targets)
        ),
    }


def chart2table_dtw_evaluator(data):
    """Evaluate chart data using DTW similarity.

    Args:
        data: List of dictionaries with 'gt_answer' and 'model_answer' keys

    Returns:
        DTW similarity score (higher is better)
    """
    refs = []
    hyps = []
    reversed_hyps = []
    axes_bounds = []
    for item in data:
        ref = "title |\n" + item["gt_answer"].strip().lower()
        refs.append(ref)
        hyp = "title |\n" + item["model_answer"].strip().lower()
        hyps.append(hyp)
        reversed_hyp = "title |\n" + item["model_answer_reversed"].strip().lower()
        reversed_hyps.append(reversed_hyp)
        axes_bounds.append(
            {
                k.lower() if len(item["axes_bounds"]) > 1 else "B": v
                for k, v in item["axes_bounds"].items()
            }
        )

    dtw_similarity = table_dtw_similarity(refs, hyps, reversed_hyps, axes_bounds)
    return (
        dtw_similarity["table_dtw_similarity"],
        dtw_similarity["table_dtw_similarity_axis_normalized"],
        dtw_similarity["table_dtw_similarity_series_normalized"],
    )


# Example usage
if __name__ == "__main__":
    # Load data
    data = json.load(
        open("data/EpiCurveBench_processed/tinychart_adv_preds_gt_analysis.json")
    )

    # Calculate DTW similarity
    dtw_score = chart2table_dtw_evaluator(data)
    print(f"Overall DTW Similarity Score (relative): {dtw_score[0]:.2f}%")
    print(f"Overall DTW Similarity Score (axis normalized): {dtw_score[1]:.2f}%")
    print(f"Overall DTW Similarity Score (series normalized): {dtw_score[2]:.2f}%")
