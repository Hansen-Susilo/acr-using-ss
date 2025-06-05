import numpy as np

def fix_overlapping_intervals(intervals, labels, epsilon=1e-6):
    fixed_intervals = []
    fixed_labels = []
    prev_end = 0.0

    for (start, end), label in zip(intervals, labels):
        # Ensure start is not before prev_end
        if start < prev_end:
            start = prev_end

        # Avoid zero-duration intervals
        if end - start < epsilon:
            continue

        fixed_intervals.append([start, end])
        fixed_labels.append(label)
        prev_end = end

    return np.array(fixed_intervals), fixed_labels