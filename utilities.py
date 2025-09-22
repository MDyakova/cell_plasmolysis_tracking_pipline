""""
Functions for track.py code
"""

import numpy as np
from collections import defaultdict
from itertools import combinations

# Function to join mask pairs
def join_pairs(pairs):
    # Create a dictionary to map each element to its group
    groups = defaultdict(set)

    # Iterate over pairs and add them to the corresponding group
    for pair in pairs:
        a, b = pair
        # Merge the groups of a and b
        group_a = groups[a]
        group_b = groups[b]
        combined_group = group_a | group_b | {a, b}
        # Update both a and b's groups with the combined group
        for elem in combined_group:
            groups[elem] = combined_group

    # Remove duplicates and return the result
    unique_groups = set(frozenset(group) for group in groups.values())
    return [list(group) for group in unique_groups]

def find_similar_contours_fast(image_for_masks):
    mask_id, numbers = np.unique(image_for_masks.reshape(-1), return_counts=True)
    mask_numbers = dict(zip(mask_id, numbers))
    groups_short = [image_for_masks[i, j, :] for i in range(image_for_masks.shape[0]) for j in range(image_for_masks.shape[1])]
    groups_short = [sorted(list(set(group)))[1:] for group in groups_short]
    groups_short = [group for group in groups_short if len(group)>1]

    # Dictionary to store pair counts
    pair_counts = defaultdict(int)
    # Loop through each sublist and generate pairs
    for sublist in groups_short:
        for pair in combinations(sorted(sublist), 2):  # Generate sorted pairs to avoid (1, 2) and (2, 1) being counted separately
            pair_counts[pair] += 1
    # Convert the defaultdict to a regular dictionary for better readability (optional)
    pair_counts = dict(pair_counts)

    pairs = []
    for key, value in pair_counts.items():
        value_0 = mask_numbers[key[0]]
        value_1 = mask_numbers[key[1]]
        k = value/min(value_0, value_1)
        if k>0.2:
            pairs.append(key)

    merged_pairs = join_pairs(pairs)

    for group in merged_pairs:
        main_id = group[0]
        for similar_id in group[1:]:
            image_for_masks[image_for_masks == similar_id] = main_id
    image_for_masks_max = image_for_masks.max(axis=2)
    return image_for_masks_max