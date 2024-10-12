import numpy as np

def calculate_percentile_threshold(distances, percentile=95):
    """Calculate the threshold at a given percentile directly from distance scores."""
    return np.percentile(distances, percentile)

def active_learning_loop(model, data_loader, unlabeled_data_loader, retrain_func, extract_features_func, calculate_distances_func, annotation_budget):
    """An active learning loop that dynamically adjusts the threshold for OOD detection."""
    # Initial setup: Assuming some initial labeled data is available for training
    model.train()
    retrain_func(model, data_loader)

    # Initial calculation of distances on a validation or hold-out set
    val_features, _ = extract_features_func(model, data_loader)
    val_distances = calculate_distances_func(val_features)
    current_threshold = calculate_percentile_threshold(val_distances)

    while annotation_budget > 0:
        model.eval()

        # Process the batch of unlabeled data
        unlabeled_features, _ = extract_features_func(model, unlabeled_data_loader)
        unlabeled_distances = calculate_distances_func(unlabeled_features)

        # Apply the current threshold to classify data as IND (0) or OOD (1)
        predictions = [0 if dist <= current_threshold else 1 for dist in unlabeled_distances]
        
        # Calculate entropy or other uncertainty measures to select samples for annotation
        entropy_scores = calculate_entropy(unlabeled_features)  # Define calculate_entropy or similar function
        samples_to_annotate = select_samples_based_on_entropy(entropy_scores, annotation_budget)  # Define this selection logic

        # Human annotation process (simulated or actual)
        true_labels = human_annotation_process(samples_to_annotate)  # Define how labels are obtained

        # Update the model and retrain with new data
        update_training_set(data_loader, samples_to_annotate, true_labels)  # Update your training dataset
        retrain_func(model, data_loader)

        # Recalculate the threshold using updated model on validation or hold-out set
        val_features, _ = extract_features_func(model, data_loader)
        val_distances = calculate_distances_func(val_features)
        current_threshold = calculate_percentile_threshold(val_distances)

        # Update the annotation budget
        annotation_budget -= len(samples_to_annotate)

    return model, current_threshold


# wasserstein_distance
def calculate_wasserstein_distance_to_classes(sample, class_features):
    distances = []
    for label, features in class_features.items():
        class_distribution = np.mean(features, axis=0)  # Assuming mean vector represents the class
        distance = wasserstein_distance(sample.flatten(), class_distribution.flatten())
        distances.append(distance)
    return np.min(distances)






#Furthr classify
def calculate_ood_threshold(distances, percentile=95):
    """
    Calculate the threshold for OOD detection based on a given percentile.
    Args:
        distances (np.array): Array of distances of validation samples from the nearest class centroid.
        percentile (int): Percentile to use for thresholding (default 95).
    Returns:
        float: Calculated threshold.
    """
    return np.percentile(distances, percentile)

def classify_samples(distances, class_means, threshold):
    """
    Classify samples as OOD or assign them to the closest IND class based on distances.
    Args:
        distances (np.array): 2D array where each column represents distances of all samples to a particular class.
        class_means (dict): Dictionary with class labels as keys and their centroids as values.
        threshold (float): Distance threshold to determine if a sample is OOD.
    Returns:
        list: List of tuples (classification, class_label) where 'classification' is 'OOD' or 'IND',
              and 'class_label' is None if OOD or the class label if IND.
    """
    classifications = []
    for distance_set in distances.T:  # Transpose to iterate over samples
        if min(distance_set) > threshold:
            classifications.append(('OOD', None))
        else:
            closest_class_index = np.argmin(distance_set)
            class_label = list(class_means.keys())[closest_class_index]
            classifications.append(('IND', class_label))
    return classifications

# Example usage
# Assuming `val_features` from validation set and `class_means`, `pooled_cov` from training set
val_distances = np.array([mahalanobis_distance(val_features, mean, pooled_cov) for mean in class_means.values()])
val_distances = val_distances.T  # Transpose so that each column is a sample's distance to all class centroids

# Calculate the threshold for OOD detection
threshold = calculate_ood_threshold(val_distances.flatten())  # Flatten if you want a global threshold

# Classify each sample based on distances and threshold
classifications = classify_samples(val_distances, class_means, threshold)
