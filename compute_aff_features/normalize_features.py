import numpy as np

def extract_scalar(value):
    if isinstance(value, (list, np.ndarray)):
        value = np.array(value).flatten()
        return float(value[0]) if value.size > 0 else 0.0
    else:
        return float(value)

def normalize_features(features, normalized_features):
    # normalize features
    num_features = len(features[0])
    min_values = [float("inf")] * num_features
    max_values = [-float("inf")] * num_features

    for feature in features:
        for i in range(1, len(feature)):
            value = extract_scalar(feature[i])
            if min_values[i] > value:
                min_values[i] = value
            if max_values[i] < value:
                max_values[i] = value

    for feature in features:
        normalized_feature = [feature[0]]
        for i in range(1, len(feature)):
            a = (max_values[i] + min_values[i]) / 2
            b = (max_values[i] - min_values[i]) / 2
            value = extract_scalar(feature[i])
            normalized_feature.append(0 if b == 0 else (value - a) / b)
        normalized_features.append(normalized_feature)
