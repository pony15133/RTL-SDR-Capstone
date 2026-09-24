# ML Structure and Feature Extraction Review

## Scope reviewed

This review covers the current machine-learning flow in the SDR prototype, focusing on:

- the feature extractor in `src/features/extractor.py`
- the training script in `src/ml/train.py`
- the inference path in `src/detection/ml_detector.py`
- how the rule-based detector and ML detector share the same feature vector

## Overall ML structure

The project has a sensible separation of concerns:

1. `src/features/extractor.py`
   - defines the canonical feature schema
   - computes signal statistics from a spectrogram
   - exposes a reusable `FeatureVector`
   - converts the vector to a numeric array for scikit-learn

2. `src/ml/train.py`
   - loads a labelled CSV dataset
   - validates required columns and labels
   - builds a RandomForestClassifier
   - trains and evaluates the model
   - saves a `ModelBundle` with metadata and feature importance

3. `src/detection/ml_detector.py`
   - loads a saved model bundle
   - converts a `FeatureVector` to a model-ready array
   - runs prediction and returns confidence score / detection result
   - handles missing or invalid models without crashing the main pipeline

4. `src/detect.py` and `src/main.py`
   - the rule-based detector and ML detector both consume the same extracted feature vector
   - this avoids duplicate logic and keeps feature definitions consistent

This is a good structure for an initial prototype because the feature logic is centralized and the training/inference paths are intentionally separated.

## Feature extraction design

The current `FeatureVector` includes the following ML features:

- `snr_db`
- `frequency_drift_hz`
- `drift_rate_hz_per_second`
- `smoothness_score`
- `valid_signal_ratio`
- `peak_power`
- `mean_power`
- `occupied_bandwidth_hz`
- `signal_duration_seconds`

Those features are derived from the strongest trace in the spectrogram, using per-time-slice strongest-bin tracking and median noise-floor estimation.

### How the extractor works

The logic in `extract_features()` does the following:

- identifies the strongest frequency bin per time slice
- estimates the noise floor from the median power in each slice
- marks valid slices where peak power exceeds the noise floor by a threshold
- tracks the strongest-bin frequency trace over time
- smooths the trace with a median filter
- measures drift, duration, and signal ratio
- computes bandwidth and power-related features

This is a practical and interpretable feature set for synthetic or early real-world detection tasks. It avoids feeding raw IQ into the model, which is consistent with the project rule that the classifier should see engineered features rather than raw arrays.

## Strengths

### 1. Single source of truth for features
The `FeatureVector` and `FEATURE_NAMES` tuple ensure both training and inference use the same feature ordering and the same underlying math.

### 2. Explicit schema validation
`FeatureSchemaError` and `feature_vector_to_array()` catch feature mismatches between a saved model and the current extractor. This is an important safeguard against silent model breakage.

### 3. Clean model lifecycle
The training script validates CSV data, checks for invalid values, guards against accidental synthetic training unless explicitly allowed, and saves metadata with evaluation results.

### 4. Realistic model output expectations
The ML detector returns:

- a binary result
- confidence score for the positive class
- model version
- status such as unavailable, invalid, or available

This is consistent with the intended pipeline contract.

## Risks and gaps

### 1. Potential data leakage risk in train/test splitting
The current training code uses `train_test_split` on the full dataset without grouping by `recording_id` or another capture-level identifier.

This is a meaningful issue because rows from the same signal capture can be highly correlated. If multiple rows originate from the same recording, random splitting can leak information into the test set and inflate evaluation metrics.

Recommendation:

- add a `recording_id` or equivalent capture identifier to the dataset schema
- split by group, not by raw row index
- ensure all rows from the same recording remain in either train or test

This is especially important because the project rules explicitly call out this leakage rule.

### 2. Feature set is useful but still relatively simple
The feature list is good for a prototype, but it may not capture enough nuance for real operational classification. Additional features could improve discrimination, such as:

- spectral centroid
- spectral flatness
- higher-order moments of the signal trace
- SNR variance over time
- variance in occupied bandwidth across slices
- full trace curvature or slope statistics

### 3. Synthetic-data guard is good, but dataset quality still needs validation
The project is careful about not silently training on synthetic data, which is good. However, there is still a risk that a real labelled dataset may be too small, imbalanced, or inconsistent unless the label quality is checked more explicitly.

### 4. The model output is still a binary decision
The ML output is binary (`bool(prediction)`) and confidence is the positive-class probability. That is practical for a prototype, but the project requirements mention support for uncertain/further-examination outcomes. The code currently does not appear to expose a third classification state.

## Recommended next steps

1. Add grouped splitting by capture/recording ID
   - this is the most important improvement to avoid leakage

2. Expand the feature schema
   - add a few more spectral and temporal descriptors

3. Add a dedicated ML dataset definition
   - make `recording_id`, `capture_id`, labels, and synthetic flags explicit in the schema

4. Support uncertainty handling
   - add a threshold-based "uncertain" outcome when confidence falls in a non-decisive band

5. Add tests for feature extraction and split strategy
   - both should be covered by pytest to make the prototype more reliable

## Conclusion

The existing ML structure is well-organized and consistent with the intended architecture. The central feature-extraction layer is a strong design choice, and the training/inference split is clear and maintainable.

The main improvement needed is data-splitting discipline: current training does not enforce grouping by recording or capture ID, which risks evaluation leakage. The feature set is appropriate for a prototype, but adding a few stronger spectral and temporal descriptors would improve robustness.

Overall, this is a solid initial implementation with a clear path toward a more defensible ML pipeline.
