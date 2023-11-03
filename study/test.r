# from https://github.com/rivolli/utiml#running-ensemble-of-classifier-chains

library(utiml)

# Base classifiers (SVM and Random Forest)
install.packages(c("e1071", "randomForest"))

# Create three partitions (train, val, test) of emotions dataset
partitions <- c(train = 0.6, val = 0.2, test = 0.2)
ds <- create_holdout_partition(emotions, partitions, method = "iterative")

# Create an Ensemble of Classifier Chains using Random Forest (randomForest package)
eccmodel <- ecc(ds$train, "RF", m = 3, seed = 123)

# Predict
val <- predict(eccmodel, ds$val)
test <- predict(eccmodel, ds$test)

# Apply a threshold
thresholds <- scut_threshold(val, ds$val)
new.val <- fixed_threshold(val, thresholds)
new.test <- fixed_threshold(test, thresholds)

# Evaluate the models
measures <- c("subset-accuracy", "F1", "hamming-loss", "macro-based")

result <- cbind(
    Test = multilabel_evaluate(ds$tes, test, measures),
    TestWithThreshold = multilabel_evaluate(ds$tes, new.test, measures),
    Validation = multilabel_evaluate(ds$val, val, measures),
    ValidationWithThreshold = multilabel_evaluate(ds$val, new.val, measures)
)

print(round(result, 3))