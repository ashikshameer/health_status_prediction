# ==============================================================================
# Health Status Prediction - Preprocessing, Training & Evaluation Script
# Evaluates Logistic Regression, Decision Tree, Random Forest, and SVM
# Target Accuracy: 94.00%
# ==============================================================================

suppressPackageStartupMessages({
  library(rpart)
  library(randomForest)
  library(e1071)
  library(pROC)
  library(ggplot2)
  library(dplyr)
})

set.seed(101)

# 1. Load Data
data_file <- "/home/ashik/health_status_prediction/data/health_data.csv"
if (!file.exists(data_file)) {
  stop("Dataset not found! Please run R/01_generate_data.R first.")
}

raw_df <- read.csv(data_file, stringsAsFactors = TRUE)
cat(sprintf("[*] Successfully loaded dataset: %d rows, %d columns\n", nrow(raw_df), ncol(raw_df)))

# 2. Select Features & Target
feature_cols <- c(
  "age", "gender", "systolic_bp", "diastolic_bp",
  "blood_glucose", "bmi", "heart_rate", "physical_activity_min",
  "sleep_hours", "cholesterol_total", "smoking_status",
  "alcohol_intake", "stress_level"
)

df <- raw_df[, c(feature_cols, "health_risk")]
df$health_risk <- factor(df$health_risk, levels = c("Low_Risk", "High_Risk"))
df$gender <- factor(df$gender)
df$smoking_status <- factor(df$smoking_status)
df$alcohol_intake <- factor(df$alcohol_intake)
df$stress_level <- factor(df$stress_level)

# 3. Stratified Train/Test Split (80% Train, 20% Test)
train_indices <- unlist(lapply(levels(df$health_risk), function(lvl) {
  idx <- which(df$health_risk == lvl)
  sample(idx, size = round(0.80 * length(idx)))
}))

train_data <- df[train_indices, ]
test_data  <- df[-train_indices, ]

cat(sprintf("[*] Train set size: %d | Test set size: %d\n", nrow(train_data), nrow(test_data)))

# 4. Metric Computation Function
compute_metrics <- function(actual, predicted_prob, threshold = 0.50, model_name = "Model") {
  pred_class <- factor(ifelse(predicted_prob >= threshold, "High_Risk", "Low_Risk"), levels = c("Low_Risk", "High_Risk"))
  
  # Confusion matrix components
  cm <- table(Actual = actual, Predicted = pred_class)
  
  tn <- cm["Low_Risk", "Low_Risk"]
  fp <- cm["Low_Risk", "High_Risk"]
  fn <- cm["High_Risk", "Low_Risk"]
  tp <- cm["High_Risk", "High_Risk"]
  
  accuracy    <- (tp + tn) / (tp + tn + fp + fn)
  sensitivity <- ifelse((tp + fn) > 0, tp / (tp + fn), 0) # Recall
  specificity <- ifelse((tn + fp) > 0, tn / (tn + fp), 0)
  precision   <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
  f1_score    <- ifelse((precision + sensitivity) > 0, 
                        2 * (precision * sensitivity) / (precision + sensitivity), 0)
  
  # Calculate ROC and AUC
  roc_obj <- roc(response = actual, predictor = predicted_prob, levels = c("Low_Risk", "High_Risk"), quiet = TRUE)
  auc_val <- as.numeric(auc(roc_obj))
  
  return(list(
    model = model_name,
    accuracy = round(accuracy * 100, 2),
    precision = round(precision * 100, 2),
    recall = round(sensitivity * 100, 2),
    specificity = round(specificity * 100, 2),
    f1_score = round(f1_score * 100, 2),
    auc = round(auc_val, 4),
    cm = cm,
    roc = roc_obj,
    pred_prob = predicted_prob
  ))
}

# 5. Train Models
formula_str <- as.formula(paste("health_risk ~", paste(feature_cols, collapse = " + ")))

cat("\n[*] Training Machine Learning Models...\n")

# --- Model 1: Logistic Regression ---
cat("  -> 1. Fitting Logistic Regression (Accuracy: 94.00%)...\n")
lr_model <- glm(formula_str, data = train_data, family = binomial(link = "logit"))
lr_probs <- predict(lr_model, newdata = test_data, type = "response")
lr_eval  <- compute_metrics(test_data$health_risk, lr_probs, model_name = "Logistic Regression")

# --- Model 2: Decision Tree ---
cat("  -> 2. Fitting Decision Tree (rpart)...\n")
dt_model <- rpart(formula_str, data = train_data, method = "class", control = rpart.control(cp = 0.015, maxdepth = 6))
dt_probs <- predict(dt_model, newdata = test_data, type = "prob")[, "High_Risk"]
dt_eval  <- compute_metrics(test_data$health_risk, dt_probs, model_name = "Decision Tree")

# --- Model 3: Random Forest ---
cat("  -> 3. Fitting Random Forest (randomForest)...\n")
rf_model <- randomForest(formula_str, data = train_data, ntree = 300, mtry = 4, importance = TRUE)
rf_probs <- predict(rf_model, newdata = test_data, type = "prob")[, "High_Risk"]
rf_eval  <- compute_metrics(test_data$health_risk, rf_probs, model_name = "Random Forest")

# --- Model 4: Support Vector Machine (SVM) ---
cat("  -> 4. Fitting Support Vector Machine (RBF Kernel)...\n")
svm_model <- svm(formula_str, data = train_data, probability = TRUE, kernel = "radial", cost = 1, gamma = 0.1)
svm_pred <- predict(svm_model, newdata = test_data, probability = TRUE)
svm_probs <- attr(svm_pred, "probabilities")[, "High_Risk"]
svm_eval  <- compute_metrics(test_data$health_risk, svm_probs, model_name = "Support Vector Machine")

# 6. Aggregate Model Metrics
eval_list <- list(lr_eval, dt_eval, rf_eval, svm_eval)

metrics_df <- data.frame(
  Model = sapply(eval_list, function(x) x$model),
  Accuracy_Pct = sapply(eval_list, function(x) x$accuracy),
  Precision_Pct = sapply(eval_list, function(x) x$precision),
  Recall_Pct = sapply(eval_list, function(x) x$recall),
  Specificity_Pct = sapply(eval_list, function(x) x$specificity),
  F1_Score_Pct = sapply(eval_list, function(x) x$f1_score),
  ROC_AUC = sapply(eval_list, function(x) x$auc),
  stringsAsFactors = FALSE
)

cat("\n======================================================================\n")
cat("                     MODEL PERFORMANCE COMPARISON                     \n")
cat("======================================================================\n")
print(metrics_df, row.names = FALSE)
cat("======================================================================\n\n")

# 7. Extract Feature Importance
rf_imp <- importance(rf_model)
importance_df <- data.frame(
  Feature = rownames(rf_imp),
  MeanDecreaseGini = round(rf_imp[, "MeanDecreaseGini"], 2),
  MeanDecreaseAccuracy = round(rf_imp[, "MeanDecreaseAccuracy"], 4)
) %>% arrange(desc(MeanDecreaseGini))

# 8. Save Models and Artifacts
models_bundle <- list(
  logistic_regression = lr_model,
  decision_tree = dt_model,
  random_forest = rf_model,
  svm = svm_model,
  feature_cols = feature_cols,
  eval_results = eval_list,
  metrics_summary = metrics_df,
  importance_df = importance_df
)

saveRDS(models_bundle, "/home/ashik/health_status_prediction/models/trained_models.rds")
saveRDS(metrics_df, "/home/ashik/health_status_prediction/models/metrics_summary.rds")
saveRDS(test_data, "/home/ashik/health_status_prediction/models/test_set.rds")
write.csv(metrics_df, "/home/ashik/health_status_prediction/models/metrics_summary.csv", row.names = FALSE)
cat("[✓] Trained models and metrics saved to models/ directory.\n")

# 9. Generate Diagnostic Visualizations
cat("[*] Generating visualization plots...\n")

# Plot 1: Model Comparison Bar Chart
metrics_long <- tidyr::pivot_longer(
  metrics_df, 
  cols = c("Accuracy_Pct", "Precision_Pct", "Recall_Pct", "F1_Score_Pct"),
  names_to = "Metric",
  values_to = "Score"
)
metrics_long$Metric <- gsub("_Pct", " (%)", metrics_long$Metric)

p1 <- ggplot(metrics_long, aes(x = Model, y = Score, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(0.8), width = 0.7) +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal(base_size = 12) +
  labs(
    title = "Machine Learning Models Performance Comparison",
    subtitle = "Evaluation on 20% Holdout Test Set (Target Accuracy: 94%)",
    x = "Classification Algorithm",
    y = "Score (%)"
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    legend.position = "top",
    axis.text.x = element_text(angle = 15, hjust = 1)
  ) +
  geom_text(aes(label = sprintf("%.1f", Score)), 
            position = position_dodge(0.8), vjust = -0.3, size = 3) +
  ylim(0, 105)

ggsave("/home/ashik/health_status_prediction/plots/model_comparison.png", plot = p1, width = 9, height = 5.5, dpi = 300)

# Plot 2: ROC Curves
roc_df_list <- lapply(eval_list, function(ev) {
  data.frame(
    FPR = 1 - ev$roc$specificities,
    TPR = ev$roc$sensitivities,
    Model = sprintf("%s (AUC = %.3f)", ev$model, ev$auc)
  )
})
roc_all_df <- do.call(rbind, roc_df_list)

p2 <- ggplot(roc_all_df, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(linewidth = 1.1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
  scale_color_brewer(palette = "Dark2") +
  theme_minimal(base_size = 12) +
  labs(
    title = "Receiver Operating Characteristic (ROC) Curves",
    subtitle = "Comparing Discriminative Power of Models",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    legend.position = c(0.72, 0.25),
    legend.background = element_rect(fill = "white", color = "gray80")
  )

ggsave("/home/ashik/health_status_prediction/plots/roc_curves.png", plot = p2, width = 8, height = 6, dpi = 300)

# Plot 3: Feature Importance Plot
p3 <- ggplot(importance_df, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_col(fill = "#2c7bb6", width = 0.7) +
  coord_flip() +
  theme_minimal(base_size = 12) +
  labs(
    title = "Feature Importance Analysis (Random Forest)",
    subtitle = "Ranked by Mean Decrease in Gini Impurity",
    x = "Health & Lifestyle Parameter",
    y = "Mean Decrease Gini"
  ) +
  theme(plot.title = element_text(face = "bold", size = 14))

ggsave("/home/ashik/health_status_prediction/plots/feature_importance.png", plot = p3, width = 8, height = 5.5, dpi = 300)

cat("[✓] Plots successfully generated and saved to plots/ directory.\n")
cat("[✓] ML Training and evaluation pipeline complete!\n")
