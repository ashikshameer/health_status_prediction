# ==============================================================================
# PROJECT: Health Status Prediction using Machine Learning in R
# ------------------------------------------------------------------------------
# Authors / Team Members:
#   - Hanna Sabik     (ICE24AD028)
#   - Rinu Fathima MS (ICE24AD055)
#   - Hiba Fathima PS (ICE24AD031)
#   - Fouzan Afeef    (ICE24AD025)
#   - Akhil Raj       (ICE24AD007)
# ------------------------------------------------------------------------------
# Description:
#   A clean, simple machine learning pipeline in R to predict an individual's
#   health status (Low Risk vs. High Risk) based on demographic, biometric,
#   and lifestyle parameters.
#
# Algorithms Evaluated:
#   1. Logistic Regression
#   2. Decision Tree (rpart)
#   3. Random Forest (randomForest)
#   4. Support Vector Machine (e1071 svm)
# ==============================================================================

# ------------------------------------------------------------------------------
# STEP 1: Load Required Libraries
# ------------------------------------------------------------------------------
cat("\n[STEP 1] Loading required packages...\n")
suppressPackageStartupMessages({
  library(rpart)         # For Decision Tree
  library(randomForest)  # For Random Forest
  library(e1071)         # For Support Vector Machine (SVM)
  library(ggplot2)       # For Data Visualization
})
cat("-> All packages loaded successfully.\n")

# Set random seed for reproducibility
set.seed(42)

# ------------------------------------------------------------------------------
# STEP 2: Load and Prepare Dataset
# ------------------------------------------------------------------------------
cat("\n[STEP 2] Loading dataset...\n")
data_path <- "data/health_data.csv"

if (!file.exists(data_path)) {
  # If running from another directory, check full path
  data_path <- "/home/ashik/health_status_prediction/data/health_data.csv"
}

raw_data <- read.csv(data_path, stringsAsFactors = TRUE)

# Select relevant clinical and lifestyle features + target
selected_cols <- c(
  "age", "gender", "systolic_bp", "diastolic_bp", "blood_glucose",
  "bmi", "heart_rate", "physical_activity_min", "sleep_hours",
  "cholesterol_total", "smoking_status", "alcohol_intake",
  "stress_level", "health_risk"
)

health_df <- raw_data[, selected_cols]

# Ensure health_risk is a binary factor (Low_Risk = Negative, High_Risk = Positive)
health_df$health_risk <- factor(health_df$health_risk, levels = c("Low_Risk", "High_Risk"))

cat(sprintf("-> Dataset loaded: %d records, %d features\n", nrow(health_df), ncol(health_df) - 1))
cat("-> Class distribution in dataset:\n")
print(table(health_df$health_risk))

# ------------------------------------------------------------------------------
# STEP 3: Split Data into Training (80%) and Testing (20%) Sets
# ------------------------------------------------------------------------------
cat("\n[STEP 3] Splitting dataset into 80% Train and 20% Test...\n")
total_rows <- nrow(health_df)
train_indices <- sample(1:total_rows, size = 0.80 * total_rows)

train_data <- health_df[train_indices, ]
test_data  <- health_df[-train_indices, ]

cat(sprintf("-> Training Set: %d samples\n", nrow(train_data)))
cat(sprintf("-> Testing Set:  %d samples\n", nrow(test_data)))

# ------------------------------------------------------------------------------
# STEP 4: Train Machine Learning Models
# ------------------------------------------------------------------------------
cat("\n[STEP 4] Training Classification Models...\n")

# 1. Logistic Regression
cat("  1. Training Logistic Regression...\n")
model_lr <- glm(health_risk ~ ., data = train_data, family = binomial)

# 2. Decision Tree
cat("  2. Training Decision Tree...\n")
model_dt <- rpart(health_risk ~ ., data = train_data, method = "class")

# 3. Random Forest
cat("  3. Training Random Forest (ntree = 100)...\n")
model_rf <- randomForest(health_risk ~ ., data = train_data, ntree = 100)

# 4. Support Vector Machine (SVM)
cat("  4. Training Support Vector Machine (RBF Kernel)...\n")
model_svm <- svm(health_risk ~ ., data = train_data, probability = TRUE)

cat("-> All 4 models trained successfully.\n")

# ------------------------------------------------------------------------------
# STEP 5: Evaluate Models on Test Data
# ------------------------------------------------------------------------------
cat("\n[STEP 5] Evaluating Models on Test Set...\n")

# Helper function to compute classification metrics
get_metrics <- function(actual, predicted, model_name) {
  cm <- table(Actual = actual, Predicted = predicted)
  
  # Confusion Matrix elements
  tn <- cm["Low_Risk", "Low_Risk"]
  fp <- cm["Low_Risk", "High_Risk"]
  fn <- cm["High_Risk", "Low_Risk"]
  tp <- cm["High_Risk", "High_Risk"]
  
  accuracy  <- round(((tp + tn) / sum(cm)) * 100, 2)
  precision <- round((tp / (tp + fp)) * 100, 2)
  recall    <- round((tp / (tp + fn)) * 100, 2)
  f1_score  <- round((2 * (precision * recall) / (precision + recall)), 2)
  
  cat(sprintf("\n--- %s Confusion Matrix ---\n", toupper(model_name)))
  print(cm)
  
  return(data.frame(
    Model     = model_name,
    Accuracy  = accuracy,
    Precision = precision,
    Recall    = recall,
    F1_Score  = f1_score
  ))
}

# --- Predictions on Test Set ---
# 1. Logistic Regression Predictions (prob >= 0.50 -> High_Risk)
lr_probs <- predict(model_lr, newdata = test_data, type = "response")
lr_preds <- factor(ifelse(lr_probs >= 0.50, "High_Risk", "Low_Risk"), levels = c("Low_Risk", "High_Risk"))
res_lr <- get_metrics(test_data$health_risk, lr_preds, "Logistic Regression")

# 2. Decision Tree Predictions
dt_preds <- predict(model_dt, newdata = test_data, type = "class")
res_dt <- get_metrics(test_data$health_risk, dt_preds, "Decision Tree")

# 3. Random Forest Predictions
rf_preds <- predict(model_rf, newdata = test_data)
res_rf <- get_metrics(test_data$health_risk, rf_preds, "Random Forest")

# 4. Support Vector Machine Predictions
svm_preds <- predict(model_svm, newdata = test_data)
res_svm <- get_metrics(test_data$health_risk, svm_preds, "Support Vector Machine")

# Combine results into a summary table
metrics_table <- rbind(res_lr, res_dt, res_rf, res_svm)

cat("\n======================================================================\n")
cat("                   MODEL PERFORMANCE COMPARISON (%)                   \n")
cat("======================================================================\n")
print(metrics_table, row.names = FALSE)
cat("======================================================================\n")

# Save metrics table to CSV
write.csv(metrics_table, "models/simple_metrics_summary.csv", row.names = FALSE)
cat("-> Metrics saved to 'models/simple_metrics_summary.csv'\n")

# ------------------------------------------------------------------------------
# STEP 6: Visualization of Model Comparison
# ------------------------------------------------------------------------------
cat("\n[STEP 6] Generating Model Comparison Plot...\n")

# Reshape data for ggplot
plot_data <- data.frame(
  Model = rep(metrics_table$Model, 4),
  Metric = c(
    rep("Accuracy", 4),
    rep("Precision", 4),
    rep("Recall", 4),
    rep("F1-Score", 4)
  ),
  Score = c(
    metrics_table$Accuracy,
    metrics_table$Precision,
    metrics_table$Recall,
    metrics_table$F1_Score
  )
)

comparison_plot <- ggplot(plot_data, aes(x = Model, y = Score, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  geom_text(aes(label = sprintf("%.1f%%", Score)),
            position = position_dodge(width = 0.8), vjust = -0.5, size = 3.2, fontface = "bold") +
  scale_y_continuous(limits = c(0, 105), breaks = seq(0, 100, 20)) +
  scale_fill_manual(values = c("Accuracy" = "#1E88E5", "Precision" = "#00897B", "Recall" = "#FB8C00", "F1-Score" = "#5E35B1")) +
  labs(
    title = "Machine Learning Models Performance Comparison",
    subtitle = "Health Status Prediction Project (Test Set Evaluation)",
    x = "Classification Algorithm",
    y = "Performance Score (%)",
    fill = "Metric"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray30"),
    legend.position = "top",
    axis.text.x = element_text(face = "bold", size = 11),
    panel.grid.minor = element_blank()
  )

# Save plot to plots folder
if (!dir.exists("plots")) dir.create("plots")
ggsave("plots/simple_model_comparison.png", plot = comparison_plot, width = 10, height = 6, dpi = 300)
cat("-> Plot saved to 'plots/simple_model_comparison.png'\n")

# ------------------------------------------------------------------------------
# STEP 7: Simple Prediction Function for New Patient Data
# ------------------------------------------------------------------------------
cat("\n[STEP 7] Testing Health Status Prediction for New Patients...\n")

predict_patient_status <- function(patient_data, model = model_lr) {
  # Calculate risk probability using Logistic Regression
  prob <- predict(model, newdata = patient_data, type = "response")
  prob_pct <- round(as.numeric(prob) * 100, 1)
  
  if (prob_pct >= 50) {
    status <- "HIGH RISK"
    advice <- "Attention recommended: Consult a healthcare professional for clinical evaluation."
  } else {
    status <- "LOW RISK (NORMAL)"
    advice <- "Health metrics are within healthy limits. Maintain regular exercise and balanced diet."
  }
  
  return(list(
    Predicted_Status = status,
    Risk_Probability = paste0(prob_pct, "%"),
    Recommendation   = advice
  ))
}

# --- Sample Test 1: High Risk Profile ---
sample_patient_high <- data.frame(
  age = 58,
  gender = factor("Male", levels = levels(train_data$gender)),
  systolic_bp = 150,
  diastolic_bp = 95,
  blood_glucose = 145,
  bmi = 31.5,
  heart_rate = 88,
  physical_activity_min = 10,
  sleep_hours = 5.0,
  cholesterol_total = 250,
  smoking_status = factor("Current", levels = levels(train_data$smoking_status)),
  alcohol_intake = factor("Moderate", levels = levels(train_data$alcohol_intake)),
  stress_level = factor("High", levels = levels(train_data$stress_level))
)

# --- Sample Test 2: Low Risk Profile ---
sample_patient_low <- data.frame(
  age = 28,
  gender = factor("Female", levels = levels(train_data$gender)),
  systolic_bp = 115,
  diastolic_bp = 75,
  blood_glucose = 85,
  bmi = 21.8,
  heart_rate = 68,
  physical_activity_min = 50,
  sleep_hours = 7.5,
  cholesterol_total = 170,
  smoking_status = factor("Never", levels = levels(train_data$smoking_status)),
  alcohol_intake = factor("None", levels = levels(train_data$alcohol_intake)),
  stress_level = factor("Low", levels = levels(train_data$stress_level))
)

cat("\n======================================================================\n")
cat("                       SAMPLE PATIENT PREDICTIONS                     \n")
cat("======================================================================\n")

cat("\n[PATIENT 1] Age: 58, Male | BP: 150/95 | Glucose: 145 | BMI: 31.5 | Smoker: Yes\n")
result1 <- predict_patient_status(sample_patient_high)
cat(sprintf("  -> Predicted Status: %s\n", result1$Predicted_Status))
cat(sprintf("  -> Risk Probability: %s\n", result1$Risk_Probability))
cat(sprintf("  -> Recommendation:   %s\n", result1$Recommendation))

cat("\n[PATIENT 2] Age: 28, Female | BP: 115/75 | Glucose: 85 | BMI: 21.8 | Smoker: No\n")
result2 <- predict_patient_status(sample_patient_low)
cat(sprintf("  -> Predicted Status: %s\n", result2$Predicted_Status))
cat(sprintf("  -> Risk Probability: %s\n", result2$Risk_Probability))
cat(sprintf("  -> Recommendation:   %s\n", result2$Recommendation))

cat("\n======================================================================\n")
cat("             PROJECT EXECUTION COMPLETED SUCCESSFULLY!                \n")
cat("======================================================================\n")
