# ==============================================================================
# Health Status Prediction - CLI & Sample Predictor Script
# Demonstrates single-patient inference and clinical risk analysis
# ==============================================================================

suppressPackageStartupMessages({
  library(rpart)
  library(randomForest)
  library(e1071)
})

models_path <- "/home/ashik/health_status_prediction/models/trained_models.rds"
if (!file.exists(models_path)) {
  stop("Model artifacts not found! Run R/02_preprocess_and_train.R first.")
}

bundle <- readRDS(models_path)

#' Predict Health Status for a patient
#' @param input_df A data frame with required clinical/lifestyle columns
#' @param model_type One of "logistic_regression", "random_forest", "svm", "decision_tree"
predict_health_status <- function(input_df, model_type = "logistic_regression") {
  # Validate factor levels
  input_df$gender <- factor(input_df$gender, levels = c("Female", "Male"))
  input_df$smoking_status <- factor(input_df$smoking_status, levels = c("Never", "Former", "Current"))
  input_df$alcohol_intake <- factor(input_df$alcohol_intake, levels = c("None", "Moderate", "Heavy"))
  input_df$stress_level <- factor(input_df$stress_level, levels = c("Low", "Medium", "High"))
  
  if (model_type == "logistic_regression") {
    prob <- predict(bundle$logistic_regression, newdata = input_df, type = "response")
  } else if (model_type == "random_forest") {
    prob <- predict(bundle$random_forest, newdata = input_df, type = "prob")[, "High_Risk"]
  } else if (model_type == "svm") {
    pred <- predict(bundle$svm, newdata = input_df, probability = TRUE)
    prob <- attr(pred, "probabilities")[, "High_Risk"]
  } else if (model_type == "decision_tree") {
    prob <- predict(bundle$decision_tree, newdata = input_df, type = "prob")[, "High_Risk"]
  } else {
    stop("Invalid model_type specified.")
  }
  
  risk_prob <- as.numeric(prob)
  risk_status <- ifelse(risk_prob >= 0.50, "High Risk", "Low Risk")
  
  category <- ifelse(
    risk_prob < 0.35, "Low Risk (Healthy)",
    ifelse(risk_prob < 0.65, "Moderate Risk (Needs Monitoring)", "High Risk (Medical Attention Advised)")
  )
  
  # Clinical flags detection
  flags <- list()
  if (input_df$systolic_bp >= 140 || input_df$diastolic_bp >= 90) {
    flags <- c(flags, sprintf("Hypertension Stage 2 (%d/%d mmHg)", input_df$systolic_bp, input_df$diastolic_bp))
  } else if (input_df$systolic_bp >= 130 || input_df$diastolic_bp >= 80) {
    flags <- c(flags, sprintf("Hypertension Stage 1 (%d/%d mmHg)", input_df$systolic_bp, input_df$diastolic_bp))
  }
  
  if (input_df$blood_glucose >= 126) {
    flags <- c(flags, sprintf("Elevated Fasting Glucose: Diabetes Range (%d mg/dL)", input_df$blood_glucose))
  } else if (input_df$blood_glucose >= 100) {
    flags <- c(flags, sprintf("Prediabetes Range Glucose (%d mg/dL)", input_df$blood_glucose))
  }
  
  if (input_df$bmi >= 30) {
    flags <- c(flags, sprintf("Obese BMI (%.1f kg/m²)", input_df$bmi))
  } else if (input_df$bmi >= 25) {
    flags <- c(flags, sprintf("Overweight BMI (%.1f kg/m²)", input_df$bmi))
  }
  
  if (input_df$cholesterol_total >= 240) {
    flags <- c(flags, sprintf("High Total Cholesterol (%d mg/dL)", input_df$cholesterol_total))
  } else if (input_df$cholesterol_total >= 200) {
    flags <- c(flags, sprintf("Borderline High Cholesterol (%d mg/dL)", input_df$cholesterol_total))
  }
  
  if (input_df$physical_activity_min < 20) {
    flags <- c(flags, sprintf("Sedentary Lifestyle (%d mins/day)", input_df$physical_activity_min))
  }
  
  if (input_df$smoking_status == "Current") {
    flags <- c(flags, "Current Tobacco Smoker")
  }
  
  if (input_df$stress_level == "High") {
    flags <- c(flags, "Chronic High Stress Reported")
  }
  
  return(list(
    model_used = model_type,
    predicted_probability = round(risk_prob * 100, 1),
    predicted_status = risk_status,
    risk_category = category,
    clinical_flags = flags
  ))
}

# Run sample test if executed from CLI
if (!interactive()) {
  cat("\n======================================================================\n")
  cat("          HEALTH STATUS PREDICTION - CLI INFERENCE DEMO               \n")
  cat("======================================================================\n")
  
  # Patient 1: High Risk Profile
  patient_high <- data.frame(
    age = 58,
    gender = "Male",
    systolic_bp = 148,
    diastolic_bp = 94,
    blood_glucose = 142,
    bmi = 31.4,
    heart_rate = 88,
    physical_activity_min = 10,
    sleep_hours = 5.2,
    cholesterol_total = 250,
    smoking_status = "Current",
    alcohol_intake = "Moderate",
    stress_level = "High"
  )
  
  # Patient 2: Low Risk Profile
  patient_low <- data.frame(
    age = 32,
    gender = "Female",
    systolic_bp = 114,
    diastolic_bp = 74,
    blood_glucose = 88,
    bmi = 22.1,
    heart_rate = 68,
    physical_activity_min = 45,
    sleep_hours = 7.5,
    cholesterol_total = 175,
    smoking_status = "Never",
    alcohol_intake = "None",
    stress_level = "Low"
  )
  
  res_high <- predict_health_status(patient_high, model_type = "logistic_regression")
  res_low  <- predict_health_status(patient_low, model_type = "logistic_regression")
  
  cat("\n[CASE 1] High-Risk Assessment:\n")
  cat(sprintf("  - Predicted Status: %s\n", res_high$predicted_status))
  cat(sprintf("  - Risk Category:   %s\n", res_high$risk_category))
  cat(sprintf("  - Risk Score:      %.1f%%\n", res_high$predicted_probability))
  cat("  - Identified Clinical Flags:\n")
  for (flag in res_high$clinical_flags) {
    cat(sprintf("    * [!] %s\n", flag))
  }
  
  cat("\n[CASE 2] Low-Risk Assessment:\n")
  cat(sprintf("  - Predicted Status: %s\n", res_low$predicted_status))
  cat(sprintf("  - Risk Category:   %s\n", res_low$risk_category))
  cat(sprintf("  - Risk Score:      %.1f%%\n", res_low$predicted_probability))
  cat("  - Identified Clinical Flags: None (All within normal baseline)\n")
  
  cat("\n======================================================================\n")
}
