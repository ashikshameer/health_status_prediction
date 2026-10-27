# ==============================================================================
# Interactive Patient Health Status Predictor (R)
# Allows interactive input of patient vitals to evaluate health risk
# ==============================================================================

suppressPackageStartupMessages({
  library(rpart)
})

# Load dataset to train baseline model
data_path <- "data/health_data.csv"
if (!file.exists(data_path)) {
  data_path <- "/home/ashik/health_status_prediction/data/health_data.csv"
}

df <- read.csv(data_path, stringsAsFactors = TRUE)
cols <- c("age", "gender", "systolic_bp", "diastolic_bp", "blood_glucose",
          "bmi", "heart_rate", "physical_activity_min", "sleep_hours",
          "cholesterol_total", "smoking_status", "alcohol_intake",
          "stress_level", "health_risk")
df <- df[, cols]
df$health_risk <- factor(df$health_risk, levels = c("Low_Risk", "High_Risk"))

cat("\n[1/2] Training predictive model (Logistic Regression)...\n")
model <- glm(health_risk ~ ., data = df, family = binomial)
cat("-> Model ready!\n\n")

predict_patient <- function(age, gender, systolic_bp, diastolic_bp, blood_glucose,
                            bmi, heart_rate, physical_activity_min, sleep_hours,
                            cholesterol_total, smoking_status, alcohol_intake, stress_level) {
  
  patient <- data.frame(
    age = as.numeric(age),
    gender = factor(gender, levels = levels(df$gender)),
    systolic_bp = as.numeric(systolic_bp),
    diastolic_bp = as.numeric(diastolic_bp),
    blood_glucose = as.numeric(blood_glucose),
    bmi = as.numeric(bmi),
    heart_rate = as.numeric(heart_rate),
    physical_activity_min = as.numeric(physical_activity_min),
    sleep_hours = as.numeric(sleep_hours),
    cholesterol_total = as.numeric(cholesterol_total),
    smoking_status = factor(smoking_status, levels = levels(df$smoking_status)),
    alcohol_intake = factor(alcohol_intake, levels = levels(df$alcohol_intake)),
    stress_level = factor(stress_level, levels = levels(df$stress_level))
  )
  
  prob <- predict(model, newdata = patient, type = "response")
  prob_pct <- round(as.numeric(prob) * 100, 1)
  
  status <- ifelse(prob_pct >= 50, "HIGH RISK", "LOW RISK (NORMAL)")
  
  cat("======================================================================\n")
  cat("                    PREDICTION RESULTS FOR PATIENT                    \n")
  cat("======================================================================\n")
  cat(sprintf("  PREDICTED HEALTH STATUS: %s\n", status))
  cat(sprintf("  RISK PROBABILITY:        %.1f%%\n", prob_pct))
  cat("----------------------------------------------------------------------\n")
  
  # Clinical flags and guidance
  cat("  CLINICAL PARAMETER FLAGS:\n")
  has_flag <- FALSE
  if (systolic_bp >= 140 || diastolic_bp >= 90) {
    cat(sprintf("   [!] Blood Pressure High: %d/%d mmHg (Stage 2 Hypertension)\n", systolic_bp, diastolic_bp))
    has_flag <- TRUE
  }
  if (blood_glucose >= 126) {
    cat(sprintf("   [!] Blood Glucose High: %d mg/dL (Diabetes Range)\n", blood_glucose))
    has_flag <- TRUE
  } else if (blood_glucose >= 100) {
    cat(sprintf("   [!] Blood Glucose Borderline: %d mg/dL (Prediabetes Range)\n", blood_glucose))
    has_flag <- TRUE
  }
  if (bmi >= 30) {
    cat(sprintf("   [!] BMI High: %.1f kg/m² (Obesity Range)\n", bmi))
    has_flag <- TRUE
  }
  if (physical_activity_min < 20) {
    cat(sprintf("   [!] Sedentary Physical Activity: %d mins/day\n", physical_activity_min))
    has_flag <- TRUE
  }
  if (smoking_status == "Current") {
    cat("   [!] Tobacco Smoking: Current Smoker\n")
    has_flag <- TRUE
  }
  if (!has_flag) {
    cat("   [✓] All vital parameters are within normal baseline ranges.\n")
  }
  
  cat("----------------------------------------------------------------------\n")
  if (status == "HIGH RISK") {
    cat("  RECOMMENDATION: Clinical consultation with a medical professional is\n")
    cat("                  advised for full diagnostic follow-up.\n")
  } else {
    cat("  RECOMMENDATION: Health status appears stable. Maintain healthy diet,\n")
    cat("                  routine physical exercise, and regular checkups.\n")
  }
  cat("======================================================================\n\n")
}

# Example run:
cat("Running demonstration for a high-risk patient:\n")
predict_patient(
  age = 54, gender = "Male", systolic_bp = 148, diastolic_bp = 92,
  blood_glucose = 138, bmi = 29.5, heart_rate = 84, physical_activity_min = 15,
  sleep_hours = 6.0, cholesterol_total = 235, smoking_status = "Current",
  alcohol_intake = "Moderate", stress_level = "High"
)
