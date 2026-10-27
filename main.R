# ==============================================================================
# Health Status Prediction - Master Execution Script
# ==============================================================================

cat("\n======================================================================\n")
cat("      HEALTH STATUS PREDICTION - MACHINE LEARNING SYSTEM (R)          \n")
cat("======================================================================\n")

# Step 1: Generate Dataset if not present
if (!file.exists("data/health_data.csv")) {
  cat("\n[1/3] Generating health & lifestyle dataset...\n")
  source("R/01_generate_data.R")
} else {
  cat("\n[1/3] Dataset data/health_data.csv already exists.\n")
}

# Step 2: Train and evaluate ML models if not present
if (!file.exists("models/trained_models.rds")) {
  cat("\n[2/3] Training and evaluating ML models...\n")
  source("R/02_preprocess_and_train.R")
} else {
  cat("\n[2/3] Trained models and metrics already prepared.\n")
}

# Step 3: Launch Interactive Shiny Web Application
cat("\n[3/3] Launching Interactive Shiny Web Application...\n")
cat("      Access URL: http://127.0.0.1:3838\n")
cat("      Press Ctrl+C in the terminal to stop the server.\n\n")

suppressPackageStartupMessages(library(shiny))
runApp(".", port = 3838, host = "0.0.0.0", launch.browser = FALSE)
