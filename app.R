# ==============================================================================
# Health Status Prediction System - Interactive R Shiny Web Application
# ==============================================================================

suppressPackageStartupMessages({
  library(shiny)
  library(shinydashboard)
  library(ggplot2)
  library(plotly)
  library(DT)
  library(dplyr)
  library(rpart)
  library(randomForest)
  library(e1071)
  library(pROC)
})

# Load Pretrained Models & Artifacts
models_path  <- "/home/ashik/health_status_prediction/models/trained_models.rds"
metrics_path <- "/home/ashik/health_status_prediction/models/metrics_summary.rds"
data_path    <- "/home/ashik/health_status_prediction/data/health_data.csv"

if (!file.exists(models_path) || !file.exists(data_path)) {
  stop("Model artifacts or dataset missing! Please run R/01_generate_data.R and R/02_preprocess_and_train.R first.")
}

bundle     <- readRDS(models_path)
metrics_df <- readRDS(metrics_path)
health_df  <- read.csv(data_path, stringsAsFactors = TRUE)

# Custom CSS for modern, clean clinical UI
custom_css <- "
  .main-header .logo { font-weight: bold; font-size: 20px; }
  .content-wrapper { background-color: #f8fafc; }
  .box { border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-top: 3px solid #3c8dbc; }
  .box-header { font-weight: 600; font-size: 16px; }
  .risk-card {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    color: white;
    margin-bottom: 20px;
    transition: all 0.3s ease;
  }
  .risk-low { background: linear-gradient(135deg, #10b981, #059669); }
  .risk-mod { background: linear-gradient(135deg, #f59e0b, #d97706); }
  .risk-high { background: linear-gradient(135deg, #ef4444, #b91c1c); }
  .disclaimer-box {
    background-color: #fef3c7;
    border-left: 5px solid #f59e0b;
    padding: 12px 16px;
    margin-bottom: 20px;
    border-radius: 4px;
    color: #92400e;
    font-size: 13.5px;
  }
  .metric-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
  }
  .badge-normal { background-color: #d1fae5; color: #065f46; }
  .badge-warning { background-color: #fef3c7; color: #92400e; }
  .badge-danger { background-color: #fee2e2; color: #991b1b; }
"

# ------------------------------------------------------------------------------
# User Interface (UI)
# ------------------------------------------------------------------------------
ui <- dashboardPage(
  skin = "blue",
  dashboardHeader(title = "Health Status AI"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Risk Predictor", tabName = "predictor", icon = icon("heartbeat")),
      menuItem("Model Performance", tabName = "models", icon = icon("chart-line")),
      menuItem("Dataset & Analytics", tabName = "eda", icon = icon("database")),
      menuItem("Clinical Guidelines", tabName = "about", icon = icon("info-circle"))
    )
  ),
  
  dashboardBody(
    tags$head(tags$style(HTML(custom_css))),
    
    # Global Medical Disclaimer Banner
    div(
      class = "disclaimer-box",
      icon("exclamation-triangle"),
      strong(" Medical Disclaimer: "),
      "This system is designed as a machine learning-based preventive screening & awareness tool. It is NOT a substitute for clinical diagnosis, professional medical advice, or treatment."
    ),
    
    tabItems(
      # ------------------------------------------------------------------------
      # TAB 1: HEALTH RISK PREDICTOR
      # ------------------------------------------------------------------------
      tabItem(
        tabName = "predictor",
        fluidRow(
          # Left Column: User Health Inputs
          column(
            width = 5,
            box(
              width = NULL,
              title = "Patient Biometrics & Lifestyle Inputs",
              status = "primary",
              solidHeader = TRUE,
              
              fluidRow(
                column(6, numericInput("age", "Age (years):", value = 45, min = 18, max = 95, step = 1)),
                column(6, selectInput("gender", "Gender:", choices = c("Male", "Female"), selected = "Male"))
              ),
              
              hr(style = "margin: 8px 0;"),
              h5(strong("Cardiovascular & Metabolic Metrics")),
              fluidRow(
                column(6, numericInput("systolic_bp", "Systolic BP (mmHg):", value = 125, min = 80, max = 220)),
                column(6, numericInput("diastolic_bp", "Diastolic BP (mmHg):", value = 82, min = 50, max = 130))
              ),
              fluidRow(
                column(6, numericInput("blood_glucose", "Fasting Glucose (mg/dL):", value = 98, min = 60, max = 300)),
                column(6, numericInput("cholesterol_total", "Total Cholesterol (mg/dL):", value = 190, min = 100, max = 400))
              ),
              fluidRow(
                column(6, numericInput("bmi", "BMI (kg/m²):", value = 24.5, min = 14.0, max = 50.0, step = 0.1)),
                column(6, numericInput("heart_rate", "Resting Heart Rate (bpm):", value = 72, min = 40, max = 150))
              ),
              
              hr(style = "margin: 8px 0;"),
              h5(strong("Lifestyle & Daily Habits")),
              fluidRow(
                column(6, sliderInput("physical_activity_min", "Physical Activity (min/day):", min = 0, max = 120, value = 30, step = 5)),
                column(6, sliderInput("sleep_hours", "Sleep (hours/night):", min = 3, max = 12, value = 7.0, step = 0.5))
              ),
              fluidRow(
                column(4, selectInput("smoking_status", "Smoking:", choices = c("Never", "Former", "Current"), selected = "Never")),
                column(4, selectInput("alcohol_intake", "Alcohol:", choices = c("None", "Moderate", "Heavy"), selected = "Moderate")),
                column(4, selectInput("stress_level", "Stress:", choices = c("Low", "Medium", "High"), selected = "Medium"))
              ),
              
              hr(style = "margin: 8px 0;"),
              selectInput(
                "selected_model", "Select ML Algorithm:",
                choices = c(
                  "Logistic Regression (Accuracy: 94.00%)" = "logistic_regression",
                  "Support Vector Machine (Accuracy: 92.17%)" = "svm",
                  "Random Forest (Accuracy: 91.83%)" = "random_forest",
                  "Decision Tree (Accuracy: 84.00%)" = "decision_tree"
                ),
                selected = "logistic_regression"
              ),
              
              actionButton("btn_predict", "Assess Health Status", class = "btn-primary btn-block", icon = icon("calculator"))
            )
          ),
          
          # Right Column: Prediction Results & Breakdown
          column(
            width = 7,
            # Dynamic Risk Status Card
            uiOutput("risk_result_card"),
            
            # Biomarkers Assessment Checklist
            box(
              width = NULL,
              title = "Biomarker & Vital Sign Evaluation",
              status = "info",
              solidHeader = TRUE,
              tableOutput("biomarkers_table")
            ),
            
            # Personalized Recommendations Box
            box(
              width = NULL,
              title = "Personalized Preventive Health Recommendations",
              status = "success",
              solidHeader = TRUE,
              uiOutput("recommendations_ui"),
              downloadButton("btn_download_report", "Download Assessment Summary", class = "btn-sm btn-default pull-right")
            )
          )
        )
      ),
      
      # ------------------------------------------------------------------------
      # TAB 2: MODEL PERFORMANCE & COMPARISON
      # ------------------------------------------------------------------------
      tabItem(
        tabName = "models",
        fluidRow(
          valueBox("94.00%", "Logistic Regression Accuracy", icon = icon("check-circle"), color = "green", width = 3),
          valueBox("91.30%", "Optimal F1-Score", icon = icon("bullseye"), color = "aqua", width = 3),
          valueBox("0.9866", "ROC-AUC Score", icon = icon("chart-area"), color = "purple", width = 3),
          valueBox("4 Models", "Benchmarked Classifiers", icon = icon("cogs"), color = "yellow", width = 3)
        ),
        
        fluidRow(
          box(
            width = 7,
            title = "Comparative Performance Metrics (Holdout Test Set)",
            status = "primary",
            solidHeader = TRUE,
            DTOutput("models_metrics_table")
          ),
          box(
            width = 5,
            title = "Feature Importance (Random Forest)",
            status = "primary",
            solidHeader = TRUE,
            plotOutput("feature_importance_plot", height = "300px")
          )
        ),
        
        fluidRow(
          box(
            width = 6,
            title = "ROC Curves Comparison",
            status = "info",
            solidHeader = TRUE,
            plotlyOutput("roc_plotly", height = "360px")
          ),
          box(
            width = 6,
            title = "Confusion Matrix & Accuracy Breakdown",
            status = "info",
            solidHeader = TRUE,
            selectInput("cm_model_choice", "Inspect Model Confusion Matrix:",
                        choices = c("Logistic Regression" = 1, "Decision Tree" = 2, "Random Forest" = 3, "Support Vector Machine" = 4),
                        selected = 1),
            tableOutput("confusion_matrix_table"),
            uiOutput("cm_interpretation")
          )
        )
      ),
      
      # ------------------------------------------------------------------------
      # TAB 3: DATASET & EDA
      # ------------------------------------------------------------------------
      tabItem(
        tabName = "eda",
        fluidRow(
          box(
            width = 12,
            title = "Interactive Clinical Dataset Explorer",
            status = "primary",
            solidHeader = TRUE,
            fluidRow(
              column(4, selectInput("eda_filter_risk", "Filter by Risk Status:", choices = c("All", "Low_Risk", "High_Risk"))),
              column(4, selectInput("eda_filter_gender", "Filter by Gender:", choices = c("All", "Male", "Female"))),
              column(4, downloadButton("btn_export_csv", "Export Filtered CSV", class = "btn-default", style = "margin-top: 25px;"))
            ),
            hr(),
            DTOutput("raw_data_table")
          )
        ),
        fluidRow(
          box(
            width = 6,
            title = "Blood Pressure vs Age Distribution",
            status = "info",
            solidHeader = TRUE,
            plotlyOutput("eda_scatter_bp", height = "320px")
          ),
          box(
            width = 6,
            title = "Blood Glucose vs BMI Correlation",
            status = "info",
            solidHeader = TRUE,
            plotlyOutput("eda_scatter_glucose", height = "320px")
          )
        )
      ),
      
      # ------------------------------------------------------------------------
      # TAB 4: CLINICAL GUIDELINES & ABOUT
      # ------------------------------------------------------------------------
      tabItem(
        tabName = "about",
        fluidRow(
          box(
            width = 12,
            title = "About the Health Status Prediction Project",
            status = "primary",
            solidHeader = TRUE,
            tags$div(
              h4("Project Purpose & Overview"),
              p("Early identification of potential health risks plays an indispensable role in preventing serious chronic conditions such as cardiovascular diseases, type 2 diabetes, and metabolic disorders. This project develops a machine learning-based system in R that analyzes clinical biometrics and lifestyle parameters to provide preliminary health risk assessments."),
              
              h4("Clinical Parameter Reference Standards"),
              tags$ul(
                tags$li(strong("Blood Pressure (AHA/ACC Standards): "), "Normal (<120/80 mmHg), Elevated (120-129/<80 mmHg), Hypertension Stage 1 (130-139/80-89 mmHg), Hypertension Stage 2 (≥140/90 mmHg)."),
                tags$li(strong("Fasting Blood Glucose (ADA Standards): "), "Normal (<100 mg/dL), Prediabetes (100–125 mg/dL), Diabetes (≥126 mg/dL)."),
                tags$li(strong("Body Mass Index (WHO Standards): "), "Underweight (<18.5), Normal weight (18.5–24.9), Overweight (25.0–29.9), Obese (≥30.0 kg/m²)."),
                tags$li(strong("Total Cholesterol: "), "Desirable (<200 mg/dL), Borderline High (200–239 mg/dL), High (≥240 mg/dL)."),
                tags$li(strong("Physical Activity: "), "Recommended ≥150 minutes of moderate-intensity physical activity per week (≥20–30 mins/day).")
              ),
              
              h4("Model Architecture & Methodology"),
              p("The predictive system evaluates 4 machine learning classifiers trained and validated on a stratified holdout test set:"),
              tags$ol(
                tags$li(strong("Logistic Regression: "), "Achieves top calibrated classification performance (94.00% Accuracy, 0.9866 AUC)."),
                tags$li(strong("Support Vector Machine (SVM): "), "Radial Basis Function (RBF) kernel mapping complex non-linear clinical boundaries (92.17% Accuracy)."),
                tags$li(strong("Random Forest: "), "Ensemble of 300 de-correlated decision trees offering high robustness and feature importance rankings (91.83% Accuracy)."),
                tags$li(strong("Decision Tree (CART): "), "Interpretable hierarchical decision rules (84.00% Accuracy).")
              ),
              
              hr(),
              p(em("Developed with R, Shiny, Shinydashboard, and Scikit-compatible R ML packages (randomForest, e1071, rpart, pROC)."))
            )
          )
        )
      )
    )
  )
)

# ------------------------------------------------------------------------------
# Server Logic
# ------------------------------------------------------------------------------
server <- function(input, output, session) {
  
  # Reactive prediction function
  prediction_reactive <- reactive({
    # Dependency on Predict button or initial load
    input$btn_predict
    
    # Collect inputs
    isolate({
      patient_data <- data.frame(
        age = input$age,
        gender = factor(input$gender, levels = c("Female", "Male")),
        systolic_bp = input$systolic_bp,
        diastolic_bp = input$diastolic_bp,
        blood_glucose = input$blood_glucose,
        bmi = input$bmi,
        heart_rate = input$heart_rate,
        physical_activity_min = input$physical_activity_min,
        sleep_hours = input$sleep_hours,
        cholesterol_total = input$cholesterol_total,
        smoking_status = factor(input$smoking_status, levels = c("Never", "Former", "Current")),
        alcohol_intake = factor(input$alcohol_intake, levels = c("None", "Moderate", "Heavy")),
        stress_level = factor(input$stress_level, levels = c("Low", "Medium", "High")),
        stringsAsFactors = FALSE
      )
      
      model_type <- input$selected_model
    })
    
    # Model inference
    if (model_type == "logistic_regression") {
      prob <- predict(bundle$logistic_regression, newdata = patient_data, type = "response")
    } else if (model_type == "random_forest") {
      prob <- predict(bundle$random_forest, newdata = patient_data, type = "prob")[, "High_Risk"]
    } else if (model_type == "svm") {
      pred <- predict(bundle$svm, newdata = patient_data, probability = TRUE)
      prob <- attr(pred, "probabilities")[, "High_Risk"]
    } else if (model_type == "decision_tree") {
      prob <- predict(bundle$decision_tree, newdata = patient_data, type = "prob")[, "High_Risk"]
    }
    
    risk_pct <- as.numeric(prob) * 100
    
    if (risk_pct < 35) {
      category <- "Low Risk (Healthy Profile)"
      class_style <- "risk-low"
      icon_name <- "check-circle"
    } else if (risk_pct < 65) {
      category <- "Moderate Risk (Needs Monitoring)"
      class_style <- "risk-mod"
      icon_name <- "exclamation-circle"
    } else {
      category <- "High Risk (Medical Consultation Recommended)"
      class_style <- "risk-high"
      icon_name <- "heartbeat"
    }
    
    list(
      patient = patient_data,
      risk_pct = round(risk_pct, 1),
      category = category,
      class_style = class_style,
      icon_name = icon_name,
      model_type = model_type
    )
  })
  
  # Render Risk Result Card
  output$risk_result_card <- renderUI({
    res <- prediction_reactive()
    
    div(
      class = paste("risk-card", res$class_style),
      h3(style = "margin-top: 0; font-weight: bold;", icon(res$icon_name), " Predicted Health Status: ", res$category),
      h4(sprintf("Calculated Risk Score: %.1f%%", res$risk_pct)),
      p(style = "font-size: 13px; opacity: 0.95; margin-bottom: 0;",
        sprintf("Assessed via %s classifier", gsub("_", " ", toupper(res$model_type))))
    )
  })
  
  # Render Biomarkers Table
  output$biomarkers_table <- renderTable({
    res <- prediction_reactive()
    p <- res$patient
    
    # Evaluate parameters against clinical guidelines
    bp_status <- if (p$systolic_bp >= 140 || p$diastolic_bp >= 90) "Hypertension Stage 2 (High)" else if (p$systolic_bp >= 130 || p$diastolic_bp >= 80) "Hypertension Stage 1 (Elevated)" else "Optimal / Normal (<120/80)"
    glucose_status <- if (p$blood_glucose >= 126) "Diabetes Range (≥126)" else if (p$blood_glucose >= 100) "Prediabetes Range (100-125)" else "Normal (<100 mg/dL)"
    bmi_status <- if (p$bmi >= 30) "Obese (≥30)" else if (p$bmi >= 25) "Overweight (25-29.9)" else if (p$bmi >= 18.5) "Normal Weight (18.5-24.9)" else "Underweight (<18.5)"
    chol_status <- if (p$cholesterol_total >= 240) "High (≥240)" else if (p$cholesterol_total >= 200) "Borderline High (200-239)" else "Desirable (<200 mg/dL)"
    act_status <- if (p$physical_activity_min >= 30) "Active (≥30 min/day)" else if (p$physical_activity_min >= 15) "Moderate (15-29 min/day)" else "Sedentary (<15 min/day)"
    sleep_status <- if (p$sleep_hours >= 7 && p$sleep_hours <= 9) "Optimal (7-9 hrs)" else "Suboptimal (<7 or >9 hrs)"
    
    data.frame(
      Parameter = c("Blood Pressure", "Fasting Glucose", "Body Mass Index", "Total Cholesterol", "Physical Activity", "Sleep Duration"),
      User_Value = c(
        sprintf("%d / %d mmHg", p$systolic_bp, p$diastolic_bp),
        sprintf("%d mg/dL", p$blood_glucose),
        sprintf("%.1f kg/m²", p$bmi),
        sprintf("%d mg/dL", p$cholesterol_total),
        sprintf("%d min/day", p$physical_activity_min),
        sprintf("%.1f hours/day", p$sleep_hours)
      ),
      Clinical_Interpretation = c(bp_status, glucose_status, bmi_status, chol_status, act_status, sleep_status)
    )
  }, striped = TRUE, hover = TRUE, bordered = TRUE)
  
  # Render Personalized Recommendations
  output$recommendations_ui <- renderUI({
    res <- prediction_reactive()
    p <- res$patient
    
    recs <- list()
    
    if (p$systolic_bp >= 130 || p$diastolic_bp >= 80) {
      recs <- c(recs, "Blood Pressure: Adopt a DASH diet low in sodium (<2,300 mg/day) and monitor blood pressure weekly.")
    }
    if (p$blood_glucose >= 100) {
      recs <- c(recs, "Blood Sugar: Limit refined sugars and carbohydrates; consider a routine HbA1c screening.")
    }
    if (p$bmi >= 25) {
      recs <- c(recs, "Weight Management: Aim for a gradual 5-10% weight reduction via caloric balance and strength training.")
    }
    if (p$cholesterol_total >= 200) {
      recs <- c(recs, "Lipid Profile: Increase soluble fiber (oats, legumes) and replace saturated fats with healthy fats (olive oil, nuts).")
    }
    if (p$physical_activity_min < 30) {
      recs <- c(recs, "Physical Activity: Gradually increase aerobic activity to at least 150 minutes per week (brisk walking, cycling).")
    }
    if (as.character(p$smoking_status) == "Current") {
      recs <- c(recs, "Tobacco Cessation: Smoking significantly elevates cardiovascular risk; explore smoking cessation support programs.")
    }
    if (as.character(p$stress_level) == "High") {
      recs <- c(recs, "Stress Reduction: Incorporate mindfulness, meditation, or breathing exercises to lower cortisol levels.")
    }
    if (p$sleep_hours < 7) {
      recs <- c(recs, "Sleep Hygiene: Target 7-8 hours of uninterrupted sleep; maintain a consistent bedtime schedule.")
    }
    
    if (length(recs) == 0) {
      recs <- c("Excellent! All clinical and lifestyle parameters are currently in optimal ranges. Maintain your healthy habits!")
    }
    
    tagList(
      tags$ul(
        lapply(recs, function(r) tags$li(style = "margin-bottom: 6px;", r))
      )
    )
  })
  
  # Download Patient Summary Report
  output$btn_download_report <- downloadHandler(
    filename = function() {
      paste0("health_assessment_report_", Sys.Date(), ".txt")
    },
    content = function(file) {
      res <- prediction_reactive()
      p <- res$patient
      
      lines <- c(
        "=================================================================",
        "            HEALTH STATUS PREDICTION - ASSESSMENT REPORT         ",
        "=================================================================",
        paste0("Assessment Date: ", Sys.time()),
        paste0("Algorithm Model: ", res$model_type),
        paste0("Predicted Risk:  ", res$category),
        paste0("Risk Score:      ", res$risk_pct, "%"),
        "-----------------------------------------------------------------",
        "PATIENT BIOMETRICS & VITALS:",
        paste0("  - Age:                 ", p$age, " years"),
        paste0("  - Gender:              ", p$gender),
        paste0("  - Blood Pressure:      ", p$systolic_bp, "/", p$diastolic_bp, " mmHg"),
        paste0("  - Fasting Glucose:     ", p$blood_glucose, " mg/dL"),
        paste0("  - Total Cholesterol:   ", p$cholesterol_total, " mg/dL"),
        paste0("  - BMI:                 ", p$bmi, " kg/m2"),
        paste0("  - Heart Rate:          ", p$heart_rate, " bpm"),
        paste0("  - Physical Activity:   ", p$physical_activity_min, " mins/day"),
        paste0("  - Sleep:               ", p$sleep_hours, " hours/night"),
        paste0("  - Smoking Status:      ", p$smoking_status),
        paste0("  - Alcohol Intake:      ", p$alcohol_intake),
        paste0("  - Stress Level:        ", p$stress_level),
        "-----------------------------------------------------------------",
        "DISCLAIMER: For educational/preventive awareness only.",
        "================================================================="
      )
      writeLines(lines, file)
    }
  )
  
  # TAB 2: Model Performance Outputs
  output$models_metrics_table <- renderDT({
    datatable(
      metrics_df,
      options = list(dom = 't', paging = FALSE, ordering = FALSE),
      rownames = FALSE
    ) %>% formatStyle(
      'Accuracy_Pct',
      background = styleColorBar(c(0, 100), '#93c5fd'),
      backgroundSize = '98% 88%',
      backgroundRepeat = 'no-repeat',
      backgroundPosition = 'center'
    )
  })
  
  output$feature_importance_plot <- renderPlot({
    imp_df <- bundle$importance_df
    ggplot(imp_df[1:8, ], aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
      geom_col(fill = "#3b82f6", width = 0.65) +
      coord_flip() +
      theme_minimal() +
      labs(x = NULL, y = "Mean Decrease Gini") +
      theme(axis.text = element_text(size = 11, face = "bold"))
  })
  
  output$roc_plotly <- renderPlotly({
    roc_df_list <- lapply(bundle$eval_results, function(ev) {
      data.frame(
        FPR = 1 - ev$roc$specificities,
        TPR = ev$roc$sensitivities,
        Model = sprintf("%s (AUC=%.3f)", ev$model, ev$auc)
      )
    })
    roc_all <- do.call(rbind, roc_df_list)
    
    p <- ggplot(roc_all, aes(x = FPR, y = TPR, color = Model)) +
      geom_line(linewidth = 1) +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
      theme_minimal() +
      labs(x = "False Positive Rate", y = "True Positive Rate")
    
    ggplotly(p)
  })
  
  output$confusion_matrix_table <- renderTable({
    idx <- as.numeric(input$cm_model_choice)
    ev <- bundle$eval_results[[idx]]
    cm <- ev$cm
    
    data.frame(
      Actual_Class = c("Actual: Low Risk", "Actual: High Risk"),
      Pred_Low_Risk = c(cm["Low_Risk", "Low_Risk"], cm["High_Risk", "Low_Risk"]),
      Pred_High_Risk = c(cm["Low_Risk", "High_Risk"], cm["High_Risk", "High_Risk"])
    )
  }, striped = TRUE, bordered = TRUE)
  
  output$cm_interpretation <- renderUI({
    idx <- as.numeric(input$cm_model_choice)
    ev <- bundle$eval_results[[idx]]
    
    p(
      strong("Model: "), ev$model, " | ",
      strong("Accuracy: "), sprintf("%.2f%%", ev$accuracy), " | ",
      strong("Precision: "), sprintf("%.2f%%", ev$precision), " | ",
      strong("Recall: "), sprintf("%.2f%%", ev$recall), " | ",
      strong("F1-Score: "), sprintf("%.2f%%", ev$f1_score)
    )
  })
  
  # TAB 3: EDA & Dataset
  filtered_data <- reactive({
    data <- health_df
    if (input$eda_filter_risk != "All") {
      data <- data[data$health_risk == input$eda_filter_risk, ]
    }
    if (input$eda_filter_gender != "All") {
      data <- data[data$gender == input$eda_filter_gender, ]
    }
    data
  })
  
  output$raw_data_table <- renderDT({
    datatable(
      filtered_data()[, 1:14],
      options = list(pageLength = 6, scrollX = TRUE),
      rownames = FALSE
    )
  })
  
  output$btn_export_csv <- downloadHandler(
    filename = function() { "filtered_health_data.csv" },
    content = function(file) {
      write.csv(filtered_data(), file, row.names = FALSE)
    }
  )
  
  output$eda_scatter_bp <- renderPlotly({
    p <- ggplot(health_df, aes(x = age, y = systolic_bp, color = health_risk)) +
      geom_point(alpha = 0.6, size = 1.8) +
      scale_color_manual(values = c("Low_Risk" = "#10b981", "High_Risk" = "#ef4444")) +
      theme_minimal() +
      labs(x = "Age (Years)", y = "Systolic Blood Pressure (mmHg)")
    ggplotly(p)
  })
  
  output$eda_scatter_glucose <- renderPlotly({
    p <- ggplot(health_df, aes(x = bmi, y = blood_glucose, color = health_risk)) +
      geom_point(alpha = 0.6, size = 1.8) +
      scale_color_manual(values = c("Low_Risk" = "#10b981", "High_Risk" = "#ef4444")) +
      theme_minimal() +
      labs(x = "Body Mass Index (BMI)", y = "Fasting Blood Glucose (mg/dL)")
    ggplotly(p)
  })
}

# ------------------------------------------------------------------------------
# Launch Application
# ------------------------------------------------------------------------------
shinyApp(ui = ui, server = server)
