# R code
# PMML generation for audit, Naive Bayes model
install.packages('e1071')
install.packages('pmml')
library(e1071)
library('pmml')

audit.train <- read.table("~/work/sumatosoft/rubyconf2018ML/audit.csv", header = TRUE, sep = ",")
audit.train$TARGET_Adjusted <- as.factor(audit.train$TARGET_Adjusted)

# Train Naive Bayes model

audit.bn <- naiveBayes(TARGET_Adjusted ~ . - ID - Age - Employment - Education - Marital - Occupation - Income - Gender - Deductions - Hours - IGNORE_Accounts - RISK_Adjustment,
                      data = audit.train)

# Generate pmml from model
pmml <- pmml(audit.bn, dataset=audit.train, predictedField="TARGET_Adjusted")
saveXML(pmml, '~/work/sumatosoft/rubyconf2018ML/r_bayes.pmml')
