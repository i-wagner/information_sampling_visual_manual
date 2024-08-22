#library(lme4)
#library(nlme)
#library(reshape)
#library(robustbase)
#library(robustlmm)
library(WRS2)

PATH_TO_FILES = "/Users/ilja/Dropbox/12_work/mr_informationSamplingVisualManual/3_analysis/statistics/"


##### Completed trials
filepath = paste(PATH_TO_FILES, "completedTrials_short.csv", sep="")
data <- read.csv(filepath)

# Single visual vs. single manual
t.test(data$single_visual, data$single_manual, paired=T) # Non-robust
yuend(data$single_visual, data$single_manual, tr = 0.2) # Robust

# Single visual vs. double visual
t.test(data$single_visual, data$double_visual, paired=T) # Non-robust
yuend(data$single_visual, data$double_visual, tr = 0.2) # Robust

# Single manual vs. double manual
t.test(data$single_manual, data$double_manual, paired=T) # Non-robust
yuend(data$single_manual, data$double_manual, tr = 0.2) # Robust


##### Bonus payout
filepath = paste(PATH_TO_FILES, "bonusPayout_short.csv", sep="")
data <- read.csv(filepath)

# Single visual vs. single manual
t.test(data$single_visual, data$single_manual, paired=T) # Non-robust
yuend(data$single_visual, data$single_manual, tr = 0.2) # Robust

# Single visual vs. double visual
t.test(data$single_visual, data$double_visual, paired=T) # Non-robust
yuend(data$single_visual, data$double_visual, tr = 0.2) # Robust

# Single manual vs. double manual
t.test(data$single_manual, data$double_manual, paired=T) # Non-robust
yuend(data$single_manual, data$double_manual, tr = 0.2) # Robust


##### Regression parameter
filepath = paste(PATH_TO_FILES, "regressionPar_short.csv", sep="")
data <- read.csv(filepath)

# Intercepts visual vs. Intercepts manual
t.test(data$intercepts_visual, data$intercepts_manual, paired=T) # Non-robust
yuend(data$intercepts_visual, data$intercepts_manual, tr = 0.2) # Robust

# Slopes visual vs. Intercepts manual
t.test(data$slopes_visual, data$slopes_manual, paired=T) # Non-robust
yuend(data$slopes_visual, data$slopes_manual, tr = 0.2) # Robust


##### Sigmoid parameter
filepath = paste(PATH_TO_FILES, "sigmoidPar_short.csv", sep="")
data <- read.csv(filepath)

# Means visual vs. Means manual
t.test(data$intercepts_visual, data$means_manual, paired=T) # Non-robust
yuend(data$intercepts_visual, data$means_manual, tr = 0.2) # Robust

# Slopes visual vs. Intercepts manual
t.test(data$slopes_visual, data$slopes_manual, paired=T) # Non-robust
yuend(data$slopes_visual, data$slopes_manual, tr = 0.2) # Robust


##### Gain per time
filepath = paste(PATH_TO_FILES, "gainPerTime_short.csv", sep="")
data <- read.csv(filepath)

# Empirical double visual vs. ideal observer double visual
t.test(data$emp_double_visual, data$idealObs_double_visual, paired=T) # Non-robust
yuend(data$emp_double_visual, data$idealObs_double_visual, tr = 0.2) # Robust

# Empirical double manual vs. ideal observer double manual
t.test(data$emp_double_manual, data$idealObs_double_manual, paired=T) # Non-robust
yuend(data$emp_double_manual, data$idealObs_double_manual, tr = 0.2) # Robust


##### Decision noise
filepath = paste(PATH_TO_FILES, "decisionNoise_short.csv", sep="")
data <- read.csv(filepath)

t.test(data$double_visual, data$double_manual, paired=T) # Non-robust
yuend(data$double_visual, data$double_manual, tr = 0.2) # Robust


##### Fixation noise
filepath = paste(PATH_TO_FILES, "fixationNoise_short.csv", sep="")
data <- read.csv(filepath)

t.test(data$double_visual, data$double_manual, paired=T) # Non-robust
yuend(data$double_visual, data$double_manual, tr = 0.2) # Robust
