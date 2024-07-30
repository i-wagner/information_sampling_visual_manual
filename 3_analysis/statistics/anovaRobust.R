library('readr') # For reading .csvs
#library("robustlmm")
source("/Users/ilja/Dropbox/12_work/mr_informationSamplingVisualManual/3_analysis/statistics/Rallfun-v43.txt")

PATH_TO_FILES = "/Users/ilja/Dropbox/12_work/mr_informationSamplingVisualManual/3_analysis/statistics/"
N_GROUPS_FACTOR_ONE = 2
N_GROUPS_FACTOR_TWO = 2
N_BOOTSAMPLES = 10000


##### Perceptual performance
file_path <- paste(PATH_TO_FILES, "perceptualPerformance_long.csv", sep="")
data <- read_csv(file_path)

# See page 18 of 
# Wilcox, Introduction to Robust Estimation and Hypothesis Testing.
data_formated = fac2list(data["Dv"], data[c("Exp", "Targ")])

# See page 470 of
# Wilcox, Introduction to Robust Estimation and Hypothesis Testing.
wwtrim(N_GROUPS_FACTOR_ONE, N_GROUPS_FACTOR_TWO, data_formated)
wwtrimbt(N_GROUPS_FACTOR_ONE, N_GROUPS_FACTOR_TWO, data_formated, nboot=N_BOOTSAMPLES)


##### Planning times
file_path <- paste(PATH_TO_FILES, "planningTime_long.csv", sep="")
data <- read_csv(file_path)
data_formated = fac2list(data["Dv"], data[c("Exp", "Targ")])

wwtrim(N_GROUPS_FACTOR_ONE, N_GROUPS_FACTOR_TWO, data_formated)
wwtrimbt(N_GROUPS_FACTOR_ONE, N_GROUPS_FACTOR_TWO, data_formated, nboot=N_BOOTSAMPLES)


##### Inspection times
file_path <- paste(PATH_TO_FILES, "inspectionTime_long.csv", sep="")
data <- read_csv(file_path)
data_formated = fac2list(data["Dv"], data[c("Exp", "Targ")])

wwtrim(N_GROUPS_FACTOR_ONE, N_GROUPS_FACTOR_TWO, data_formated)
wwtrimbt(N_GROUPS_FACTOR_ONE, N_GROUPS_FACTOR_TWO, data_formated, nboot=N_BOOTSAMPLES)


##### Responose times
file_path <- paste(PATH_TO_FILES, "responseTime_long.csv", sep="")
data <- read_csv(file_path)
data_formated = fac2list(data["Dv"], data[c("Exp", "Targ")])

wwtrim(N_GROUPS_FACTOR_ONE, N_GROUPS_FACTOR_TWO, data_formated)
wwtrimbt(N_GROUPS_FACTOR_ONE, N_GROUPS_FACTOR_TWO, data_formated, nboot=N_BOOTSAMPLES)


##### Gaze shifts on chosen
file_path <- paste(PATH_TO_FILES, "propOnChosen_long.csv", sep="")
data <- read_csv(file_path)
data_formated = fac2list(data["Dv"], data[c("Exp", "Targ")])

wwtrim(N_GROUPS_FACTOR_ONE, N_GROUPS_FACTOR_TWO, data_formated)
wwtrimbt(N_GROUPS_FACTOR_ONE, N_GROUPS_FACTOR_TWO, data_formated, nboot=N_BOOTSAMPLES)


##### Gaze shifts on easy set
file_path <- paste(PATH_TO_FILES, "propOnEasy_long.csv", sep="")
data <- read_csv(file_path)
data_formated = fac2list(data["Dv"], data[c("Exp", "Targ")])

wwtrim(N_GROUPS_FACTOR_ONE, N_GROUPS_FACTOR_TWO, data_formated)
wwtrimbt(N_GROUPS_FACTOR_ONE, N_GROUPS_FACTOR_TWO, data_formated, nboot=N_BOOTSAMPLES)


##### Gaze shifts on smaller set
file_path <- paste(PATH_TO_FILES, "propOnSmaller_long.csv", sep="")
data <- read_csv(file_path)
data_formated = fac2list(data["Dv"], data[c("Exp", "Targ")])

wwtrim(N_GROUPS_FACTOR_ONE, N_GROUPS_FACTOR_TWO, data_formated)
wwtrimbt(N_GROUPS_FACTOR_ONE, N_GROUPS_FACTOR_TWO, data_formated, nboot=N_BOOTSAMPLES)


##### Gaze shifts on closest element
file_path <- paste(PATH_TO_FILES, "propOnClosest_long.csv", sep="")
data <- read_csv(file_path)
data_formated = fac2list(data["Dv"], data[c("Exp", "Targ")])

wwtrim(N_GROUPS_FACTOR_ONE, N_GROUPS_FACTOR_TWO, data_formated)
wwtrimbt(N_GROUPS_FACTOR_ONE, N_GROUPS_FACTOR_TWO, data_formated, nboot=N_BOOTSAMPLES)


##### Gaze shift latencies
file_path <- paste(PATH_TO_FILES, "latencies_long.csv", sep="")
data <- read_csv(file_path)
data_formated = fac2list(data["Dv"], data[c("Exp", "Targ")])

wwtrim(N_GROUPS_FACTOR_ONE, N_GROUPS_FACTOR_TWO, data_formated)
wwtrimbt(N_GROUPS_FACTOR_ONE, N_GROUPS_FACTOR_TWO, data_formated, nboot=N_BOOTSAMPLES)
