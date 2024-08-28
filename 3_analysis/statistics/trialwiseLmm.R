library('readr') # For reading .csvs
library('nlme') # For LMM
library('summarytools') # For summary statistics
library("performance") # For model check
library("DHARMa") # For model check
library("ggeffects")
#library("lme4")
library(ggplot2)

# Clear the console and the global environment
rm(list = ls())
cat("\014")

# Load data from each file
file_path <- "/Users/ilja/Dropbox/12_work/mr_informationSamplingVisualManual/planningTime_withTrial.csv"
data <- read_csv(file_path)

# Define factors
data$SubNo <- factor(data$SubNo)
data$Exp <- factor(data$Exp, 
                   levels=c(1, 2),
                   labels=c('visual', 'manual'))
data$Targ <- factor(data$Targ, 
                    levels=c(1, 2),
                    labels=c('Easy', 'Difficult'))

# Fit linear mixed model
summary_stats <- aggregate(Dv~Exp*Targ, data, 
                           function(x) c(mean = mean(x), sd = sd(x)))
print(summary_stats)

ggplot(data = data, aes(x = Targ, y = Dv, group = Exp)) +
  geom_point() +
  theme_classic() +
  stat_summary(size = 2, colour = 'dark green') +
  stat_summary(size = 2, colour = 'dark green', geom = 'line')
# facet_wrap(~Exp)

# For temporal measures: log transformation seems to help
#
#vf1 <- varIdent(form = ~ 1 | Targ)
#vf2 <- varPower(form = ~ 1 | Targ)
#vf3 <- varIdent(form = ~ Exp | Targ)
#fit1 = lme(Dv ~ Exp * Targ, random=~1|SubNo/Exp/Targ, data=data, weights=vf3)
#fit2 = lme(Dv ~ Exp * Targ, random=~1|SubNo/Exp, data=data, weights=vf3)
#fit3 = lme(Dv ~ Exp * Targ, random=~1|SubNo, data=data, weights=vf3)
fit1 = lme(Dv ~ Exp * Targ, random=~1|SubNo/Exp/Targ, data=data)
fit2 = lme(Dv ~ Exp * Targ, random=~1|SubNo/Exp, data=data)
fit3 = lme(Dv ~ Exp * Targ, random=~1|SubNo, data=data)
anova(fit1, fit2, fit3)

qqnorm(resid(fit3))
qqline(resid(fit3))
performance::check_model(fit3, dot_size=2, line_size=0.8, panel=TRUE, check="all")
CookD(fit3)
anova(fit3)
