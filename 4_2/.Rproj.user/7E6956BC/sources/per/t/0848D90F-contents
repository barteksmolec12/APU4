install.packages("randomForest")
install.packages("e1071")
install.packages("party")
install.packages("mlr")
install.packages("rFerns")

library(randomForest)
library(rpart.plot)
library(party)
library(mlr)
library(rFerns)



ipady_rtv=read.csv("ramka_dane.csv")
ipady_rtv$tablet = factor(ipady_rtv$tablet)
ipady_rtv$ocena = factor(ipady_rtv$ocena)

summarizeColumns(ipady_rtv)

rdesc = makeResampleDesc("CV", iters = 10)

task = makeClassifTask(id = deparse(substitute(ipady_rtv)), ipady_rtv, "ocena",
                       weights = NULL, blocking = NULL, coordinates = NULL,
                       positive = NA_character_, fixup.data = "warn", check.data = TRUE)
lrns <- makeLearners(c("rpart", "C50", "ctree", "naiveBayes", "randomForest"), type = "classif")


bmr <- benchmark(learners = lrns, tasks = task, rdesc, models = TRUE, measures = list(acc, ber))
p = getBMRPredictions(bmr)
plotBMRSummary(bmr)

