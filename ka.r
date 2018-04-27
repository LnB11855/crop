if (!require("pacman")) install.packages("pacman")
pacman::p_load(knitr, pryr, tidyverse, data.table, lubridate, caret, tictoc, DescTools, lightgbm)
library(xgboost)
set.seed(84)
options(scipen = 9999, warn = -1, digits= 4)
train_path <- "D:/IE583/kaggle challenge/train.csv"
chunk_rows <- 30817315
cat("
    --------------------------------------------------------------
    Part 1: Data processing + modelling  
    --------------------------------------------------------------
    ")
cat("
    -------------------------------------------
    Processing chunk 1  
    -------------------------------------------
    ")

total_rows <- 184903890
skip_rows  <- total_rows - chunk_rows
# features builder 
add_features <- function(df) {
  df <- as.data.table(df)
  df=df[order(click_time)]
  cat("Addidng interaction features..", "\n")
  df[, C_a:=.N,   by=list(app)]
  df[, C_i_a:=.N,   by=list(ip,app)]
  df[, C_i_a:=.N,   by=list(ip,app)]
  df[, C_i_a_d_h:=.N,   by=list(ip,app,day,hour)]
  df[, C_i_a_o_d_h:=.N,   by=list(ip,app,os,day,hour)]
  cat("Addidng next click time features..", "\n")
  #S=Sys.time()
  df[, N_i_a_o:=as.POSIXct(shift(click_time,type="lead"))-as.POSIXct(click_time), by=list(ip, app,os)]
  df$N_i_a_o=as.numeric(df$N_i_a_o,units="secs")
  df[, N_a_o_c:=as.POSIXct(shift(click_time,type="lead"))-as.POSIXct(click_time), by=list(app,os,channel)]
  df$N_a_o_c=as.numeric(df$N_a_o_c,units="secs")
  #E=Sys.time()
  #print(E-S)
  cat("Addidng unique features..", "\n")
  df[, U_i_c:= uniqueN(channel)/.N,   by=list(ip)]
  df[, U_i_o_c:= uniqueN(channel)/.N,   by=list(ip,os)]
  cat("Addidng variance features..", "\n") 
  df[, V_i_a_d_h:= var(hour),   by=list(ip,app,day)]
  df <- as.data.frame(df) %>% 
    select(-c(ip, day,click_time))
  return(df)
}
train <- fread(train_path, skip=skip_rows, nrows= chunk_rows, 
               colClasses=list(numeric=1:5), showProgress = FALSE, 
               col.names = c("ip", "app", "device", "os", "channel", "click_time", 
                             "attributed_time", "is_attributed")) %>% 
  select(-c(attributed_time)) %>%
  mutate(day = Weekday(click_time), hour = hour(click_time))
train.index <- createDataPartition(train$is_attributed, p = 0.9, list = FALSE)
dtrain <- train[ train.index,]
dvalid <- train[-train.index,]
rm(train, train.index, total_rows, skip_rows)
invisible(gc())

dtrain <- add_features(dtrain)
dvalid <- add_features(dvalid)

xgbtrain <- xgb.DMatrix(data = data.matrix(dtrain[, colnames(dtrain) != "is_attributed"]), label = dtrain$is_attributed,categorical_feature = c("app", "device", "os", "channel", "hour"))
xgbvalid <- xgb.DMatrix(data = data.matrix(dvalid[, colnames(dtrain) != "is_attributed"]), label = dvalid$is_attributed,categorical_feature = c("app", "device", "os", "channel", "hour"))
p <- list(objective = "binary:logistic",
          booster = "gbtree",
          eval_metric = "auc",
          nthread = 8,
          eta = 0.07,
          max_depth = 4,
          min_child_weight = 24,
          gamma = 36.7126,
          subsample = 0.9821,
          colsample_bytree = 0.3929,
          colsample_bylevel = 0.6818,
          alpha = 72.7519,
          lambda = 5.4826,
          max_delta_step = 5.7713,
          scale_pos_weight = 99.7,
          nrounds = 1000)

m_xgb <- xgb.train(p, xgbtrain, p$nrounds, list(val = xgbvalid), print_every_n = 50, early_stopping_rounds = 150)

imp <- xgb.importance(model=m_xgb)

xgb.plot.importance(imp,measure = "Gain")
dtrain <- lgb.Dataset(data  = as.matrix(dtrain[, colnames(dtrain) != "is_attributed"]),
                      label = dtrain$is_attributed, 
                      categorical_feature = c("app", "device", "os", "channel", "hour"))


dvalid <- lgb.Dataset(data  = as.matrix(dvalid[, colnames(dvalid) != "is_attributed"]),
                      label = dvalid$is_attributed, 
                      categorical_feature = c("app", "device", "os", "channel", "hour"))
invisible(gc())

params <- list(objective = "binary", metric = "auc", learning_rate= 0.1, num_leaves= 7,
               max_depth= 4, scale_pos_weight= 99.7)

model1 <- lgb.train(params, dtrain, valids = list(validation = dvalid), nthread = 4,
                   nrounds = 1000,verbose= 1, early_stopping_rounds = 50, eval_freq = 100)


cat("--------------------------------", "\n")
cat("model 1 valid AUC: ", max(unlist(model1$record_evals[["validation"]][["auc"]][["eval"]])), "\n")
cat("--------------------------------", "\n")
lgb.save(model1, "model1")
lgb.plot.importance(lgb.importance(model1, percentage = TRUE), measure = "Gain")
rm(list=ls()) # removes everything from workspace

cat("
    -------------------------------------------
    Processing chunk 2  
    -------------------------------------------
    ")

if (!require("pacman")) install.packages("pacman")
pacman::p_load(knitr, pryr, tidyverse, data.table, lubridate, caret, tictoc, DescTools, lightgbm)
set.seed(84)
options(scipen = 9999, warn = -1, digits= 4)

train_path <- "D:/IE583/kaggle challenge/train.csv"

total_rows <- 154086575
chunk_rows <- 30817315
skip_rows  <- total_rows - chunk_rows

add_features <- function(df) {
  df <- as.data.table(df)
  df=df[order(click_time)]
  cat("Addidng interaction features..", "\n")
  df[, C_a:=.N,   by=list(app)]
  df[, C_i_a:=.N,   by=list(ip,app)]
  df[, C_i_a:=.N,   by=list(ip,app)]
  df[, C_i_a_d_h:=.N,   by=list(ip,app,day,hour)]
  df[, C_i_a_o_d_h:=.N,   by=list(ip,app,os,day,hour)]
  cat("Addidng next click time features..", "\n")
  #S=Sys.time()
  df[, N_i_a_o:=as.POSIXct(shift(click_time,type="lead"))-as.POSIXct(click_time), by=list(ip, app,os)]
  df$N_i_a_o=as.numeric(df$N_i_a_o,units="secs")
  df[, N_a_o_c:=as.POSIXct(shift(click_time,type="lead"))-as.POSIXct(click_time), by=list(app,os,channel)]
  df$N_a_o_c=as.numeric(df$N_a_o_c,units="secs")
  #E=Sys.time()
  #print(E-S)
  cat("Addidng unique features..", "\n")
  df[, U_i_c:= uniqueN(channel)/.N,   by=list(ip)]
  df[, U_i_o_c:= uniqueN(channel)/.N,   by=list(ip,os)]
  cat("Addidng variance features..", "\n") 
  df[, V_i_a_d_h:= var(hour),   by=list(ip,app,day)]
  df <- as.data.frame(df) %>% 
    select(-c(ip, day,click_time))
  return(df)
}

train <- fread(train_path, skip=skip_rows, nrows= chunk_rows, 
               colClasses=list(numeric=1:5), showProgress = FALSE, 
               col.names = c("ip", "app", "device", "os", "channel", "click_time", 
                             "attributed_time", "is_attributed")) %>% 
  select(-c(attributed_time)) %>%
  mutate(day = Weekday(click_time), hour = hour(click_time)) 


train.index <- createDataPartition(train$is_attributed, p = 0.9, list = FALSE)
dtrain <- train[ train.index,]
dvalid <- train[-train.index,]
rm(train, train.index, total_rows)
invisible(gc())
dtrain <- add_features(dtrain)
dvalid <- add_features(dvalid)

xgbtrain <- xgb.DMatrix(data = data.matrix(dtrain[, colnames(dtrain) != "is_attributed"]), label = dtrain$is_attributed,categorical_feature = c("app", "device", "os", "channel", "hour"))
xgbvalid <- xgb.DMatrix(data = data.matrix(dvalid[, colnames(dtrain) != "is_attributed"]), label = dvalid$is_attributed,categorical_feature = c("app", "device", "os", "channel", "hour"))
p <- list(objective = "binary:logistic",
          booster = "gbtree",
          eval_metric = "auc",
          nthread = 8,
          eta = 0.07,
          max_depth = 4,
          min_child_weight = 24,
          gamma = 36.7126,
          subsample = 0.9821,
          colsample_bytree = 0.3929,
          colsample_bylevel = 0.6818,
          alpha = 72.7519,
          lambda = 5.4826,
          max_delta_step = 5.7713,
          scale_pos_weight = 99.7,
          nrounds = 1000)

m_xgb <- xgb.train(p, xgbtrain, p$nrounds, list(val = xgbvalid), print_every_n = 50, early_stopping_rounds = 150)

imp <- xgb.importance(model=m_xgb)

xgb.plot.importance(imp,measure = "Gain")
dtrain <- lgb.Dataset(data  = as.matrix(dtrain[, colnames(dtrain) != "is_attributed"]),
                      label = dtrain$is_attributed, 
                      categorical_feature = c("app", "device", "os", "channel", "hour"))


dvalid <- lgb.Dataset(data  = as.matrix(dvalid[, colnames(dvalid) != "is_attributed"]),
                      label = dvalid$is_attributed, 
                      categorical_feature = c("app", "device", "os", "channel", "hour"))
invisible(gc())

params <- list(objective = "binary", metric = "auc", learning_rate= 0.1, num_leaves= 7,
               max_depth= 4, scale_pos_weight= 99.7)

model2 <- lgb.train(params, dtrain, valids = list(validation = dvalid), nthread = 4,
                    nrounds = 1000,verbose= 1, early_stopping_rounds = 50, eval_freq = 50)
cat("--------------------------------", "\n")
cat("model 2 valid AUC: ", max(unlist(model2$record_evals[["validation"]][["auc"]][["eval"]])), "\n")
cat("--------------------------------", "\n")
lgb.save(model2, "model2")
lgb.plot.importance(lgb.importance(model2, percentage = TRUE), measure = "Gain")
rm(list=ls())

cat("
    -------------------------------------------
    Processing chunk 3 
    -------------------------------------------
    ")

if (!require("pacman")) install.packages("pacman")
pacman::p_load(knitr, pryr, tidyverse, data.table, lubridate, caret, tictoc, DescTools, lightgbm)
set.seed(84)
options(scipen = 9999, warn = -1, digits= 4)

train_path <- "D:/IE583/kaggle challenge/train.csv"

total_rows <- 123269260
chunk_rows <- 30817315
skip_rows  <- total_rows - chunk_rows

add_features <- function(df) {
  df <- as.data.table(df)
  df=df[order(click_time)]
  cat("Addidng interaction features..", "\n")
  df[, C_a:=.N,   by=list(app)]
  df[, C_i_a:=.N,   by=list(ip,app)]
  df[, C_i_a:=.N,   by=list(ip,app)]
  df[, C_i_a_d_h:=.N,   by=list(ip,app,day,hour)]
  df[, C_i_a_o_d_h:=.N,   by=list(ip,app,os,day,hour)]
  cat("Addidng next click time features..", "\n")
  #S=Sys.time()
  df[, N_i_a_o:=as.POSIXct(shift(click_time,type="lead"))-as.POSIXct(click_time), by=list(ip, app,os)]
  df$N_i_a_o=as.numeric(df$N_i_a_o,units="secs")
  df[, N_a_o_c:=as.POSIXct(shift(click_time,type="lead"))-as.POSIXct(click_time), by=list(app,os,channel)]
  df$N_a_o_c=as.numeric(df$N_a_o_c,units="secs")
  #E=Sys.time()
  #print(E-S)
  cat("Addidng unique features..", "\n")
  df[, U_i_c:= uniqueN(channel)/.N,   by=list(ip)]
  df[, U_i_o_c:= uniqueN(channel)/.N,   by=list(ip,os)]
  cat("Addidng variance features..", "\n") 
  df[, V_i_a_d_h:= var(hour),   by=list(ip,app,day)]
  df <- as.data.frame(df) %>% 
    select(-c(ip, day,click_time))
  return(df)
}

train <- fread(train_path, skip=skip_rows, nrows= chunk_rows, 
               colClasses=list(numeric=1:5), showProgress = FALSE, 
               col.names = c("ip", "app", "device", "os", "channel", "click_time", 
                             "attributed_time", "is_attributed")) %>% 
  select(-c(attributed_time)) %>%
  mutate(day = Weekday(click_time), hour = hour(click_time)) 


train.index <- createDataPartition(train$is_attributed, p = 0.9, list = FALSE)
dtrain <- train[ train.index,]
dvalid <- train[-train.index,]
rm(train, train.index, total_rows)
invisible(gc())

dtrain <- add_features(dtrain)
dvalid <- add_features(dvalid)

xgbtrain <- xgb.DMatrix(data = data.matrix(dtrain[, colnames(dtrain) != "is_attributed"]), label = dtrain$is_attributed,categorical_feature = c("app", "device", "os", "channel", "hour"))
xgbvalid <- xgb.DMatrix(data = data.matrix(dvalid[, colnames(dtrain) != "is_attributed"]), label = dvalid$is_attributed,categorical_feature = c("app", "device", "os", "channel", "hour"))
p <- list(objective = "binary:logistic",
          booster = "gbtree",
          eval_metric = "auc",
          nthread = 8,
          eta = 0.07,
          max_depth = 4,
          min_child_weight = 24,
          gamma = 36.7126,
          subsample = 0.9821,
          colsample_bytree = 0.3929,
          colsample_bylevel = 0.6818,
          alpha = 72.7519,
          lambda = 5.4826,
          max_delta_step = 5.7713,
          scale_pos_weight = 99.7,
          nrounds = 1000)

m_xgb <- xgb.train(p, xgbtrain, p$nrounds, list(val = xgbvalid), print_every_n = 50, early_stopping_rounds = 150)

imp <- xgb.importance(model=m_xgb)

xgb.plot.importance(imp,measure = "Gain")
dtrain <- lgb.Dataset(data  = as.matrix(dtrain[, colnames(dtrain) != "is_attributed"]),
                      label = dtrain$is_attributed, 
                      categorical_feature = c("app", "device", "os", "channel", "hour"))


dvalid <- lgb.Dataset(data  = as.matrix(dvalid[, colnames(dvalid) != "is_attributed"]),
                      label = dvalid$is_attributed, 
                      categorical_feature = c("app", "device", "os", "channel", "hour"))
invisible(gc())
params <- list(objective = "binary", metric = "auc", learning_rate= 0.1, num_leaves= 7,
               max_depth= 4, scale_pos_weight= 99.7)

model3 <- lgb.train(params, dtrain, valids = list(validation = dvalid), nthread = 4,
                    nrounds = 1000,verbose= 1, early_stopping_rounds = 50, eval_freq = 50)
cat("--------------------------------", "\n")
cat("model 3 valid AUC: ", max(unlist(model3$record_evals[["validation"]][["auc"]][["eval"]])), "\n")
cat("--------------------------------", "\n")
lgb.save(model3, "model3")
lgb.plot.importance(lgb.importance(model3, percentage = TRUE), measure = "Gain")
rm(list=ls())

cat("
    -------------------------------------------
    Processing chunk 4  
    -------------------------------------------
    ")

if (!require("pacman")) install.packages("pacman")
pacman::p_load(knitr, pryr, tidyverse, data.table, lubridate, caret, tictoc, DescTools, lightgbm)
set.seed(84)
options(scipen = 9999, warn = -1, digits= 4)

train_path <- "D:/IE583/kaggle challenge/train.csv"

total_rows <- 92451945
chunk_rows <- 30817315
skip_rows  <- total_rows - chunk_rows

add_features <- function(df) {
  df <- as.data.table(df)
  df=df[order(click_time)]
  cat("Addidng interaction features..", "\n")
  df[, C_a:=.N,   by=list(app)]
  df[, C_i_a:=.N,   by=list(ip,app)]
  df[, C_i_a:=.N,   by=list(ip,app)]
  df[, C_i_a_d_h:=.N,   by=list(ip,app,day,hour)]
  df[, C_i_a_o_d_h:=.N,   by=list(ip,app,os,day,hour)]
  cat("Addidng next click time features..", "\n")
  #S=Sys.time()
  df[, N_i_a_o:=as.POSIXct(shift(click_time,type="lead"))-as.POSIXct(click_time), by=list(ip, app,os)]
  df$N_i_a_o=as.numeric(df$N_i_a_o,units="secs")
  df[, N_a_o_c:=as.POSIXct(shift(click_time,type="lead"))-as.POSIXct(click_time), by=list(app,os,channel)]
  df$N_a_o_c=as.numeric(df$N_a_o_c,units="secs")
  #E=Sys.time()
  #print(E-S)
  cat("Addidng unique features..", "\n")
  df[, U_i_c:= uniqueN(channel)/.N,   by=list(ip)]
  df[, U_i_o_c:= uniqueN(channel)/.N,   by=list(ip,os)]
  cat("Addidng variance features..", "\n") 
  df[, V_i_a_d_h:= var(hour),   by=list(ip,app,day)]
  df <- as.data.frame(df) %>% 
    select(-c(ip, day,click_time))
  return(df)
}
train <- fread(train_path, skip=skip_rows, nrows= chunk_rows, 
               colClasses=list(numeric=1:5), showProgress = FALSE, 
               col.names = c("ip", "app", "device", "os", "channel", "click_time", 
                             "attributed_time", "is_attributed")) %>% 
  select(-c(attributed_time)) %>%
  mutate(day = Weekday(click_time), hour = hour(click_time)) 


train.index <- createDataPartition(train$is_attributed, p = 0.9, list = FALSE)
dtrain <- train[ train.index,]
dvalid <- train[-train.index,]
rm(train, train.index, total_rows)
invisible(gc())

dtrain <- add_features(dtrain)
dvalid <- add_features(dvalid)

xgbtrain <- xgb.DMatrix(data = data.matrix(dtrain[, colnames(dtrain) != "is_attributed"]), label = dtrain$is_attributed,categorical_feature = c("app", "device", "os", "channel", "hour"))
xgbvalid <- xgb.DMatrix(data = data.matrix(dvalid[, colnames(dtrain) != "is_attributed"]), label = dvalid$is_attributed,categorical_feature = c("app", "device", "os", "channel", "hour"))
p <- list(objective = "binary:logistic",
          booster = "gbtree",
          eval_metric = "auc",
          nthread = 8,
          eta = 0.07,
          max_depth = 4,
          min_child_weight = 24,
          gamma = 36.7126,
          subsample = 0.9821,
          colsample_bytree = 0.3929,
          colsample_bylevel = 0.6818,
          alpha = 72.7519,
          lambda = 5.4826,
          max_delta_step = 5.7713,
          scale_pos_weight = 99.7,
          nrounds = 1000)

m_xgb <- xgb.train(p, xgbtrain, p$nrounds, list(val = xgbvalid), print_every_n = 50, early_stopping_rounds = 150)

imp <- xgb.importance(model=m_xgb)

xgb.plot.importance(imp,measure = "Gain")
dtrain <- lgb.Dataset(data  = as.matrix(dtrain[, colnames(dtrain) != "is_attributed"]),
                      label = dtrain$is_attributed, 
                      categorical_feature = c("app", "device", "os", "channel", "hour"))


dvalid <- lgb.Dataset(data  = as.matrix(dvalid[, colnames(dvalid) != "is_attributed"]),
                      label = dvalid$is_attributed, 
                      categorical_feature = c("app", "device", "os", "channel", "hour"))
invisible(gc())
params <- list(objective = "binary", metric = "auc", learning_rate= 0.1, num_leaves= 7,
               max_depth= 4, scale_pos_weight= 99.7)

model4 <- lgb.train(params, dtrain, valids = list(validation = dvalid), nthread = 4,
                    nrounds = 1000,verbose= 1, early_stopping_rounds = 50, eval_freq = 50)
cat("--------------------------------", "\n")
cat("model 4 valid AUC: ", max(unlist(model4$record_evals[["validation"]][["auc"]][["eval"]])), "\n")
cat("--------------------------------", "\n")
lgb.save(model4, "model4")
lgb.plot.importance(lgb.importance(model4, percentage = TRUE), measure = "Gain")
rm(list=ls())

cat("
    -------------------------------------------
    Processing chunk 5 
    -------------------------------------------
    ")

if (!require("pacman")) install.packages("pacman")
pacman::p_load(knitr, pryr, tidyverse, data.table, lubridate, caret, tictoc, DescTools, lightgbm)
set.seed(84)
options(scipen = 9999, warn = -1, digits= 4)

train_path <- "D:/IE583/kaggle challenge/train.csv"

total_rows <- 61634630
chunk_rows <- 30817315
skip_rows  <- total_rows - chunk_rows

add_features <- function(df) {
  df <- as.data.table(df)
  df=df[order(click_time)]
  cat("Addidng interaction features..", "\n")
  df[, C_a:=.N,   by=list(app)]
  df[, C_i_a:=.N,   by=list(ip,app)]
  df[, C_i_a:=.N,   by=list(ip,app)]
  df[, C_i_a_d_h:=.N,   by=list(ip,app,day,hour)]
  df[, C_i_a_o_d_h:=.N,   by=list(ip,app,os,day,hour)]
  cat("Addidng next click time features..", "\n")
  #S=Sys.time()
  df[, N_i_a_o:=as.POSIXct(shift(click_time,type="lead"))-as.POSIXct(click_time), by=list(ip, app,os)]
  df$N_i_a_o=as.numeric(df$N_i_a_o,units="secs")
  df[, N_a_o_c:=as.POSIXct(shift(click_time,type="lead"))-as.POSIXct(click_time), by=list(app,os,channel)]
  df$N_a_o_c=as.numeric(df$N_a_o_c,units="secs")
  #E=Sys.time()
  #print(E-S)
  cat("Addidng unique features..", "\n")
  df[, U_i_c:= uniqueN(channel)/.N,   by=list(ip)]
  df[, U_i_o_c:= uniqueN(channel)/.N,   by=list(ip,os)]
  cat("Addidng variance features..", "\n") 
  df[, V_i_a_d_h:= var(hour),   by=list(ip,app,day)]
  df <- as.data.frame(df) %>% 
    select(-c(ip, day,click_time))
  return(df)
}

train <- fread(train_path, skip=skip_rows, nrows= chunk_rows, 
               colClasses=list(numeric=1:5), showProgress = FALSE, 
               col.names = c("ip", "app", "device", "os", "channel", "click_time", 
                             "attributed_time", "is_attributed")) %>% 
  select(-c(attributed_time)) %>%
  mutate(day = Weekday(click_time), hour = hour(click_time)) 



train.index <- createDataPartition(train$is_attributed, p = 0.9, list = FALSE)
dtrain <- train[ train.index,]
dvalid <- train[-train.index,]
rm(train, train.index, total_rows)
invisible(gc())

dtrain <- add_features(dtrain)
dvalid <- add_features(dvalid)

xgbtrain <- xgb.DMatrix(data = data.matrix(dtrain[, colnames(dtrain) != "is_attributed"]), label = dtrain$is_attributed,categorical_feature = c("app", "device", "os", "channel", "hour"))
xgbvalid <- xgb.DMatrix(data = data.matrix(dvalid[, colnames(dtrain) != "is_attributed"]), label = dvalid$is_attributed,categorical_feature = c("app", "device", "os", "channel", "hour"))
p <- list(objective = "binary:logistic",
          booster = "gbtree",
          eval_metric = "auc",
          nthread = 8,
          eta = 0.07,
          max_depth = 4,
          min_child_weight = 24,
          gamma = 36.7126,
          subsample = 0.9821,
          colsample_bytree = 0.3929,
          colsample_bylevel = 0.6818,
          alpha = 72.7519,
          lambda = 5.4826,
          max_delta_step = 5.7713,
          scale_pos_weight = 99.7,
          nrounds = 1000)

m_xgb <- xgb.train(p, xgbtrain, p$nrounds, list(val = xgbvalid), print_every_n = 50, early_stopping_rounds = 150)

imp <- xgb.importance(model=m_xgb)

xgb.plot.importance(imp,measure = "Gain")
dtrain <- lgb.Dataset(data  = as.matrix(dtrain[, colnames(dtrain) != "is_attributed"]),
                      label = dtrain$is_attributed, 
                      categorical_feature = c("app", "device", "os", "channel", "hour"))


dvalid <- lgb.Dataset(data  = as.matrix(dvalid[, colnames(dvalid) != "is_attributed"]),
                      label = dvalid$is_attributed, 
                      categorical_feature = c("app", "device", "os", "channel", "hour"))
invisible(gc())

params <- list(objective = "binary", metric = "auc", learning_rate= 0.1, num_leaves= 7,
               max_depth= 4, scale_pos_weight= 99.7)

model5 <- lgb.train(params, dtrain, valids = list(validation = dvalid), nthread = 4,
                    nrounds = 1000,verbose= 1, early_stopping_rounds = 50, eval_freq = 50)
cat("--------------------------------", "\n")
cat("model 5 valid AUC: ", max(unlist(model5$record_evals[["validation"]][["auc"]][["eval"]])), "\n")
cat("--------------------------------", "\n")
lgb.save(model5, "model5")
lgb.plot.importance(lgb.importance(model5, percentage = TRUE), measure = "Gain")
rm(list=ls())

cat("
    -------------------------------------------
    Processing chunk 6 (last)  
    -------------------------------------------
    ")

if (!require("pacman")) install.packages("pacman")
pacman::p_load(knitr, pryr, tidyverse, data.table, lubridate, caret, tictoc, DescTools, lightgbm)
set.seed(84)
options(scipen = 9999, warn = -1, digits= 4)

train_path <- "D:/IE583/kaggle challenge/train.csv"

total_rows <- 30817315
chunk_rows <- 30817315
skip_rows  <- total_rows - chunk_rows

add_features <- function(df) {
  df <- as.data.table(df)
  df=df[order(click_time)]
  cat("Addidng interaction features..", "\n")
  df[, C_a:=.N,   by=list(app)]
  df[, C_i_a:=.N,   by=list(ip,app)]
  df[, C_i_a:=.N,   by=list(ip,app)]
  df[, C_i_a_d_h:=.N,   by=list(ip,app,day,hour)]
  df[, C_i_a_o_d_h:=.N,   by=list(ip,app,os,day,hour)]
  cat("Addidng next click time features..", "\n")
  #S=Sys.time()
  df[, N_i_a_o:=as.POSIXct(shift(click_time,type="lead"))-as.POSIXct(click_time), by=list(ip, app,os)]
  df$N_i_a_o=as.numeric(df$N_i_a_o,units="secs")
  df[, N_a_o_c:=as.POSIXct(shift(click_time,type="lead"))-as.POSIXct(click_time), by=list(app,os,channel)]
  df$N_a_o_c=as.numeric(df$N_a_o_c,units="secs")
  #E=Sys.time()
  #print(E-S)
  cat("Addidng unique features..", "\n")
  df[, U_i_c:= uniqueN(channel)/.N,   by=list(ip)]
  df[, U_i_o_c:= uniqueN(channel)/.N,   by=list(ip,os)]
  cat("Addidng variance features..", "\n") 
  df[, V_i_a_d_h:= var(hour),   by=list(ip,app,day)]
  df <- as.data.frame(df) %>% 
    select(-c(ip, day,click_time))
  return(df)
}

train <- fread(train_path, skip=skip_rows, nrows= chunk_rows, 
               colClasses=list(numeric=1:5), showProgress = FALSE, 
               col.names = c("ip", "app", "device", "os", "channel", "click_time", 
                             "attributed_time", "is_attributed")) %>% 
  select(-c(attributed_time)) %>%
  mutate(day = Weekday(click_time), hour = hour(click_time)) 


train.index <- createDataPartition(train$is_attributed, p = 0.9, list = FALSE)
dtrain <- train[ train.index,]
dvalid <- train[-train.index,]
rm(train, train.index, total_rows)
invisible(gc())
dtrain <- add_features(dtrain)
dvalid <- add_features(dvalid)

xgbtrain <- xgb.DMatrix(data = data.matrix(dtrain[, colnames(dtrain) != "is_attributed"]), label = dtrain$is_attributed,categorical_feature = c("app", "device", "os", "channel", "hour"))
xgbvalid <- xgb.DMatrix(data = data.matrix(dvalid[, colnames(dtrain) != "is_attributed"]), label = dvalid$is_attributed,categorical_feature = c("app", "device", "os", "channel", "hour"))
p <- list(objective = "binary:logistic",
          booster = "gbtree",
          eval_metric = "auc",
          nthread = 8,
          eta = 0.07,
          max_depth = 4,
          min_child_weight = 24,
          gamma = 36.7126,
          subsample = 0.9821,
          colsample_bytree = 0.3929,
          colsample_bylevel = 0.6818,
          alpha = 72.7519,
          lambda = 5.4826,
          max_delta_step = 5.7713,
          scale_pos_weight = 99.7,
          nrounds = 1000)

m_xgb <- xgb.train(p, xgbtrain, p$nrounds, list(val = xgbvalid), print_every_n = 50, early_stopping_rounds = 150)

imp <- xgb.importance(model=m_xgb)

xgb.plot.importance(imp,measure = "Gain")
dtrain <- lgb.Dataset(data  = as.matrix(dtrain[, colnames(dtrain) != "is_attributed"]),
                      label = dtrain$is_attributed, 
                      categorical_feature = c("app", "device", "os", "channel", "hour"))


dvalid <- lgb.Dataset(data  = as.matrix(dvalid[, colnames(dvalid) != "is_attributed"]),
                      label = dvalid$is_attributed, 
                      categorical_feature = c("app", "device", "os", "channel", "hour"))
invisible(gc())

params <- list(objective = "binary", metric = "auc", learning_rate= 0.1, num_leaves= 7,
               max_depth= 4, scale_pos_weight= 99.7)

model6 <- lgb.train(params, dtrain, valids = list(validation = dvalid), nthread = 4,
                    nrounds = 1000,verbose= 1, early_stopping_rounds = 50, eval_freq = 50)
cat("--------------------------------", "\n")
cat("model 6 valid AUC: ", max(unlist(model6$record_evals[["validation"]][["auc"]][["eval"]])), "\n")
cat("--------------------------------", "\n")
lgb.save(model6, "model6")
lgb.plot.importance(lgb.importance(model6, percentage = TRUE), measure = "Gain")
rm(list=ls())


cat("
    --------------------------------------------------------------
    Part 2: Predictions  
    --------------------------------------------------------------
    ")

print("processing test data...")

if (!require("pacman")) install.packages("pacman")
pacman::p_load(knitr, pryr, tidyverse, data.table, lubridate, caret, tictoc, DescTools, lightgbm)
set.seed(84)
options(scipen = 9999, warn = -1, digits= 4)

add_features <- function(df) {
  df <- as.data.table(df)
  df=df[order(click_time)]
  cat("Addidng interaction features..", "\n")
  df[, C_a:=.N,   by=list(app)]
  df[, C_i_a:=.N,   by=list(ip,app)]
  df[, C_i_a:=.N,   by=list(ip,app)]
  df[, C_i_a_d_h:=.N,   by=list(ip,app,day,hour)]
  df[, C_i_a_o_d_h:=.N,   by=list(ip,app,os,day,hour)]
  cat("Addidng next click time features..", "\n")
  #S=Sys.time()
  df[, N_i_a_o:=as.POSIXct(shift(click_time,type="lead"))-as.POSIXct(click_time), by=list(ip, app,os)]
  df$N_i_a_o=as.numeric(df$N_i_a_o,units="secs")
  df[, N_a_o_c:=as.POSIXct(shift(click_time,type="lead"))-as.POSIXct(click_time), by=list(app,os,channel)]
  df$N_a_o_c=as.numeric(df$N_a_o_c,units="secs")
  #E=Sys.time()
  #print(E-S)
  cat("Addidng unique features..", "\n")
  df[, U_i_c:= uniqueN(channel)/.N,   by=list(ip)]
  df[, U_i_o_c:= uniqueN(channel)/.N,   by=list(ip,os)]
  cat("Addidng variance features..", "\n") 
  df[, V_i_a_d_h:= var(hour),   by=list(ip,app,day)]
  df <- as.data.frame(df) %>% 
    select(-c(ip, day,click_time))
  return(df)
}



test <- fread("D:/IE583/kaggle challenge/test.csv", colClasses=list(numeric=2:6), showProgress = FALSE) %>%
  mutate(day = Weekday(click_time), hour = hour(click_time)) 
sub  <- data.table(click_id = test$click_id, is_attributed = NA)
test$click_id <- NULL
test <- add_features(test)
test <- as.matrix(test[, colnames(test)])

model1 <- lgb.load("model1")
model2 <- lgb.load("model2")
model3 <- lgb.load("model3")
model4 <- lgb.load("model4")
model5 <- lgb.load("model5")
model6 <- lgb.load("model6")

print("processing individual predictions...")
preds <- data.frame(
  p1 = round(as.data.frame(predict(model1, data = test, n = model1$best_iter)),4),
  p2 = round(as.data.frame(predict(model2, data = test, n = model2$best_iter)),4),
  p3 = round(as.data.frame(predict(model3, data = test, n = model3$best_iter)),4),
  p4 = round(as.data.frame(predict(model4, data = test, n = model4$best_iter)),4),
  p5 = round(as.data.frame(predict(model5, data = test, n = model5$best_iter)),4),
  p6 = round(as.data.frame(predict(model6, data = test, n = model6$best_iter)),4)
)

print("Preparing submission file...")
sub$is_attributed <- round(as.data.frame(rowMeans(preds)),4)
head(sub)
fwrite(sub, "sub_lgb_R_full.csv")

