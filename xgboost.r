if (!require("pacman")) install.packages("pacman")
pacman::p_load(knitr, pryr, tidyverse, data.table, lubridate, caret, tictoc, DescTools,xgboost)

set.seed(84)
options(scipen = 9999, warn = -1, digits= 4)
train_path <- "D:/prediction/challenge/kaggle/train.csv"
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
  cat("Addidng unique features..", "\n")
  df[, U_i_c:= uniqueN(channel)/.N,   by=list(ip)]
  df[, U_i_o_c:= uniqueN(channel)/.N,   by=list(ip,os)]
  cat("Addidng variance features..", "\n") 
  df[, V_i_a_d_h:= var(hour),   by=list(ip,app,day)]
  df <- as.data.frame(df) %>% 
    select(-c(ip, day,click_time))
  return(df)
}
train <- fread(train_path,  
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

xgbtrain <- xgb.DMatrix(data = data.matrix(dtrain[, colnames(dtrain) != "is_attributed"]), label = dtrain$is_attributed)
xgbvalid <- xgb.DMatrix(data = data.matrix(dvalid[, colnames(dtrain) != "is_attributed"]), label = dvalid$is_attributed)
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
rm(xgbtrain, xgbvalid,m_xgb)

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

train_path <- "D:/prediction/challenge/kaggle/train.csv"

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

train_path <- "D:/prediction/challenge/kaggle/train.csv"

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

train_path <- "D:/prediction/challenge/kaggle/train.csv"

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

train_path <- "D:/prediction/challenge/kaggle/train.csv"

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

train_path <- "D:/prediction/challenge/kaggle/train.csv"

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
