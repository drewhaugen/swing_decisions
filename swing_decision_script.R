library(readr)
library(dplyr)
library(ggplot2)
library(xgboost)
library(caTools)
library(caret)
library(ParBayesianOptimization)

data1 <- read_csv("hitter_data_19.csv")
data2 <- read_csv("hitter_data_20.csv")
data3 <- read_csv("hitter_data_21.csv")

data <- rbind(data1, data2, data3)
rm(data1, data2, data3)

swing_data <- data%>%
  filter(description != "ball" & description != "called_strike" & description != "blocked_ball" & description != "pitchout" & description != "hit_by_pitch")%>% 
  mutate(outcome = ifelse(description == "swinging_strike" | description == "swinging_strike_blocked" | description == "foul_tip" | (strikes == 2 & description == "foul_bunt") | description == "bunt_foul_tip" | description == "missed_bunt", 1,
                          ifelse(description == "foul" | description == "foul_bunt", 2,  
                                 ifelse(events == "single", 3, ifelse(events == "double", 4, 
                                                                      ifelse(events == "triple", 5, ifelse(events == "home_run", 6, 7)))))))%>% 
  mutate(stand = as.factor(stand))%>% 
  filter(!is.na(outcome), 
         !is.na(plate_x), 
         !is.na(plate_z), 
         !is.na(pfx_x), 
         !is.na(pfx_z), 
         !is.na(release_speed), 
         !is.na(stand), 
         !is.na(p_throws))%>% 
  select(outcome, plate_x, plate_z, pfx_x, pfx_z, release_speed, stand)%>%
  mutate(outcome = outcome - 1)

rv <- data%>%
  filter(description != "ball" & description != "called_strike" & description != "blocked_ball" & description != "pitchout" & description != "hit_by_pitch")%>% 
  mutate(outcome = ifelse(description == "swinging_strike" | description == "swinging_strike_blocked" | description == "foul_tip" | (strikes == 2 & description == "foul_bunt") | description == "bunt_foul_tip" | description == "missed_bunt", 1,
                          ifelse(description == "foul" | description == "foul_bunt", 2,  
                                 ifelse(events == "single", 3, ifelse(events == "double", 4, 
                                                                      ifelse(events == "triple", 5, ifelse(events == "home_run", 6, 7)))))))%>% 
  mutate(stand = as.factor(stand))%>% 
  filter(!is.na(outcome), 
         !is.na(plate_x), 
         !is.na(plate_z), 
         !is.na(pfx_x), 
         !is.na(pfx_z), 
         !is.na(release_speed), 
         !is.na(stand), 
         !is.na(p_throws))%>% 
  mutate(outcome = outcome - 1)%>%
  group_by(outcome)%>%
  summarize(mean_rv = mean(delta_run_exp, na.rm = T))%>%
  ungroup()
des <- c("miss", "foul", "single", "double", "triple", "home_run", "out")
rv <- data.frame(rv, des)
rm(des)

factor <- swing_data%>%
  select(stand)

dummy <- dummyVars( ~ ., data = factor)
dummy_data <- data.frame(predict(dummy, newdata = factor))

swing_data_f <-  swing_data%>%
  select(-stand)%>%
  data.frame(dummy_data)

rm(dummy, dummy_data, factor, swing_data)

set.seed(11)
swing_sample = sample.split(swing_data_f$outcome, SplitRatio = .75) 
swing_train = subset(swing_data_f, swing_sample == TRUE) 
swing_test = subset(swing_data_f, swing_sample == FALSE) 

x_data <- swing_train %>%
  select(-outcome) %>%
  as.matrix()

y_data <- swing_train %>%
  pull(outcome)

x_data_test <- swing_test %>%
  select(-outcome) %>%
  as.matrix()

y_data_test <- swing_test %>%
  pull(outcome)

set.seed(22)
folds <- list(fold1 = as.integer(seq(1, nrow(x_data), by = 3)),
              fold2 = as.integer(seq(2, nrow(x_data), by = 3)),
              fold3 = as.integer(seq(3, nrow(x_data), by = 3)),
              fold4 = as.integer(seq(4, nrow(x_data), by = 3)),
              fold5 = as.integer(seq(5, nrow(x_data), by = 3)))


obj_func <- function(max_depth, min_child_weight, subsample, colsample_bytree) {
  
  param <- list(
    
    # Hyper parameters 
    eta = 0.1,
    num_class = 8,
    max_depth = max_depth,
    min_child_weight = min_child_weight,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    
    # Tree model 
    booster = "gbtree",
    
    # Classification problem 
    objective = "multi:softprob",
    
    
    eval_metric = "merror")
  
  xgbcv <- xgboost::xgb.cv(params = param,
                           data = x_data,
                           label = y_data,
                           nround = 150,
                           folds = folds,
                           prediction = TRUE,
                           early_stopping_rounds = 25,
                           verbose = 1,
                           maximize = F)
  
  lst <- list(
    
    # First argument must be named as "Score"
    # Function finds maxima so inverting the output
    Score = -min(xgbcv$evaluation_log$test_merror_mean),
    
    # Get number of trees for the best performing model
    nrounds = xgbcv$best_iteration
  )
  
  return(lst)
}

bounds <- list(max_depth = c(1L, 10L),
               min_child_weight = c(3, 10),
               subsample = c(0.25, 1),
               colsample_bytree = c(.1, 1))

set.seed(33)
bayes_opt <- bayesOpt(FUN = obj_func, bounds = bounds, initPoints = length(bounds) + 6, iters.n = 15,
                      plotProgress = TRUE)

bayes_opt$scoreSummary
data.frame(getBestPars(bayes_opt))

params <- list(booster = "gbtree", num_class = 7, eval_metric = "merror",
               eta = 0.1, objective = "multi:softprob", max_depth = 10, 
               min_child_weight = 3, subsample = 0.8040254, colsample_bytree = 1)

xgb.train <- xgb.DMatrix(data = x_data, label = y_data)
xgb.test <- xgb.DMatrix(data = x_data_test, label = y_data_test)

xgbcv <- xgb.cv(params = params, data = xgb.train, nrounds = 250, nfold = 7, showsd = T,
                stratified = T, print_every_n = 3, early_stopping_rounds = 20, maximize = F)
#Best Iteration: 75

swing_xgb <- xgb.train(data = xgb.train, params = params, nrounds = 75,
                       watchlist = list(val = xgb.test, train = xgb.train))

saveRDS(swing_xgb, "swing_xgb.rds")

library(mgcv)

takes <- data%>% 
  filter(!is.na(plate_x), 
         !is.na(plate_z))%>% 
  filter(description == "ball" | description == "called_strike")%>% 
  mutate(strike = ifelse(description == "called_strike", 1,0)) 

takes_l <- takes%>% 
  filter(stand == "L") 

takes_r<- takes%>% 
  filter(stand == "R") 

cs_gam_l <- gam(strike ~ s(plate_x, plate_z), data = takes_l, family = binomial()) 
cs_gam_r <- gam(strike ~ s(plate_x, plate_z), data = takes_r, family = binomial()) 

saveRDS(cs_gam_l, "cs_gam_l.rds")
saveRDS(cs_gam_r, "cs_gam_r.rds")

sd_rv <- function(swing_model, cs_mod_l, cs_mod_r, df){
  library(caret)
  library(dplyr)
  
  df <- df%>%
    filter(!is.na(plate_x), 
           !is.na(plate_z), 
           !is.na(pfx_x), 
           !is.na(pfx_z), 
           !is.na(release_speed), 
           !is.na(stand), 
           !is.na(p_throws))
  
  swing_data <- df%>%
    filter(!is.na(plate_x), 
           !is.na(plate_z), 
           !is.na(pfx_x), 
           !is.na(pfx_z), 
           !is.na(release_speed), 
           !is.na(stand), 
           !is.na(p_throws))%>%
    mutate(stand = as.factor(stand))%>%
    select(plate_x, plate_z, pfx_x, pfx_z, release_speed, stand)
  
  factor <- swing_data%>%
    select(stand)
  
  dummy <- dummyVars( ~ ., data = factor)
  dummy_data <- data.frame(predict(dummy, newdata = factor))
  
  swing_data_f <-  swing_data%>%
    select(-stand)%>%
    data.frame(dummy_data)
  
  rm(dummy, dummy_data, factor, swing_data)
  
  
  
  swing_preds <- as.data.frame(predict(swing_model, newdata = as.matrix(swing_data_f), reshape = T))
  
  new_df <- data.frame(df, swing_preds)
  
  df_l <- new_df%>%
    filter(stand == "L")
  
  df_r <- new_df%>%
    filter(stand == "R")
  
  cs_pred_l <- as.data.frame(predict(cs_mod_l, newdata = df_l, type = "response"))
  cs_pred_l <- cs_pred_l%>%
    rename(strike_prob_yes = 'predict(cs_mod_l, newdata = df_l, type = "response")')%>%
    mutate(strike_prob_no = 1- strike_prob_yes)
  
  cs_pred_r <- as.data.frame(predict(cs_mod_r, newdata = df_r, type = "response"))
  cs_pred_r <- cs_pred_r%>%
    rename(strike_prob_yes = 'predict(cs_mod_r, newdata = df_r, type = "response")')%>%
    mutate(strike_prob_no = 1- strike_prob_yes)
  
  full_l <- data.frame(df_l, cs_pred_l)
  full_r <- data.frame(df_r, cs_pred_r)
  
  full_df <- rbind(full_l, full_r)
  
  full_df <- full_df%>%
    mutate(swing = ifelse(description %in%c("foul_tip", "swinging_strike",  
                                            "swinging_strike_blocked", "missed_bunt",
                                            "foul", "hit_into_play", "foul_bunt",
                                            "bunt_foul_tip"), 1, 0))%>%
    mutate(take_rv = (strike_prob_no * 0.0586) + (strike_prob_yes * -0.0665))%>%
    mutate(swing_rv = (-0.12170905 * V1) + (-0.03804139 * V2) + (0.48190947 * V3) +
             (0.76847131 * V4) + (1.05381872 * V5) + (1.37475405 * V6) + (-0.25045168 * V7))%>%
    mutate(should_swing = ifelse(swing_rv > take_rv, 1, 0), correct = ifelse(should_swing == swing, 1, 0))%>%
    mutate(decision_rv = ifelse(swing == 1, swing_rv - take_rv, take_rv - swing_rv),
           sd_plus = as.numeric(psych::rescale(decision_rv, mean = 100, sd = 50, df = F)))
  
  return(full_df)
  
}

library(baseballr)

d1 <- scrape_statcast_savant(start_date = "2022-04-07", 
                             end_date = "2022-04-15",
                             player_type = "batter") 

d2 <- scrape_statcast_savant(start_date = "2022-04-16", 
                             end_date = "2022-04-24",
                             player_type = "batter") 

d3 <- scrape_statcast_savant(start_date = "2022-04-25", 
                             end_date = "2022-04-30",
                             player_type = "batter")

d4 <- scrape_statcast_savant(start_date = "2022-05-01", 
                             end_date = "2022-05-09",
                             player_type = "batter")

d5 <- scrape_statcast_savant(start_date = "2022-05-10", 
                             end_date = "2022-05-18",
                             player_type = "batter")

d6 <- scrape_statcast_savant(start_date = "2022-05-19", 
                             end_date = "2022-05-27",
                             player_type = "batter")

d7 <- scrape_statcast_savant(start_date = "2022-05-28", 
                             end_date = "2022-06-04",
                             player_type = "batter")

d8 <- scrape_statcast_savant(start_date = "2022-06-05", 
                             end_date = "2022-06-12",
                             player_type = "batter")

d9 <- scrape_statcast_savant(start_date = "2022-06-13", 
                             end_date = "2022-06-20",
                             player_type = "batter")

d10 <- scrape_statcast_savant(start_date = "2022-06-21", 
                              end_date = "2022-06-23",
                              player_type = "batter")

data_22 <- rbind(d1, d2, d3, d4, d5, d6, d7, d8, d9, d10)
rm(d1, d2, d3, d4, d5, d6, d7, d8, d9, d10)

sd_rv22 <- sd_rv(swing_xgb, cs_gam_l, cs_gam_r, data_22)

sd_rv22%>%
  mutate(swing_new = ifelse(swing == 1, "Yes", "No"))%>%
  filter(plate_x >= -2.5, plate_x <= 2.5)%>%
  filter(plate_z >= -.5, plate_z <= 5)%>%
  ggplot(aes(plate_x, plate_z))+
  geom_point(aes(shape = as.factor(swing_new), color = sd_plus))+
  geom_path(data = kZone, aes(x,y))+
  scale_color_gradient2(low = "blue", high = "red", midpoint = 100)+
  coord_fixed()+
  theme_bw()+
  labs(shape = "Swing",
       color = "SD+",
       x = "Plate X",
       y = "Plate Z",
       title = "Swing Decisions in 2022")

#Final Graph Here: https://twitter.com/Drew_Haugen/status/1536736008762449926

#Using Same Models:

sd_rv21 <- sd_rv(swing_xgb, cs_gam_l, cs_gam_r, read_csv("hitter_data_21.csv"))

xbh_21 <- sd_rv21%>%
  mutate(p_xbh = V4 + V5 + V6)%>%
  filter(swing == 1)%>%
  group_by(player_name, batter)%>%
  summarize(n = n(), avg = mean(p_xbh)*100)%>%
  filter(n >= 500)%>%
  arrange(desc(avg))%>%
  ungroup()

xbh_22 <- sd_rv22%>%
  mutate(p_xbh = V4 + V5 + V6)%>%
  filter(swing == 1)%>%
  group_by(player_name, batter)%>%
  summarize(n_22 = n(), avg_22 = mean(p_xbh)*100)%>%
  filter(n_22 >= 250)%>%
  arrange(desc(avg_22))%>%
  ungroup()

left_join(xbh_21, xbh_22, by = "batter")%>%
  mutate(diff = round(avg_22 - avg, 3))%>%
  arrange(-diff)%>%
  head()

sd_rv21%>%
  filter(player_name == "Alvarez, Yordan")%>%
  mutate(p_xbh = V4 + V5 + V6)%>%
  filter(swing == 1)%>%
  ggplot(aes(plate_x, plate_z))+
  geom_point(aes(color = p_xbh), size = 2)+
  scale_color_gradient2(high = "red", low = "blue", midpoint = 0.05, breaks = c(.025, .05, .075),
                        limits = c(0, .1), oob = scales::squish)+
  geom_path(data = kZone, mapping = aes(x, y), lwd = 1.3, color = "black")+
  xlim(-2, 2)+
  ylim(-1, 4.5)+
  theme_bw()+
  coord_fixed()+
  annotate(geom = "text", x = -1, y = -.75, label = "Mean p(XBH): 4.77%%")+
  labs(title = "Yordan Alvarez Swings, 2021",
       x = "Plate X",
       y = "Plate Z",
       color = "p(XBH)",
       subtitle = "Probabilities generated with pitch location, \nspeed, movement, and batter hand")

sd_rv22%>%
  filter(player_name == "Alvarez, Yordan")%>%
  mutate(p_xbh = V4 + V5 + V6)%>%
  filter(swing == 1)%>%
  ggplot(aes(plate_x, plate_z))+
  geom_point(aes(color = p_xbh), size = 2)+
  scale_color_gradient2(high = "red", low = "blue", midpoint = 0.05, breaks = c(.025, .05, .075),
                        limits = c(0, .1), oob = scales::squish)+
  geom_path(data = kZone, mapping = aes(x, y), lwd = 1.3, color = "black")+
  xlim(-2, 2)+
  ylim(-1, 4.5)+
  theme_bw()+
  coord_fixed()+
  annotate(geom = "text", x = -1, y = -.75, label = "Mean p(XBH): 5.18%")+
  labs(title = "Yordan Alvarez Swings, 2022",
       x = "Plate X",
       y = "Plate Z",
       color = "p(XBH)",
       subtitle = "Probabilities generated with pitch location, \nspeed, movement, and batter hand")

#Final Graphs Here: https://twitter.com/Drew_Haugen/status/1539670093893861385