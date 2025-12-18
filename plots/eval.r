require(ggplot2)
require(tidyverse)
require(scales)
require(knitr)
require(kableExtra)

data_sim <- read.csv('data/normal.csv')
data_real <- read.csv('data/sticky_action.csv')

data_sim$file <- "original"
data_real$file <- "sticky_actions"
data <- rbind(data_sim, data_real)

grouping <- c('environment', 'geometry', 'seed', 'file')

# reshape 
data <- data %>% group_by(across(all_of(grouping))) %>%
  pivot_longer(cols = -all_of(grouping),
               names_to = "sample",
               values_to = "values") %>% ungroup()

# Average seed and samples
data <- data %>% group_by(environment, geometry, file) %>%
  summarise(mean=mean(values), stdev=sd(values))

# Comparison
data <- data %>% group_by(environment, file) %>%
  mutate(is_best = mean == max(mean)) %>%
  ungroup()

data <- data %>% mutate(mean_sd=paste0(round(mean, 1), " ± ", round(stdev, 1))) %>%
  mutate(mean_sd = ifelse(is_best, paste0("\\textbf{", mean_sd, "}"), mean_sd)) %>%
  select(-c(mean, stdev, is_best))

data <- data %>% pivot_wider(
    names_from = geometry,
    values_from = mean_sd
  )

data <- data %>% select(file, environment, everything()) %>% arrange(file) %>%
  rename(state = file) %>% mutate(environment = gsub("_", " ", environment),
                                   state = gsub("_", " ", state))

kable(data, format = "latex", booktabs = TRUE, escape = FALSE, linesep = "",
      caption = "Evaluation Returns on the same envionment compared to environment with sticky actions") %>%
  row_spec(row = 4, extra_latex_after = "\\addlinespace")