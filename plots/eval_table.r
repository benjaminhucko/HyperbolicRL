require(ggplot2)
require(tidyverse)
require(scales)
require(knitr)
require(kableExtra)
require(stringr)


clean_labels <- function(x) {
  str_replace_all(x, "_", " ") |>
    str_to_title()
}
data <- read.csv('data/eval.csv')
data <- data %>% filter(file == 0 | file == 0.15)
data <- data %>% select(file, environment, everything()) %>% arrange(file) %>%
  rename(state = file) %>%
  mutate(environment = clean_labels(environment), geometry = clean_labels(geometry)) %>%
  rename(Environment = environment)

grouping <- c('Environment', 'geometry', 'seed', 'state')

# reshape 
data <- data %>% group_by(across(all_of(grouping))) %>%
  pivot_longer(cols = -all_of(grouping),
               names_to = "sample",
               values_to = "values") %>% ungroup()

# Average seed and samples
data <- data %>% group_by(Environment, geometry, state) %>%
  summarise(report=round(mean(values, na.rm = TRUE), 2),
            stdev=round(sd(values), 2))

normal_data <- data %>% filter(state == 0) %>% 
  group_by(Environment) %>%
  mutate(improvement = report / report[geometry=="Euclidean"] - 1) %>%
  mutate(improvement=round(improvement, 2) * 100,
         is_best=report == max(report))

normal_data <- normal_data %>%
  mutate(report = paste0(report, "$\\pm$", stdev)) %>%
  mutate(report = case_when(
    geometry == "Euclidean" & is_best > 0 ~ paste0("\\textbf{", report, "}"),
    geometry == "Euclidean" ~ as.character(report),
    improvement > 0 ~ paste0("\\textbf{", report, "} (\\textcolor{ForestGreen}{+", improvement, "\\%})"),
    improvement < 0 ~ paste0(report, " (\\textcolor{BrickRed}{", improvement, "\\%})"))) %>%
  select(-c(improvement, state, stdev, is_best))


sticky_data <- data %>% 
  mutate(change=report / report[state==0]) %>% 
  filter(state != 0) %>% 
  group_by(Environment) %>%
  mutate(is_robust=change == max(change),
         is_best=report == max(report),
         change=round(change, 2) * 100)

sticky_data <- sticky_data %>%
  mutate(report = paste0(report, "$\\pm$", stdev)) %>%
  mutate(report = case_when(
    is_best & is_robust ~ paste0("\\textbf{", report, " (",  change, "\\%)}"),
    is_robust ~ paste0(report, " (\\textbf{",  change, "\\%})"),
    !is_robust ~ paste0(report, " (",  change, "\\%)"))) %>%
    select(-c(is_robust, is_best, change, state, stdev))

normal_data <- normal_data %>% pivot_wider(
    names_from = geometry,
    values_from = report
  )

sticky_data <- sticky_data %>% pivot_wider(
  names_from = geometry,
  values_from = report
)

normal <- kable(normal_data, format = "latex", booktabs = TRUE, escape = FALSE, linesep = "",
      caption = "Evaluation returns in MinAtar test suite") %>%
      kable_styling(latex_options = c("hold_position", "scale_down")) %>%
      save_kable("plots/exp1.tex")
sticky <- kable(sticky_data, format = "latex", booktabs = TRUE, escape = FALSE, linesep = "",
      caption = "Evaluation returns in MinAtar test suite with sticky actions ($\\varsigma = 0.1$)") %>%
      kable_styling(latex_options = c("hold_position", "scale_down")) %>%
      save_kable("plots/exp2.tex")