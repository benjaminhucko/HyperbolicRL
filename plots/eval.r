require(ggplot2)
require(tidyverse)
require(scales)
require(jsonlite)
require(stringr)

data <- read.csv('data/eval.csv')
grouping <- c('environment', 'geometry', 'seed', 'file')


clean_labels <- function(x) {
  str_replace_all(x, "_", " ") |>
    str_to_title()
}

# Read the JSON file
color_map <- fromJSON("color.json")

data <- data %>% filter(file <= 0.25)

# MAYBE PLOT PERFORMANCE DECREASE + TABLE

# reshape 
data <- data %>% group_by(across(all_of(grouping))) %>%
  pivot_longer(cols = -all_of(grouping),
               names_to = "sample",
               values_to = "values") %>% ungroup()

# Average seed and samples
data <- data %>% group_by(environment, geometry, file) %>%
  summarise(returns=mean(values))

# to percentage
data <- data %>% mutate(returns=returns / returns[file=='0'])

p <- ggplot(data, aes(x=file, y=returns, color=geometry)) +
  geom_smooth() +
  scale_x_continuous(labels=scales::label_number(scale_cut=cut_short_scale())) +
  ylab("Performance compared to original (ratio)") +
  xlab(expression("Sticky probability (" * varsigma * ")")) +
  scale_color_manual(values = color_map$dark, labels = clean_labels) +
  facet_wrap(~environment, scales = "free_y", labeller = labeller(environment = clean_labels)) +
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.title = element_blank(),
    legend.key = element_blank(),
    legend.text = element_text(size = 12),
    strip.text = element_text(size = 14),
    axis.title = element_text(size = 12),
    axis.title.x = element_text(margin = margin(t = 10)),
    legend.spacing.y = unit(0, "cm"),
    plot.margin = margin(t = 0, r = 0, b = 0, l = 0)
  ) +
  guides(color = guide_legend(override.aes = list(fill = NA)))



ggsave(
  filename = "plots/eval.pdf",
  plot = p,
  width = 7,
  height = 5.5,
  units = "in",
  device = cairo_pdf
)