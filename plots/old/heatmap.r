require(ggplot2)
require(tidyverse)
require(scales)

data <- read.csv('data/heatmaps.csv')
grouping <- c('environment', 'geometry', 'seed', 'file')

n_cols <- 100  # image width
# reshape
data <- data %>% pivot_longer(
    cols = -all_of(grouping),                # all pixel columns
    names_to = "pixel",
    values_to = "value"
  )

data <- data %>% mutate(
    pixel_index = as.numeric(gsub("[^0-9.]", "", pixel)),
    x = ((pixel_index) %% n_cols) + 1,
    y = ((pixel_index) %/% n_cols) + 1 
  )

data_filtered <- data %>% filter(seed == 1) %>% filter(file == 50)

p <- ggplot(data_filtered, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "blue", high = "yellow") +
  coord_fixed() +
  scale_y_reverse() +  # optional: flip y-axis to match image orientation
  theme_minimal() +
  theme(axis.text = element_blank(),
        axis.title = element_blank(),
        axis.ticks = element_blank()) + 
  facet_grid(geometry ~ environment)

ggsave(
  filename = "plots/heatmaps.pdf",
  plot = p,
  width = 10,
  height = 6,
  units = "in",
  device = cairo_pdf
)