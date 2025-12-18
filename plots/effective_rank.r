require(ggplot2)
require(tidyverse)
require(scales)

data <- read.csv('data/plots.csv')
grouping <- c('environment', 'geometry', 'seed', 'file')

# reshape 
data <- data %>% group_by(across(all_of(grouping))) %>%
                 pivot_longer(cols = -all_of(grouping),
                              names_to = "interactions", 
                              values_to = "values") %>% ungroup()

frequency <- 128 * 8
data <- data %>% mutate(interactions = as.numeric(gsub("[^0-9.]", "", 
                                                       interactions))) %>% 
                 mutate(interactions = (interactions + 1) * frequency)
# Average seed
data %>% group_by(environment, geometry, file, interactions) %>% 
  summarise(values=mean(values)) %>% ungroup()

# min(n, d) 
maximum_rank <- c(actor=128, critic=128, visual=256)
norm_col <- maximum_rank[data$file]
data <- data %>% mutate(values = values / norm_col * 100)

p <- ggplot(data, aes(x=interactions, y=values, color=geometry)) + geom_smooth() +
  scale_x_continuous(labels=scales::label_number(scale_cut=cut_short_scale())) +
  scale_y_continuous(limits=c(0, 100)) +
  ylab("Effective rank (%)") +
  xlab("Environment interactions (Frames)") +
  facet_grid(file ~ environment)

ggsave(
  filename = "plots/effective_rank.pdf",
  plot = p,
  width = 10,
  height = 6,
  units = "in",
  device = cairo_pdf
)