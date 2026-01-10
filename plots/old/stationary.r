require(ggplot2)
require(tidyverse)
require(scales)

data <- read.csv('data/stationary.csv')
grouping <- c('environment', 'geometry', 'seed')

# Average seed
data <- data %>% group_by(environment, geometry, steps) %>%
            summarise(combined=mean(combined), policy=mean(policy), value=mean(value), 
            entropy=mean(entropy)) %>% ungroup()
data <- data %>%
  pivot_longer(
    cols = c(combined, policy, value, entropy),
    names_to = "loss",
    values_to = "value"
  )

data <- data %>% filter(loss != "entropy")

p <- ggplot(data, aes(x=steps, y=value, color=geometry)) + geom_smooth() +
  scale_x_continuous(labels=scales::label_number(scale_cut=cut_short_scale())) +
  ylab("Loss") +
  xlab("Epoch") +
  facet_grid(loss ~ environment, 
             scales = "free_y")

 ggsave(
  filename = "plots/stationary.pdf",
  plot = p,
  width = 10,
  height = 6,
  units = "in",
  device = cairo_pdf
)