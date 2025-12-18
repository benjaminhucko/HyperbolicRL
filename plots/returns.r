require(ggplot2)
require(tidyverse)
require(scales)

data <- read.csv('data/dynamic.csv')
grouping <- c('environment', 'geometry', 'seed')

# Average seed
data <- data %>% group_by(environment, geometry, steps) %>%
                 summarise(returns=mean(returns)) %>% ungroup()

p <- ggplot(data, aes(x=steps, y=returns, color=geometry)) + geom_smooth() +
  scale_x_continuous(labels=scales::label_number(scale_cut=cut_short_scale())) +
  ylab("Returns") +
  xlab("Environment interactions (Frames)") +
  facet_wrap(~environment, scales = "free_y")

ggsave(
  filename = "plots/returns.pdf",
  plot = p,
  width = 10,
  height = 6,
  units = "in",
  device = cairo_pdf
)