library(ggplot2)
library(data.table)
library(lemon)
library(ggpubr)
if (Sys.info()['sysname'] == "Darwin"){
  curr_dir <- "/Users/mengbing/Documents/research/change_point_clustering/HeterRL/simulation_real"
} else{
  curr_dir <- "/home/mengbing/research/HeterRL/simulation_real"
}
setwd(curr_dir)
args = commandArgs(trailingOnly=TRUE)
print(args)
dat <- fread(paste0("output/values_20220722.csv"))
effect_sizes = c('strong', 'moderate', 'weak')
dat[["Size of Change"]] <- factor(dat[["Size of Change"]], levels = effect_sizes, labels = c("Strong", 'Moderate', 'Weak'))
# dat[["Method"]] <- factor(dat[["Method"]], levels = c('Proposed', "Oracle", "True cluster", "Estimated cluster", 'Overall'))
dat[["Method"]] <- factor(dat[["Method"]], levels = c('Proposed', "Oracle", "Stationary + Homogeneity", "Random"))

### calculate the difference with the proposed method
dat_proposed <- unique(dat[Method == "Proposed",])
colnames(dat_proposed)[ncol(dat_proposed)] <- "proposed_value"
dat_proposed$Method <- NULL
dat2 <- merge(x = dat, y = dat_proposed, by = colnames(dat_proposed)[-ncol(dat_proposed)], all.x = TRUE)
dat2[, Difference := proposed_value - Value]
dat2 <- dat2[complete.cases(dat2)]
default_colors = c('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')
ggplot(dat2[Method != "Proposed"], aes(x=`Size of Change`, y = `Difference`, fill = Method)) + 
  geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red") +
  geom_boxplot(alpha = 0.9) +
  scale_fill_manual(values=default_colors[1:length(unique(dat$Method))]) + 
  labs(y = "Value of the proposed method \nminus other methods") +
  theme_bw() + 
  theme(legend.position="right",
        legend.direction="vertical",
        # legend.box.spacing=0.4,
        axis.line=element_line(size=0.5, colour="black"),
        # panel.grid.major=element_line(colour="#d3d3d3"),
        panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        # panel.border=element_blank(),
        panel.spacing=unit(0,'npc'),
        panel.background=element_blank(),
        plot.title=element_text(size=18, face="bold"),
        text=element_text(size=13),
        axis.text.x=element_text(colour="black", size=10, angle=0),
        axis.text.y=element_text(colour="black", size=10),
  ) 
ggsave(paste0("output/real_boxplot_values.pdf"), width = 18, height = 9, units = "cm")



