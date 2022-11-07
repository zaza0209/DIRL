library(ggplot2)
library(data.table)
library(lemon)
library(ggpubr)
library("ggsci")
if (Sys.info()['sysname'] == "Darwin"){
  curr_dir <- "/Users/mengbing/Documents/research/change_point_clustering/HeterRL_private/simulation_real/"
} else{
  curr_dir <- "/home/mengbing/research/HeterRL/simulation_real"
}
setwd(curr_dir)
args = commandArgs(trailingOnly=TRUE)
print(args)
dat <- fread(as.character(args[1]))
# dat <- fread(paste0("output/20221031_C0_offline_summarized.csv"))
effect_sizes = c('strong', 'moderate', 'weak')
# threshold_types = c("Chi2", "permutation")
dat[["Size of Change"]] <- factor(dat[["Effect Size"]], levels = effect_sizes, labels = c("Strong", 'Moderate', 'Weak'))

default_colors = c('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')
p1 <- ggplot(dat, aes(x=`Size of Change`, y = cp_error, fill = `Size of Change`)) + 
  # geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red") +
  geom_boxplot(alpha = 0.9) +
  # stat_summary(geom = "bar", fun = "sum", position = "identity")
  # facet_rep_grid(facets=gamma~setting, labeller = label_parsed) +
  # facet_rep_grid(facets=gamma ~ setting, labeller = labeller(.rows = label_parsed, .multi_line = TRUE)) +# label_parsed
  # labs(x = expression(paste("Estimated Change Point T - ", kappa, "*")), y='Count', fill='Method') + 
  # scale_x_continuous(breaks=x_breaks) + 
  # scale_x_discrete(drop=FALSE) +
  # scale_fill_manual(values=default_colors[1:length(unique(dat$Threshold))]) + 
  labs(y = "Change Point Error") +
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
  ) +
  scale_fill_startrek()
p2 <- ggplot(dat, aes(x=`Size of Change`, y = `ARI`, fill = `Size of Change`)) + 
  geom_hline(yintercept = 1, size = 1, linetype="dashed", color = "red") +
  geom_boxplot(alpha = 0.9) +
  # scale_fill_manual(values=default_colors[1:length(unique(dat$Threshold))]) + 
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
  ) +
  scale_fill_startrek()
ggarrange(p1, p2, nrow = 1, common.legend = TRUE, legend="right") 

ggsave(paste0("output/real_boxplot_simresult_C0.pdf"), width = 24, height = 9, units = "cm")

cat("Frequency of selecting K: \n")
table(dat$`Size of Change`, dat$K)

