#!/sw/arcts/centos7/stacks/gcc/8.2.0/R/4.1.0/bin/Rscript
library(data.table)
library(ggplot2)
library(dplyr)
library(tidyr)
library(ggbreak)
library(RColorBrewer)
library("ggsci")
args = commandArgs(trailingOnly=TRUE)
curr_dir <- "C:/Users/test/Dropbox/tml/IHS/simu/simu/toyexample/final_perf"
setwd(curr_dir)
N <- 50
## KMeans
dat_km =fread('icmodel_2022-11-02method(7)N50_1d.csv')
dat_km$init <- factor(dat_km$init, levels = c( 'best_model', 'kmeans_K1', 'kmeans_K2', 'kmeans_K3', 'kmeans_K4'),
                    labels = c('Model Selected via the Information Criterion','K=1', 'K=2', 'K=3', 'K=4'))
dat_km$Setting <- factor(dat_km$Setting, 
                      levels = c('pwconst2', 'smooth'),
                      labels = c('Piecewise Constant', 'Smooth'))
dat2_processed <- gather(dat_km, metric, value, -c('Setting', 'seed', 'init'))
dat2_processed$metric=factor(dat2_processed$metric, levels = c('cp_err', 'ARI'),
                             labels =  c('CP Error', 'ARI'))

cbPalette=c('#cc0c00', '#5c88da','#84bd00', '#ffcc00', '#7c878e','#00b5e2','#00af66',"#E69F00","#660099")
p2 <- ggplot(dat2_processed, aes(init, value, fill=init)) + #, color=`Effect Size`
  geom_boxplot(lwd=1, fatten=1, alpha=0.8) + 
  labs(fill='') +
  xlab("") + 
  ylab("Estimation Performance") +
  # labs(fill="Method") +
  theme(
    # legend.direction="vertical",
    legend.position = "bottom",
    # panel.border=element_blank(),
    # legend.box.spacing=0.4,
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    # axis.line=element_line(size=1, colour="black"),
    panel.grid.major=element_line(colour="#d3d3d3"),
    panel.grid.minor=element_line(colour="#d3d3d3"),
    panel.background=element_blank(),
    plot.title=element_text(size=20, face="bold"),
    text=element_text(size=18),
    legend.text=element_text(size=18),
    # axis.text.x=element_text(colour="black", size=16, angle = 90),
    # axis.text.x = element_blank(),
    axis.text.x = element_text(colour="white", size=0),
    axis.text.y=element_text(colour="black", size=16),
    plot.margin=grid::unit(c(0.3,0,-2,0), "mm")
  ) +
  facet_grid(facets =metric~ Setting, scales = 'free_y')

p2 +scale_fill_startrek() 
ggsave(paste0("offlineIC_k",Sys.Date(), ".pdf"), width = 14, height = 5)
