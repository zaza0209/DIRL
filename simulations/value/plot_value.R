#!/sw/arcts/centos7/stacks/gcc/8.2.0/R/4.1.0/bin/Rscript
library(data.table)
library(ggplot2)
library(dplyr)
library(ggpubr)
library(ggbreak)
library("ggsci")
if(Sys.info()["nodename"] %in% c("PC-20181212HEBU")){
  curr_dir <- "D:/OneDrive/PhD/DIRL/IHS/simu/simu_anonymous/tuneK_iterations/value"
  setwd(curr_dir)
} else{ # greatlakes
  curr_dir <- "/home/xxx/research/RL_nonstationary/code2/simulation_nonstationary_changept_detection/output"
  setwd(curr_dir)
}

N <- 50
set.seed(20)
dat <- fread("vall_2022-11-05N50_diffsign.csv")
dat2 <- fread("vall_2024-03-18N50_1d.csv")
dat[dat$init == 'proposed', ] = dat2[dat2$init == 'proposed',]
dat[dat$init == 'only_clusters', ] = dat2[dat2$init == 'only_clusters',]
dat$Setting <- factor(dat$Setting, 
                      levels = c('pwconst2', 'smooth'),
                      labels = c('Piecewise Constant', 'Smooth'))
dat[,'Average Value'] = dat[,'Average Value'] * 10
my_colors=c('#cc0c00', '#5c88da','#84bd00', '#ffcc00', '#7c878e','#00b5e2','#00af66',"#E69F00","#660099")
dat$init = factor(dat$init, levels = c("proposed", "oracle", "overall", "only_cp","only_clusters"),
                  labels  = c("Proposed", "Oracle", "DH", "Homongeneous", "Stationary"))
p_diff_av_value <- ggboxplot(dat,x='init', y = 'Average Value',fill = 'init', alpha=0.8,
                       ylab="Average Value", xlab="",lwd=1, fatten=1,
                       facet.by = c('Setting'),alpha=0.8,palette = my_colors[1:5],
                       ggtheme = theme(
                         # legend.direction="vertical",
                         # legend.position = "None",
                         legend.position = "bottom",
                         legend.text = element_text(size=16),
                         legend.margin=margin(t = 0, unit='cm'),
                         legend.box.margin=margin(-30,0,0,0),
                         panel.border = element_rect(color = "black", fill = NA, size = 1),
                         # axis.line=element_line(size=1, colour="black"),
                         panel.grid.major=element_line(colour="#d3d3d3"),
                         panel.grid.minor=element_line(colour="#d3d3d3"),
                         panel.background=element_blank(),
                         plot.title=element_text(size=18, face="bold"),
                         text=element_text(size=16),
                         # axis.text.x = element_blank(),
                         axis.text.x=element_text(colour="white", size=0, angle = 0),
                         # strip.text.y = element_blank(),
                         axis.text.y=element_text(colour="black", size=16),
                         plot.margin=grid::unit(c(0.3,0,0,0), "mm")
                       ))#+
  # geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red")
p_diff_av_value=ggpar(p_diff_av_value, legend.title = "")
p_diff_av_value
ggsave(paste0("diff_av_value", Sys.Date(), ".pdf"), width = 14, height = 2.5)

## discounted return
p_dis_value <- ggboxplot(dat,x='init', y = 'Discounted Value',fill = 'init', alpha=0.8,
                             ylab="Value", xlab="",lwd=1, fatten=1,
                             facet.by = c('Setting'),alpha=0.8,palette = my_colors[1:5],
                             ggtheme = theme(
                               # legend.direction="vertical",
                               # legend.position = "None",
                               legend.position = "bottom",
                               legend.text = element_text(size=16),
                               legend.margin=margin(t = 0, unit='cm'),
                               legend.box.margin=margin(-30,0,0,0),
                               panel.border = element_rect(color = "black", fill = NA, size = 1),
                               # axis.line=element_line(size=1, colour="black"),
                               panel.grid.major=element_line(colour="#d3d3d3"),
                               panel.grid.minor=element_line(colour="#d3d3d3"),
                               panel.background=element_blank(),
                               plot.title=element_text(size=18, face="bold"),
                               text=element_text(size=16),
                               # axis.text.x = element_blank(),
                               axis.text.x=element_text(colour="white", size=0, angle = 0),
                               # strip.text.y = element_blank(),
                               axis.text.y=element_text(colour="black", size=16),
                               plot.margin=grid::unit(c(0.3,0,0,0), "mm")
                             ))#+
# geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red")
p_dis_value=ggpar(p_dis_value, legend.title = "")
p_dis_value
ggsave(paste0("dis_value", Sys.Date(), ".pdf"), width = 14, height = 2.5)
################### 
if(Sys.info()["nodename"] %in% c("PC-20181212HEBU")){
  curr_dir <- "D:/OneDrive/PhD/DIRL/IHS/simu/simu_anonymous/tuneK_iterations/value_samesign"
  setwd(curr_dir)
} else{ # greatlakes
  curr_dir <- "/home/xxx/research/RL_nonstationary/code2/simulation_nonstationary_changept_detection/output"
  setwd(curr_dir)
}
N <- 50
set.seed(20)
dat <- fread("vall_2022-11-06N50_samesign.csv")
dat2 <- fread('vall_2024-03-19N50_1d.csv')
dat[dat$init == 'proposed', ] = dat2[dat2$init == 'proposed',]
dat[dat$init == 'only_cluster', ] = dat[dat$init == 'only_cluster',]
dat$Setting <- factor(dat$Setting, 
                      levels = c('pwconst2', 'smooth'),
                      labels = c('Piecewise Constant', 'Smooth'))
dat[,'Average Value'] = dat[,'Average Value'] * 10
dat$init = factor(dat$init, levels = c("proposed", "oracle", "overall", "only_cp","only_clusters"),
                  labels  = c("Proposed", "Oracle", "DH", "Homongeneous", "Stationary"))

cbPalette=c('#cc0c00', '#5c88da','#84bd00', '#ffcc00', '#7c878e','#00b5e2','#00af66',"#E69F00","#660099")
p_same_av_value <- ggboxplot(dat,x='init', y = 'Average Value',fill = 'init', alpha=0.8,
                             ylab="Average Value", xlab="",lwd=1, fatten=1,
                             facet.by = c('Setting'),alpha=0.8,palette = my_colors[1:5],
                             ggtheme = theme(
                               # legend.direction="vertical",
                               # legend.position = "None",
                               legend.position = "bottom",
                               legend.text = element_text(size=16),
                               legend.margin=margin(t = 0, unit='cm'),
                               legend.box.margin=margin(-30,0,0,0),
                               panel.border = element_rect(color = "black", fill = NA, size = 1),
                               # axis.line=element_line(size=1, colour="black"),
                               panel.grid.major=element_line(colour="#d3d3d3"),
                               panel.grid.minor=element_line(colour="#d3d3d3"),
                               panel.background=element_blank(),
                               plot.title=element_text(size=18, face="bold"),
                               text=element_text(size=16),
                               # axis.text.x = element_blank(),
                               axis.text.x=element_text(colour="white", size=0, angle = 0),
                               # strip.text.y = element_blank(),
                               axis.text.y=element_text(colour="black", size=16),
                               plot.margin=grid::unit(c(0.3,0,0,0), "mm")
                             ))#+
# geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red")
p_same_av_value=ggpar(p_same_av_value, legend.title = "")
p_same_av_value
ggsave(paste0("same_av_value", Sys.Date(), ".pdf"), width = 14, height = 2.5)

## discounted return
p_dis_value <- ggboxplot(dat,x='init', y = 'Discounted Value',fill = 'init', alpha=0.8,
                         ylab="Value", xlab="",lwd=1, fatten=1,
                         facet.by = c('Setting'),alpha=0.8,palette = my_colors[1:5],
                         ggtheme = theme(
                           # legend.direction="vertical",
                           # legend.position = "None",
                           legend.position = "bottom",
                           legend.text = element_text(size=16),
                           legend.margin=margin(t = 0, unit='cm'),
                           legend.box.margin=margin(-30,0,0,0),
                           panel.border = element_rect(color = "black", fill = NA, size = 1),
                           # axis.line=element_line(size=1, colour="black"),
                           panel.grid.major=element_line(colour="#d3d3d3"),
                           panel.grid.minor=element_line(colour="#d3d3d3"),
                           panel.background=element_blank(),
                           plot.title=element_text(size=18, face="bold"),
                           text=element_text(size=16),
                           # axis.text.x = element_blank(),
                           axis.text.x=element_text(colour="white", size=0, angle = 0),
                           # strip.text.y = element_blank(),
                           axis.text.y=element_text(colour="black", size=16),
                           plot.margin=grid::unit(c(0.3,0,0,0), "mm")
                         ))#+
# geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red")
p_dis_value=ggpar(p_dis_value, legend.title = "")
p_dis_value
ggsave(paste0("same_dis_value", Sys.Date(), ".pdf"), width = 14, height = 2.5)

