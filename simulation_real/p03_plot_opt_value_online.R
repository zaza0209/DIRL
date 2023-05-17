#!/sw/arcts/centos7/stacks/gcc/8.2.0/R/4.1.0/bin/Rscript
library(data.table)
library(ggplot2)
library(dplyr)
library(ggbreak)
library(ggsci)
args = commandArgs(trailingOnly=TRUE)
# N <- as.integer(args[1])
if(Sys.info()["sysname"] %in% c("Darwin")){
  curr_dir <- "/Users/mengbing/Documents/research/change_point_clustering/HeterRL_private/simulation_real"
  setwd(curr_dir)
} else{ # greatlakes
  curr_dir <- "/home/mengbing/research/HeterRL/simulation_real"
  setwd(curr_dir)
}

set.seed(50)
# dat <- fread(paste0("data_optvalue_online_dt_N", N, ".csv"))
date <- "20230510"
dat <- fread(paste0("output/", date, "_optvalue_online.csv"))
effect_sizes <- c('strong', 'moderate', 'weak')
effect_sizes_labels <- c('Strong', 'Moderate', 'Weak')
dat <- dat[dat$`Average Reward`>0,]
dat$Method_short <- dat$Method
dat$Method_short <- factor(dat$Method_short, levels = c("proposed", "oracle", "overall", "changepoint_only", "cluster_only"),
                     labels = c("(i)", "(ii)", "(iii)", "(iv)", "(v)"))
dat$Method <- factor(dat$Method, levels = c("proposed", "oracle", "overall", "changepoint_only", "cluster_only"),
                     labels = c("(i) Proposed", "(ii) Oracle", "(iii) DH", "(iv) Homogeneous", "(v) Stationary"))
labels(dat$Method_short)
dat <- dat[`Effect Size` %in% effect_sizes,]
dat$`Effect Size` <- factor(dat$`Effect Size`,
                             levels = effect_sizes,
                             labels = effect_sizes_labels)

(p <- ggplot(dat, aes(Method_short, `Discounted Reward`, fill=Method)) + #, color=`Effect Size`
   # geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red") +
   geom_boxplot() +
  facet_grid(. ~ `Effect Size`) +
    xlab("") +
   ylab("Discounted Reward") +
   # labs(fill="Method") +
   theme(
     # legend.direction="horizontal",
     # legend.position = "right",
     panel.border = element_rect(color = "black", fill = NA, size = 1),
     # axis.line=element_line(size=1, colour="black"),
     panel.grid.major=element_line(colour="#d3d3d3"),
     panel.grid.minor=element_line(colour="#d3d3d3"),
     panel.background=element_blank(),
     plot.title=element_text(size=18, face="bold"),
     text=element_text(size=14),
     axis.text.x=element_text(colour="black", size=13, angle = 0),
     axis.text.y=element_text(colour="black", size=13),
     plot.margin=grid::unit(c(0.3,0,-2,0), "mm")
   ) +
   scale_fill_startrek())
   # scale_fill_brewer(type="qual", palette="Accent")) #=+
   # scale_y_continuous(limits = lfun, expand=expansion(0,0)))
ggsave(paste0("output/real_optvalue_online_discounted_raw.pdf"), width = 10, height = 3)


(p <- ggplot(dat, aes(Method_short, `Average Reward`, fill=Method)) + #, color=`Effect Size`
   # geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red") +
   geom_boxplot() +
   facet_grid(. ~ `Effect Size`) +
   xlab("") +
   ylab("Average Reward") +
   # labs(fill="Method") +
   theme(
     # legend.direction="horizontal",
     # legend.position = "bottom",
     # panel.border=element_blank(),
     # legend.box.spacing=0.4,
     panel.border = element_rect(color = "black", fill = NA, size = 1),
     # axis.line=element_line(size=1, colour="black"),
     panel.grid.major=element_line(colour="#d3d3d3"),
     panel.grid.minor=element_line(colour="#d3d3d3"),
     panel.background=element_blank(),
     plot.title=element_text(size=18, face="bold"),
     text=element_text(size=14),
     axis.text.x=element_text(colour="black", size=13, angle = 0),
     axis.text.y=element_text(colour="black", size=13),
     plot.margin=grid::unit(c(0.3,0,-2,0), "mm")
   ) +
   scale_fill_startrek())
# scale_fill_brewer(type="qual", palette="Accent")) #=+
# scale_y_continuous(limits = lfun, expand=expansion(0,0)))
ggsave(paste0("output/real_optvalue_online_average_raw.pdf"), width = 10, height = 3)












dat <- fread(paste0("output/", date, "_optvalue_online.csv"))
dat <- dat[dat$`Average Reward`>0,]
dat_proposed <- dat[Method == "proposed", ]
colnames(dat_proposed)[4:5] <- c("proposed_discounted", "proposed_average")
dat <- dat[`Effect Size` %in% effect_sizes,]
dat <- merge(x = dat, y = dat_proposed[, -"Method"],
             by = c("Effect Size","seed"), all.x = TRUE)
dat[,value_diff_discounted := proposed_discounted - `Discounted Reward`]
dat[,value_diff_average := proposed_average - `Average Reward`]

dat2 <- dat[Method != "proposed",]
dat2$Method <- paste0("proposed - ", dat2$Method)

dat2$Method_short <- dat2$Method
dat2$Method_short <- factor(dat2$Method_short,
                      levels = c('proposed - oracle', 'proposed - overall',
                                 "proposed - changepoint_only", 'proposed - cluster_only'),
                      labels = c('(i)', '(ii)', '(iii)', '(iv)'))
dat2$Method <- factor(dat2$Method,
                       levels = c('proposed - oracle', 'proposed - overall',
                                  "proposed - changepoint_only", 'proposed - cluster_only'),
                       labels = c('(i) Proposed - Oracle', '(ii) Proposed - DH',
                                  '(iii) Proposed - Homogeneous', '(iv) Proposed - Stationary'))
dat2$`Effect Size` <- factor(dat2$`Effect Size`,
                       levels = effect_sizes,
                       labels = effect_sizes_labels)

(p <- ggplot(dat2, aes(Method_short, value_diff_average, fill=Method)) + #, color=`Effect Size`
    # geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red") +
    geom_boxplot() +
    facet_grid(. ~ `Effect Size`) +
    xlab("") +
    ylab("Average Reward") +
    # labs(fill="Method") +
    theme(
      # legend.direction="horizontal",
      # legend.position = "bottom",
      # panel.border=element_blank(),
      # legend.box.spacing=0.4,
      panel.border = element_rect(color = "black", fill = NA, size = 1),
      # axis.line=element_line(size=1, colour="black"),
      panel.grid.major=element_line(colour="#d3d3d3"),
      panel.grid.minor=element_line(colour="#d3d3d3"),
      panel.background=element_blank(),
      plot.title=element_text(size=18, face="bold"),
      text=element_text(size=14),
      axis.text.x=element_text(colour="black", size=13, angle = 0),
      axis.text.y=element_text(colour="black", size=13),
      plot.margin=grid::unit(c(0.3,0,-2,0), "mm")
    ) +
    scale_fill_startrek())
ggsave(paste0("output/real_optvalue_online_average.pdf"), width = 10, height = 3)


(p <- ggplot(dat2, aes(Method_short, value_diff_discounted, fill=Method)) + #, color=`Effect Size`
    geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red") +
    geom_boxplot() +
    facet_grid(. ~ `Effect Size`) +
    xlab("") +
    ylab("Discounted Reward \nDifferences") +
    scale_y_continuous(limits = c(-15, 35)) +
    # labs(fill="Method") +
    theme(
      # legend.direction="horizontal",
      # legend.position = "bottom",
      # panel.border=element_blank(),
      # legend.box.spacing=0.4,
      panel.border = element_rect(color = "black", fill = NA, size = 1),
      # axis.line=element_line(size=1, colour="black"),
      panel.grid.major=element_line(colour="#d3d3d3"),
      panel.grid.minor=element_line(colour="#d3d3d3"),
      panel.background=element_blank(),
      plot.title=element_text(size=18, face="bold"),
      text=element_text(size=14),
      axis.text.x=element_text(colour="black", size=13, angle = 0),
      axis.text.y=element_text(colour="black", size=13),
      plot.margin=grid::unit(c(0.3,0,-2,0), "mm")
    ) +
    scale_fill_startrek())
ggsave(paste0("output/real_optvalue_online_discounted.pdf"), width = 10, height = 2)



