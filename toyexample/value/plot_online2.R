#!/sw/arcts/centos7/stacks/gcc/8.2.0/R/4.1.0/bin/Rscript
library(data.table)
library(ggplot2)
library(dplyr)
library(ggbreak)
library("ggsci")
args = commandArgs(trailingOnly=TRUE)
# N <- as.integer(args[1])
if(Sys.info()["nodename"] %in% c("PC-20181212HEBU")){
  curr_dir <- "C:/Users/test/Dropbox/tml/IHS/simu/simu/toyexample/value"
  setwd(curr_dir)
} else{ # greatlakes
  curr_dir <- "/home/mengbing/research/RL_nonstationary/code2/simulation_nonstationary_changept_detection/output"
  setwd(curr_dir)
}

N <- 50
set.seed(20)
dat <- fread("vall_2022-11-05N50_1d.csv")
dat$Setting <- factor(dat$Setting, 
                      levels = c('pwconst2', 'smooth'),
                      labels = c('Piecewise Constant', 'Smooth'))
dat[,'Average Value'] = dat[,'Average Value'] * 10
dat[,'Discounted Value'] = dat[,'Discounted Value'] *(1-0.9)
dat_proposed <- dat[init == "proposed", ]
colnames(dat_proposed)[4:5] <- c("proposed_average","proposed_discounted")
dat <- merge(x = dat, y = dat_proposed[, -"init"], 
             by = c("Setting","seed"), all.x = TRUE)
dat[,value_diff_discounted := proposed_discounted - `Discounted Value`]
dat[,value_diff_average := proposed_average - `Average Value`]

dat2 <- dat[init != "proposed",]
dat2$init <- paste0("proposed - ", dat2$init)
settings <- unique(dat2$Setting)
# gammas <- c("$\\gamma$ = 0.9", "$\\gamma$ = 0.95")
# effect_sizes = c("strong", "moderate", "weak")

dat2$init <- factor(dat2$init, 
                    levels = c('proposed - oracle', 'proposed - overall', 'proposed - only_cp', 'proposed - only_clusters'),
                    labels = c('Proposed - Oracle', 'Proposed - Overall', 'Proposed - Nonstationary','Proposed - Heterogeneous'))
# dat2$`$\\gamma$` <- as.character(dat2$`$\\gamma$`)
axis_limits_discounted <- data.table(
  ymax = c(4, 7, 2, 3.5, 0.75, 1.25),
  ymin = c(-0.5, -1, -0.5, -1, -0.75, -1))
lfun <- function(limits) {
  # print(limits)
  grp_gamma <- dat2$`$\\gamma$`[which(abs(dat2$value_diff_discounted - limits[1]) < 1e-7)]
  grp_effect_size <- dat2$`Effect Size`[which(abs(dat2$value_diff_discounted - limits[1]) < 1e-7)]
  lim_max <- axis_limits_discounted[gamma == grp_gamma & effect_size == grp_effect_size,]$ymax
  lim_min <- axis_limits_discounted[gamma == grp_gamma & effect_size == grp_effect_size,]$ymin
  # print(lim)
  # return(c(max(limits[1]-0.1, -2), lim)) #min(lim, 8)
  return(c(lim_min, lim_max)) #min(lim, 8)
  # return(c(-1, 8))
}
# dat$gamma <- factor(dat$gamma, levels = c(0.9, 0.95), labels = c("0.9", "0.95"))
# dat$gamma <- factor(dat$gamma, levels = c(0.9, 0.95),
#                     labels=c(expression(gamma~"= 0.9"), expression(gamma~"= 0.95")))

gamma_names <- list(
  "Transition: PC\nReward: Hm" = "Transition: PC\nReward: Hm",
  "Transition: Sm\nReward: Hm" = "Transition: Sm\nReward: Hm"
)
gamma_labeller <- function(variable,value){
  return(gamma_names[value])
}
# effect_sizes <- c("strong", "moderate", "weak")
# for (effect_size in effect_sizes) {

cbPalette=c('#cc0c00', '#5c88da','#84bd00', '#ffcc00', '#7c878e','#00b5e2','#00af66',"#E69F00","#660099")
p_legend <- ggboxplot(dat2,x='init', y = 'value_diff_average',fill = 'init', alpha=0.8,
                      ylab="Discounted Value", xlab="",lwd=1, fatten=1,
                      facet.by = c('Setting'),alpha=0.8,palette = my_colors[1:4],
                      ggtheme = theme(
                        # legend.direction="vertical",
                        legend.position = "bottom",
                        # panel.border=element_blank(),
                        # legend.box.spacing=0.4,
                        panel.border = element_rect(color = "black", fill = NA, size = 1),
                        # axis.line=element_line(size=1, colour="black"),
                        panel.grid.major=element_line(colour="#d3d3d3"),
                        panel.grid.minor=element_line(colour="#d3d3d3"),
                        panel.background=element_blank(),
                        plot.title=element_text(size=18, face="bold"),
                        text=element_text(size=18),
                        legend.text = element_text(size=18),
                        # axis.text.x = element_blank(),
                        axis.text.x=element_text(colour="white", size=0, angle = 0),
                        # strip.text.y = element_blank(),
                        axis.text.y=element_text(colour="black", size=16),
                        plot.margin=grid::unit(c(0.3,0,8,0), "mm")
                      ))+
  geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red")


p_legend=ggpar(p_legend, legend.title = "")

########
cbPalette=c('#cc0c00', '#5c88da','#84bd00', '#ffcc00', '#7c878e','#00b5e2','#00af66',"#E69F00","#660099")
p_diff_av <- ggboxplot(dat2,x='init', y = 'value_diff_average',fill = 'init', alpha=0.8,
                       ylab="Average Value", xlab="",lwd=1, fatten=1,
                       facet.by = c('Setting'),alpha=0.8,palette = my_colors[1:4],
                       ggtheme = theme(
                         # legend.direction="vertical",
                         legend.position = "None",
                         # panel.border=element_blank(),
                         # legend.box.spacing=0.4,
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
                         plot.margin=grid::unit(c(0.3,0,-5,0), "mm")
                       ))+
  geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red")

p_diff_av 

p_diff_dis <- ggboxplot(dat2,x='init', y = 'value_diff_discounted',fill = 'init', alpha=0.8,
                       facet.by = c('Setting'),alpha=0.8, palette = my_colors[1:4],lwd=1, fatten=1,
                       ylab="Discounted Value", xlab="",
                       ggtheme = theme(
                         # legend.direction="vertical",
                         legend.position = "None",
                         # panel.border=element_blank(),
                         # legend.box.spacing=0.4,
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
                         plot.margin=grid::unit(c(0.3,0,-2,0), "mm")
                       ))+
  geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red")

p_diff_dis 

################### 
if(Sys.info()["nodename"] %in% c("PC-20181212HEBU")){
  curr_dir <- "C:/Users/test/Dropbox/tml/IHS/simu/simu/toyexample/value_samesign"
  setwd(curr_dir)
} else{ # greatlakes
  curr_dir <- "/home/mengbing/research/RL_nonstationary/code2/simulation_nonstationary_changept_detection/output"
  setwd(curr_dir)
}

N <- 50
set.seed(20)
dat <- fread("vall_2022-11-06N50_1d.csv")
dat$Setting <- factor(dat$Setting, 
                      levels = c('pwconst2', 'smooth'),
                      labels = c('Piecewise Constant', 'Smooth'))
dat[,'Average Value'] = dat[,'Average Value'] * 10
dat[,'Discounted Value'] = dat[,'Discounted Value'] *(1-0.9)
dat_proposed <- dat[init == "proposed", ]
colnames(dat_proposed)[4:5] <- c("proposed_average","proposed_discounted")
dat <- merge(x = dat, y = dat_proposed[, -"init"], 
             by = c("Setting","seed"), all.x = TRUE)
dat[,value_diff_discounted := proposed_discounted - `Discounted Value`]
dat[,value_diff_average := proposed_average - `Average Value`]

dat2 <- dat[init != "proposed",]
dat2$init <- paste0("proposed - ", dat2$init)
settings <- unique(dat2$Setting)
# gammas <- c("$\\gamma$ = 0.9", "$\\gamma$ = 0.95")
# effect_sizes = c("strong", "moderate", "weak")

dat2$init <- factor(dat2$init, 
                    levels = c('proposed - oracle', 'proposed - overall', 'proposed - only_cp', 'proposed - only_clusters'),
                    labels = c('Proposed - Oracle', 'Proposed - Overall', 'Proposed - Nonstationary','Proposed - Heterogeneous'))

p_same_av <- ggboxplot(dat2, x = 'init', y='value_diff_average', fill = 'init',lwd=1, fatten=1,
                       alpha=0.8, ylab="Average Value", xlab="",palette = my_colors[1:4],
                       facet.by = c('Setting'),
                       ggtheme = theme(
                         # legend.direction="vertical",
                         legend.position = "None",
                         # panel.border=element_blank(),
                         # legend.box.spacing=0.4,
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
                         plot.margin=grid::unit(c(0.3,0,-5,0), "mm")
                       ))+
  geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red")

p_same_av

p_same_dis <- ggboxplot(dat2, x = 'init', y='value_diff_discounted', fill = 'init',lwd=1, fatten=1,
                       alpha=0.8, ylab="Discounted Value", xlab="",palette = my_colors[1:4],
                       facet.by = c('Setting'),
                       ggtheme = theme(
                         # legend.direction="vertical",
                         legend.position = "None",
                         # panel.border=element_blank(),
                         # legend.box.spacing=0.4,
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
                         plot.margin=grid::unit(c(0.3,0,-5,0), "mm")
                       ))+
  geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red")

p_same_dis

########################
p_av =ggarrange(p_diff_av, p_same_av, common.legend = 1, 
          legend.grob = get_legend(p_legend), nrow = 2, labels = c('A.', 'B.'),
          legend = 'bottom',font.label = list(size = 18))
p_av
ggsave(paste0("2figs_value_online_avr", Sys.Date(), ".pdf"), width = 14, height = 5)

p_dis =ggarrange(p_diff_dis, p_same_dis, common.legend = 1, 
                legend.grob = get_legend(p_legend), nrow = 2, labels = c('A.', 'B.'),
                legend = 'bottom',font.label = list(size = 18))
p_dis
ggsave(paste0("2figs_value_online_disr", Sys.Date(), ".pdf"), width = 14, height = 5)

