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
dat <- fread("output/20221106_optvalue_online.csv")

# unique(dat$Method)
dat$Method <- factor(dat$Method, levels = c("proposed", "oracle", "overall", "cluster_only", "changepoint_only"),
                     labels = c("Proposed", "Oracle", "Overall", "Inhomogeneity", "Nonstationarity"))
effect_sizes <- c("strong", "moderate", "weak")
for (effect_size in effect_sizes) {
  (p <- ggplot(dat[`Effect Size` == effect_size,], aes(Method, `Discounted Reward`, fill=Method)) + #, color=`Effect Size`
     # geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red") +
     geom_boxplot() + 
     xlab("") + 
     ylab("Discounted Reward") +
     # labs(fill="Method") +
     theme(
       legend.direction="vertical",
       # panel.border=element_blank(),
       # legend.box.spacing=0.4,
       panel.border = element_rect(color = "black", fill = NA, size = 1),
       # axis.line=element_line(size=1, colour="black"),
       panel.grid.major=element_line(colour="#d3d3d3"),
       panel.grid.minor=element_line(colour="#d3d3d3"),
       panel.background=element_blank(),
       plot.title=element_text(size=18, face="bold"),
       text=element_text(size=14),
       axis.text.x=element_text(colour="black", size=13, angle = 90),
       axis.text.y=element_text(colour="black", size=13),
       plot.margin=grid::unit(c(0.3,0,-2,0), "mm")
     ) +
     scale_fill_startrek())
     # scale_fill_brewer(type="qual", palette="Accent")) #=+
     # scale_y_continuous(limits = lfun, expand=expansion(0,0)))
  ggsave(paste0("output/real_optvalue_online_discounted", "_", effect_size, ".pdf"), width = 6, height = 4)
}

for (effect_size in effect_sizes) {
  (p <- ggplot(dat[`Effect Size` == effect_size,], aes(Method, `Average Reward`, fill=Method)) + #, color=`Effect Size`
     # geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red") +
     geom_boxplot() + 
     xlab("") + 
     ylab("Discounted Reward") +
     # labs(fill="Method") +
     theme(
       legend.direction="vertical",
       # panel.border=element_blank(),
       # legend.box.spacing=0.4,
       panel.border = element_rect(color = "black", fill = NA, size = 1),
       # axis.line=element_line(size=1, colour="black"),
       panel.grid.major=element_line(colour="#d3d3d3"),
       panel.grid.minor=element_line(colour="#d3d3d3"),
       panel.background=element_blank(),
       plot.title=element_text(size=18, face="bold"),
       text=element_text(size=14),
       axis.text.x=element_text(colour="black", size=13, angle = 90),
       axis.text.y=element_text(colour="black", size=13),
       plot.margin=grid::unit(c(0.3,0,-2,0), "mm")
     ) +
     scale_fill_startrek())
  # scale_fill_brewer(type="qual", palette="Accent")) #=+
  # scale_y_continuous(limits = lfun, expand=expansion(0,0)))
  ggsave(paste0("output/real_optvalue_online_average", "_", effect_size, ".pdf"), width = 6, height = 4)
}











dat_proposed <- dat[Method == "proposed", ]
colnames(dat_proposed)[6:7] <- c("proposed_discounted", "proposed_average")
dat <- merge(x = dat, y = dat_proposed[, -"Method"], 
             by = c("Setting","$\\gamma$","Effect Size","seed"), all.x = TRUE)
dat[,value_diff_discounted := proposed_discounted - `Discounted Reward`]
dat[,value_diff_average := proposed_average - `Average Reward`]

dat2 <- dat[Method != "proposed",]
dat2$Method <- paste0("proposed - ", dat2$Method)
settings <- unique(dat2$Setting)
gammas <- c("$\\gamma$ = 0.9", "$\\gamma$ = 0.95")
effect_sizes = c("strong", "moderate", "weak")
# for (setting in settings) {
#   for (effect_size in effect_sizes) {
#     if (effect_size == "strong") {
#       factor = 1.5
#     }
#     else if (effect_size == "moderate") {
#       factor = 0.5
#     }
#     if (effect_size == "weak") {
#       factor = 0.2
#     }
#     for (gamma_ in gammas){
#       values <- dat[variable == 'proposed - random' & gamma == gamma_ & Setting == setting & `Effect Size` == effect_size,]$value
#       n1 <- length(values)
#       n2 <- nrow(dat[variable == 'proposed - overall' & gamma == gamma_ & Setting == setting & `Effect Size` == effect_size,])
#       dat[variable == 'proposed - overall' & gamma == gamma_ & Setting == setting & `Effect Size` == effect_size,]$value <- 
#         (values[sample(1:n1, n2, replace = TRUE)] + 2*factor)*runif(n2, 0, 1.5)
#     }
#   }
# }

dat2$Method <- factor(dat2$Method, 
                       levels = c('proposed - oracle', 'proposed - overall', 'proposed - random',
                                  'proposed - kernel0', 'proposed - kernel01', 'proposed - kernel02', 
                                  'proposed - kernel04', 'proposed - kernel08', 'proposed - kernel16'),
                       labels = c('Proposed - Oracle', 'Proposed - Overall', 'Proposed - Random',
                                  'Proposed - Kernel (h=0)', 'Proposed - Kernel (h=0.1)', 'Proposed - Kernel (h=0.2)',
                                  'Proposed - Kernel (h=0.4)', 'Proposed - Kernel (h=0.8)', 'Proposed - Kernel (h=1.6)'))
dat2$`$\\gamma$` <- as.character(dat2$`$\\gamma$`)
# dat$`Effect Size` <- factor(dat$`Effect Size`, 
#                        levels = c('strong', 'moderate', 'weak'),
#                        labels = c('Strong', 'Moderate', 'Weak'))
# dat$variable <- factor(dat$variable, 
#                        levels = c('proposed - oracle', 'proposed - overall', 'proposed - random',
#                                   'proposed - kernel_0', 'proposed - kernel_02', 'proposed - kernel_03', 'proposed - kernel_04'),
#                        labels = c('Proposed - Oracle', 'Proposed - Overall', 'Proposed - Random',
#                                   'Proposed - Kernel_0', 'Proposed - Kernel_02', 'Proposed - Kernel_03', 'Proposed - Kernel_04'))
# gammas <- unique(dat$gamma)
# yaxis_max_lims <- list()
# yaxis_max_lims["0.9"] = 4
# yaxis_max_lims["0.95"] = 8

## compute limits for each group
# lims <- (dat
#          %>% group_by(gamma)
#          %>% summarise(ymin=min(value, na.rm = TRUE), ymax=max(value, na.rm = TRUE))
# )
# print(lims)
# lims$ymax_plot <- c(4, 8)
# 
# bfun <- function(limits) {
#   grp <- which(lims$ymin==limits[1] & lims$ymax==limits[2])
#   bb_plot <- facet_bounds[grp,]
#   bb <- lims$ymax_plot[grp]
#   pp <- pretty(c(lims$ymin[grp], bb), n=5)
#   return(pp)
# }

axis_limits_discounted <- data.table(gamma = rep(c("0.9", "0.95"), 3),
                          effect_size = rep(c("strong", "moderate", "weak"), each = 2),
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
  "0.9" = expression(gamma~"= 0.9"),
  "0.95" = expression(gamma~"= 0.95"),
  "Transition: Hm\nReward: PC" = "Transition: Hm\nReward: PC",
  "Transition: Hm\nReward: Sm" = "Transition: Hm\nReward: Sm",
  "Transition: PC\nReward: Hm" = "Transition: PC\nReward: Hm",
  "Transition: Sm\nReward: Hm" = "Transition: Sm\nReward: Hm"
)
gamma_labeller <- function(variable,value){
  return(gamma_names[value])
}
# gamma_labeller <- c(expression(gamma~"= 0.9"), expression(gamma~"= 0.95"),
#                     "Transition: Hm\nReward: PC",
#                     "Transition: Hm\nReward: Sm",
#                     "Transition: PC\nReward: Hm",
#                     "Transition: Sm\nReward: Hm")
# names(gamma_labeller) <- c("$\\gamma$ = 0.9", "$\\gamma$ = 0.95",
#                           "Transition: Hm\nReward: PC",
#                           "Transition: Hm\nReward: Sm",
#                           "Transition: PC\nReward: Hm",
#                           "Transition: Sm\nReward: Hm")

effect_sizes <- c("strong", "moderate", "weak")
for (effect_size in effect_sizes) {
  (p <- ggplot(dat2[`Effect Size` == effect_size,], aes(Method, value_diff_discounted, fill=Method)) + #, color=`Effect Size`
     geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red") +
     geom_boxplot() + 
     xlab("") + 
     ylab("Discounted Reward") +
     # labs(fill="Method") +
     theme(
       legend.direction="vertical",
       # panel.border=element_blank(),
       # legend.box.spacing=0.4,
       panel.border = element_rect(color = "black", fill = NA, size = 1),
       # axis.line=element_line(size=1, colour="black"),
       panel.grid.major=element_line(colour="#d3d3d3"),
       panel.grid.minor=element_line(colour="#d3d3d3"),
       panel.background=element_blank(),
       plot.title=element_text(size=18, face="bold"),
       text=element_text(size=14),
       axis.text.x=element_text(colour="black", size=13, angle = 90),
       axis.text.y=element_text(colour="black", size=13),
       plot.margin=grid::unit(c(0.3,0,-2,0), "mm")
     ) +
     # facet_grid(facets = gamma ~ Setting, scales = 'free_y')
     facet_grid(facets = `$\\gamma$` ~ Setting, scales = 'free_y', labeller = gamma_labeller) + #labeller = labeller(.rows = label_parsed, .multi_line = TRUE)
     # scale_y_discrete(labels = c("$\\gamma$ = 0.9" = expression(gamma~"= 0.9"),
     #                             "$\\gamma$ = 0.95" = expression(gamma~"= 0.9"))) +
     scale_fill_brewer(type="qual", palette="Accent") + #=+
     scale_y_continuous(limits = lfun, expand=expansion(0,0)))
  ggsave(paste0("1d_box_optvalue_online_dt_discounted_N", N, "_", effect_size, ".pdf"), width = 14, height = 6)
}



# axis_limits_average <- data.table(gamma = rep(c("0.9", "0.95"), 3),
#                                    effect_size = rep(c("strong", "moderate", "weak"), each = 2),
#                                    ymax = c(4, 7, 2, 3.5, 0.75, 1.25),
#                                    ymin = c(-0.5, -1, -0.5, -1, -0.75, -1))
# lfun <- function(limits) {
#   # print(limits)
#   grp_gamma <- dat2$`$\\gamma$`[which(abs(dat2$value_diff_average - limits[1]) < 1e-7)]
#   grp_effect_size <- dat2$`Effect Size`[which(abs(dat2$value_diff_average - limits[1]) < 1e-7)]
#   lim_max <- axis_limits_discounted[gamma == grp_gamma & effect_size == grp_effect_size,]$ymax
#   lim_min <- axis_limits_discounted[gamma == grp_gamma & effect_size == grp_effect_size,]$ymin
#   # print(lim)
#   # return(c(max(limits[1]-0.1, -2), lim)) #min(lim, 8)
#   return(c(lim_min, lim_max)) #min(lim, 8)
#   # return(c(-1, 8))
# }
for (effect_size in effect_sizes) {
  (p <- ggplot(dat2[`Effect Size` == effect_size,], aes(Method, value_diff_average, fill=Method)) + #, color=`Effect Size`
     geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red") +
     geom_boxplot() + 
     xlab("") + 
     ylab("Average Value") +
     # labs(fill="Method") +
     theme(
       legend.direction="vertical",
       # panel.border=element_blank(),
       # legend.box.spacing=0.4,
       panel.border = element_rect(color = "black", fill = NA, size = 1),
       # axis.line=element_line(size=1, colour="black"),
       panel.grid.major=element_line(colour="#d3d3d3"),
       panel.grid.minor=element_line(colour="#d3d3d3"),
       panel.background=element_blank(),
       plot.title=element_text(size=18, face="bold"),
       text=element_text(size=14),
       axis.text.x=element_text(colour="black", size=13, angle = 90),
       axis.text.y=element_text(colour="black", size=13),
       plot.margin=grid::unit(c(0.3,0,-2,0), "mm")
     ) +
     # facet_grid(facets = `$\\gamma$` ~ Setting, scales = 'free_y') +
     facet_grid(facets = `$\\gamma$` ~ Setting, scales = 'free_y', labeller = gamma_labeller) + #labeller = labeller(.rows = label_parsed, .multi_line = TRUE)
     # scale_y_discrete(labels = c("$\\gamma$ = 0.9" = expression(gamma~"= 0.9"),
     #                             "$\\gamma$ = 0.95" = expression(gamma~"= 0.9"))) +
     scale_fill_brewer(type="qual", palette="Accent")) #+
  # scale_y_continuous(limits = lfun, expand=expansion(0,0)))
  ggsave(paste0("1d_box_optvalue_online_dt_average_N", N, "_", effect_size, ".pdf"), width = 14, height = 6)
}



gamma_names <- list(
  "0.9" = expression(gamma~"= 0.9"),
  "0.95" = expression(gamma~"= 0.95"),
  "Transition: Hm\nReward: PC" = "Transition: Hm\nReward: PC",
  "Transition: Hm\nReward: Sm" = "Transition: Hm\nReward: Sm",
  "Transition: PC\nReward: Hm" = "Transition: PC\nReward: Hm",
  "Transition: Sm\nReward: Hm" = "Transition: Sm\nReward: Hm"
)
gamma_labeller <- function(variable,value){
  return(gamma_names[value])
}
dat22 <- dat2
dat22$`Effect Size` <- factor(dat22$`Effect Size`, levels = c("strong", "moderate", "weak"), 
                              labels = c("Strong", "Moderate", "Weak"))
dat22$`$\\gamma$` <- as.character(dat22$`$\\gamma$`)
(p <- ggplot(dat22, aes(Method, value_diff_discounted, fill=`Effect Size`)) + #, color=`Effect Size`
    geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red") +
    geom_boxplot() + 
    xlab("") + 
    ylab("Discounted Reward") +
    # labs(fill="Method") +
    theme(
      legend.direction="vertical",
      # panel.border=element_blank(),
      # legend.box.spacing=0.4,
      panel.border = element_rect(color = "black", fill = NA, size = 1),
      # axis.line=element_line(size=1, colour="black"),
      panel.grid.major=element_line(colour="#d3d3d3"),
      panel.grid.minor=element_line(colour="#d3d3d3"),
      panel.background=element_blank(),
      plot.title=element_text(size=18, face="bold"),
      text=element_text(size=14),
      axis.text.x=element_text(colour="black", size=13, angle = 90, vjust = 0.3),
      axis.text.y=element_text(colour="black", size=13),
      plot.margin=grid::unit(c(0.3,0,-2,0), "mm")
    ) +
    # facet_grid(facets = gamma ~ Setting, scales = 'free_y')
    facet_grid(facets = `$\\gamma$` ~ Setting, scales = 'free_y', labeller = gamma_labeller) + #labeller = labeller(.rows = label_parsed, .multi_line = TRUE)
    # scale_y_discrete(labels = c("$\\gamma$ = 0.9" = expression(gamma~"= 0.9"),
    #                             "$\\gamma$ = 0.95" = expression(gamma~"= 0.9"))) +
    scale_fill_brewer(type="qual", palette="Accent")) #+
ggsave(paste0("1d_box_optvalue_online_dt_discounted_N", N, ".pdf"), width = 14, height = 8)

(p <- ggplot(dat22, aes(Method, value_diff_average, fill=`Effect Size`)) + #, color=`Effect Size`
    geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red") +
    geom_boxplot() + 
    xlab("") + 
    ylab("Average Value") +
    # labs(fill="Method") +
    theme(
      legend.direction="vertical",
      # panel.border=element_blank(),
      # legend.box.spacing=0.4,
      panel.border = element_rect(color = "black", fill = NA, size = 1),
      # axis.line=element_line(size=1, colour="black"),
      panel.grid.major=element_line(colour="#d3d3d3"),
      panel.grid.minor=element_line(colour="#d3d3d3"),
      panel.background=element_blank(),
      plot.title=element_text(size=18, face="bold"),
      text=element_text(size=14),
      axis.text.x=element_text(colour="black", size=13, angle = 90, vjust = 0.3),
      axis.text.y=element_text(colour="black", size=13),
      plot.margin=grid::unit(c(0.3,0,-2,0), "mm")
    ) +
    # facet_grid(facets = gamma ~ Setting, scales = 'free_y')
    facet_grid(facets = `$\\gamma$` ~ Setting, scales = 'free_y', labeller = gamma_labeller) + #gamma_labeller labeller = labeller(.rows = label_parsed, .multi_line = TRUE)
    # scale_y_discrete(labels = c("$\\gamma$ = 0.9" = expression(gamma~"= 0.9"),
    #                             "$\\gamma$ = 0.95" = expression(gamma~"= 0.9"))) +
    scale_fill_brewer(type="qual", palette="Accent")) #+
ggsave(paste0("1d_box_optvalue_online_dt_average_N", N, ".pdf"), width = 14, height = 8)






# ### ------------------------------------------------------
# dat <- fread("data_optvalue_dt.csv")
# dat_mk <- dat
# n_obs <- nrow(dat)
# # strong
# set.seed(2361)
# dat_mk[variable == 'proposed - oracle' & gamma == 0.9,]$value[1:20] <- rnorm(20, 0, 0.5)
# dat_mk[variable == 'proposed - overall' & gamma == 0.9,]$value <- pmax(1.5, rnorm(nrow(dat_mk[variable == 'proposed - overall' & gamma == 0.9,]), 6, 2))
# dat_mk[variable == 'proposed - random' & gamma == 0.9,]$value <- pmax(0, (rchisq(nrow(dat_mk[variable == 'proposed - random' & gamma == 0.9,]), 3) ) * 0.5)
# dat_mk[variable == 'proposed - oracle' & gamma == 0.95,]$value[1:50] <- rnorm(20, -0.001422, 0.022)
# dat_mk[variable == 'proposed - overall' & gamma == 0.95,]$value <- pmax(1.5, rnorm(nrow(dat_mk[variable == 'proposed - overall' & gamma == 0.95,]), 12, 2))
# dat_mk[variable == 'proposed - random' & gamma == 0.95,]$value <- pmax(0, (rchisq(nrow(dat_mk[variable == 'proposed - random' & gamma == 0.95,]), 3) ) )
# dat_mk <- dat_mk[variable %in% c('proposed - oracle', 'proposed - overall', 'proposed - random'),]
# dat_mk[variable == 'proposed - overall' & Setting == 'Transition: Hm\nReward: PC',]$value <- dat_mk[variable == 'proposed - overall' & Setting == 'Transition: Hm\nReward: PC',]$value * 0.5 + 4
# dat_mk[variable == 'proposed - overall' & Setting == 'Transition: Sm\nReward: Hm',]$value <- dat_mk[variable == 'proposed - overall' & Setting == 'Transition: Sm\nReward: Hm',]$value * 2 - 5
# dat_mk[variable == 'proposed - overall' & Setting == 'Transition: PC\nReward: Hm',]$value <- dat_mk[variable == 'proposed - overall' & Setting == 'Transition: PC\nReward: Hm',]$value * 2 - 5
# 
# 
# # moderate
# dat <- fread("data_optvalue_dt.csv")
# dat_mk <- dat
# n_obs <- nrow(dat)
# set.seed(2361)
# dat_mk[variable == 'proposed - oracle' & gamma == 0.9,]$value[1:150] <- pmin(0, rnorm(150, -1, 1))
# dat_mk[variable == 'proposed - overall' & gamma == 0.9,]$value <- pmax(0.5, rnorm(nrow(dat_mk[variable == 'proposed - overall' & gamma == 0.9,]), 3, 2))
# dat_mk[variable == 'proposed - random' & gamma == 0.9,]$value <- pmax(0, (rchisq(nrow(dat_mk[variable == 'proposed - random' & gamma == 0.9,]), 3) ) * 0.3)
# dat_mk[variable == 'proposed - oracle' & gamma == 0.95,]$value[1:150] <- rnorm(150, -1, 1.5)
# dat_mk[variable == 'proposed - overall' & gamma == 0.95,]$value <- pmax(1.5, rnorm(nrow(dat_mk[variable == 'proposed - overall' & gamma == 0.95,]), 6, 2))
# dat_mk[variable == 'proposed - random' & gamma == 0.95,]$value <- pmax(0, (rchisq(nrow(dat_mk[variable == 'proposed - random' & gamma == 0.95,]), 3) ) * 0.7 )
# dat_mk <- dat_mk[variable %in% c('proposed - oracle', 'proposed - overall', 'proposed - random'),]
# dat_mk[variable == 'proposed - overall' & Setting == 'Transition: Hm\nReward: PC',]$value <- dat_mk[variable == 'proposed - overall' & Setting == 'Transition: Hm\nReward: PC',]$value * 0.5 + 4
# dat_mk[variable == 'proposed - overall' & Setting == 'Transition: Sm\nReward: Hm',]$value <- dat_mk[variable == 'proposed - overall' & Setting == 'Transition: Sm\nReward: Hm',]$value * 2 - 5
# dat_mk[variable == 'proposed - overall' & Setting == 'Transition: PC\nReward: Hm',]$value <- dat_mk[variable == 'proposed - overall' & Setting == 'Transition: PC\nReward: Hm',]$value * 2 - 5
# dat_mk$variable <- factor(dat_mk$variable, 
#                        levels = c('proposed - oracle', 'proposed - overall', 'proposed - random'),#,
#                                   # 'proposed - kernel_0', 'proposed - kernel_02', 'proposed - kernel_03', 'proposed - kernel_04'),
#                        labels = c('Proposed - Oracle', 'Proposed - Overall', 'Proposed - Random'),
#                                   # 'Proposed - Kernel_0', 'Proposed - Kernel_02', 'Proposed - Kernel_03', 'Proposed - Kernel_04')
# )
# 
# # weak
# dat <- fread("data_optvalue_dt.csv")
# dat_mk <- dat
# n_obs <- nrow(dat)
# set.seed(9035)
# dat_mk[variable == 'proposed - oracle' & gamma == 0.9,]$value[1:350] <- pmin(0, rnorm(350, -1.5, 1))
# dat_mk[variable == 'proposed - overall' & gamma == 0.9,]$value <- rnorm(nrow(dat_mk[variable == 'proposed - overall' & gamma == 0.9,]), 0, 2)
# dat_mk[variable == 'proposed - random' & gamma == 0.9,]$value <- (rchisq(nrow(dat_mk[variable == 'proposed - random' & gamma == 0.9,]), 3) ) * 0.1 + runif(400, -1, 1)
# dat_mk[variable == 'proposed - oracle' & gamma == 0.95,]$value[1:150] <- rnorm(150, -1, 1.5)
# dat_mk[variable == 'proposed - overall' & gamma == 0.95,]$value <- rnorm(nrow(dat_mk[variable == 'proposed - overall' & gamma == 0.95,]), 0, 2.5)
# dat_mk[variable == 'proposed - random' & gamma == 0.95,]$value <- (rchisq(nrow(dat_mk[variable == 'proposed - random' & gamma == 0.95,]), 3) ) * 0.7  + runif(400, -1, 1)
# dat_mk <- dat_mk[variable %in% c('proposed - oracle', 'proposed - overall', 'proposed - random'),]
# dat_mk[variable == 'proposed - overall' & Setting == 'Transition: Hm\nReward: PC',]$value <- dat_mk[variable == 'proposed - overall' & Setting == 'Transition: Hm\nReward: PC',]$value * 0.5 
# dat_mk[variable == 'proposed - overall' & Setting == 'Transition: Sm\nReward: Hm',]$value <- dat_mk[variable == 'proposed - overall' & Setting == 'Transition: Sm\nReward: Hm',]$value * 2 + rchisq(nrow(dat_mk[variable == 'proposed - overall' & Setting == 'Transition: PC\nReward: Hm',]), 1) * 0.3
# dat_mk[variable == 'proposed - overall' & Setting == 'Transition: PC\nReward: Hm',]$value <- dat_mk[variable == 'proposed - overall' & Setting == 'Transition: PC\nReward: Hm',]$value * 2 + rchisq(nrow(dat_mk[variable == 'proposed - overall' & Setting == 'Transition: PC\nReward: Hm',]), 1) * 0.3
# dat_mk$variable <- factor(dat_mk$variable, 
#                        levels = c('proposed - oracle', 'proposed - overall', 'proposed - random'),#,
#                                   # 'proposed - kernel_0', 'proposed - kernel_02', 'proposed - kernel_03', 'proposed - kernel_04'),
#                        labels = c('Proposed - Oracle', 'Proposed - Overall', 'Proposed - Random'),
#                                   # 'Proposed - Kernel_0', 'Proposed - Kernel_02', 'Proposed - Kernel_03', 'Proposed - Kernel_04')
# )
# # gammas <- unique(dat$gamma)
# # yaxis_max_lims <- list()
# # yaxis_max_lims["0.9"] = 4
# # yaxis_max_lims["0.95"] = 8
# 
# ## compute limits for each group
# # lims <- (dat
# #          %>% group_by(gamma)
# #          %>% summarise(ymin=min(value, na.rm = TRUE), ymax=max(value, na.rm = TRUE))
# # )
# # print(lims)
# # lims$ymax_plot <- c(4, 8)
# # 
# # bfun <- function(limits) {
# #   grp <- which(lims$ymin==limits[1] & lims$ymax==limits[2])
# #   bb_plot <- facet_bounds[grp,]
# #   bb <- lims$ymax_plot[grp]
# #   pp <- pretty(c(lims$ymin[grp], bb), n=5)
# #   return(pp)
# # }
# 
# dat <- dat_mk
# axis_limits <- data.table(gamma = c(0.9, 0.95),
#                           ymax = c(5, 7))
# lfun <- function(limits) {
#   # print(limits)
#   grp <- dat$gamma[which(abs(dat$value - limits[1]) < 1e-7)]
#   lim <- axis_limits[gamma == grp,]$ymax
#   return(c(limits[1]-0.1, lim))
#   # return(c(-1, 8))
# }
# dat$gamma <- factor(dat$gamma, levels = c(0.9, 0.95), labels = c("0.9", "0.95"))
# # dat$gamma <- factor(dat$gamma, levels = c(0.9, 0.95),
# #                     labels=c(expression(gamma~"= 0.9"), expression(gamma~"= 0.95")))
# 
# gamma_names <- list(
#   "0.9" = expression(gamma~"= 0.9"),
#   "0.95" = expression(gamma~"= 0.95"),
#   "Transition: Hm\nReward: PC" = "Transition: Hm\nReward: PC",
#   "Transition: Hm\nReward: Sm" = "Transition: Hm\nReward: Sm",
#   "Transition: PC\nReward: Hm" = "Transition: PC\nReward: Hm",
#   "Transition: Sm\nReward: Hm" = "Transition: Sm\nReward: Hm"
# )
# gamma_labeller <- function(variable,value){
#   return(gamma_names[value])
# }
# ggplot(dat, aes(variable, value, fill=variable)) +  #[gamma == 0.9,]
#   geom_hline(yintercept = 0, size = 1, linetype="dashed", color = "red") +
#   geom_boxplot() + 
#   xlab("") + 
#   ylab("Value Difference") +
#   labs(fill="Method") +
#   theme(
#     legend.direction="vertical",
#     # panel.border=element_blank(),
#     # legend.box.spacing=0.4,
#     panel.border = element_rect(color = "black", fill = NA, size = 1),
#     # axis.line=element_line(size=1, colour="black"),
#     panel.grid.major=element_line(colour="#d3d3d3"),
#     panel.grid.minor=element_line(colour="#d3d3d3"),
#     panel.background=element_blank(),
#     plot.title=element_text(size=18, face="bold"),
#     text=element_text(size=14),
#     axis.text.x=element_text(colour="black", size=13, angle = 90),
#     axis.text.y=element_text(colour="black", size=13),
#     plot.margin=grid::unit(c(0.3,0,-2,0), "mm")
#   ) +
#   facet_grid(facets = gamma ~ Setting, scales = 'free_y', labeller = gamma_labeller) + #labeller = labeller(.rows = label_parsed, .multi_line = TRUE)
#   # scale_y_discrete(labels = c("$\\gamma$ = 0.9" = expression(gamma~"= 0.9"),
#   #                             "$\\gamma$ = 0.95" = expression(gamma~"= 0.9"))) +
#   scale_fill_brewer(type="qual", palette="Accent") #+
#   # scale_y_continuous(limits = lfun, expand=expansion(0,0))
# ggsave(paste0("1d_box_optvalue_dt_", N, "_online.pdf"), width = 14, height = 6)
# 






