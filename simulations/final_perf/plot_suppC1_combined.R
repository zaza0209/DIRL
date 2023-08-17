library(ggpubr)
library(dplyr)
library(tidyr)
library(data.table)
if(Sys.info()["nodename"] %in% c("PC-20181212HEBU")){
  curr_dir <- "C:/Users/test/Dropbox/DIRL/IHS/simu/simu/output/final_perf"
  setwd(curr_dir)
} else{ # greatlakes
  curr_dir <- "/home/xxx/research/RL_nonstationary/code2/simulation_nonstationary_changept_detection/output"
  setwd(curr_dir)
}
dat_cp =fread('icmodel_2022-11-02method(235)N50_1d.csv')
# dat_cp_2 =fread('icmodel_2022-11-02method(23)N50_1d.csv')

dat_cp$init <- factor(dat_cp$init, levels = c('best_model', 'true_change_points','random_cp_K2','no_change_points'),
                        labels = c('Model Selected via the Information Criterion', 'Oracle Change Point', 'Random Change Point', 'No Change Point'))
# dat_cp_2$init <- factor(dat_cp_2$init, levels = c('best_model', 'true_change_points','no_change_points'),
#                         labels = c('Model Selected via the Information Criterion', 'Oracle Change Point', 'No Change Point'))
# dat_cp = rbind(dat_cp_1, dat_cp_2)
# dat_cp[dat_cp$init == 'Model Selected via the Information Criterion', ] = ifelse(
#   dat_cp_2[dat_cp_2$init == 'Model Selected via the Information Criterion'])
dat_cp$Setting <- factor(dat_cp$Setting,levels = c('pwconst2', 'smooth'),
                         labels = c('Piecewise Constant', 'Smooth'))
# dat_cp$compare = rep(c('setting1', 'setting2'), each = dim(dat_cp_1)[1])
dat_cp_processed <- gather(dat_cp, metric, value, -c('Setting', 'seed', 'init'))
dat_cp_processed$metric=factor(dat_cp_processed$metric, levels = c('cp_err', 'ARI'),
                               labels =  c('CP Error', 'ARI'))
my_colors=c('#cc0c00', '#5c88da','#84bd00', '#ffcc00', '#7c878e','#00b5e2','#00af66',"#E69F00","#660099")
p2 = ggboxplot(dat_cp_processed, x='init', y='value', fill='init',alpha=0.8,
               palette = my_colors[1:4],labs='g',ylab=0, xlab = 0,
               facet.by = c('metric','Setting'), scales = 'free_y',
               ggtheme = theme(
                 # legend.direction="vertical",
                 legend.title =element_blank() ,
                 legend.position = "bottom",
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
                 plot.margin=grid::unit(c(0.3,0,40,0), "mm")
               )
)
p2
ggsave(paste0("offlineIC_cp",Sys.Date(), ".pdf"), width = 14, height = 5)
