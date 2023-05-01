# Scale individual facet y-axes
scale_inidividual_facet_y_axes = function(plot, ylims) {
  init_scales_orig = plot$facet$init_scales
  
  init_scales_new = function(...) {
    r = init_scales_orig(...)
    # Extract the Y Scale Limits
    y = r$y
    # If this is not the y axis, then return the original values
    if(is.null(y)) return(r)
    # If these are the y axis limits, then we iterate over them, replacing them as specified by our ylims parameter
    for (i in seq(1, length(y))) {
      ylim = ylims[[i]]
      if(!is.null(ylim)) {
        y[[i]]$limits = ylim
      }
    }
    # Now we reattach the modified Y axis limit list to the original return object
    r$y = y
    return(r)
  }
  
  plot$facet$init_scales = init_scales_new
  
  return(plot)
}


############# get the common legend ##############
library(ggpubr)
library(dplyr)
library(tidyr)
if(Sys.info()["nodename"] %in% c("PC-20181212HEBU")){
  curr_dir <- "C:/Users/test/Dropbox/tml/IHS/simu/simu/output/final_perf"
  setwd(curr_dir)
} else{ # greatlakes
  curr_dir <- "/home/mengbing/research/RL_nonstationary/code2/simulation_nonstationary_changept_detection/output"
  setwd(curr_dir)
}
dat_cp_1 =fread('icmodel_2022-11-02method(25)N50_1d.csv')
dat_cp_2 =fread('icmodel_2022-11-02method(23)N50_1d.csv')

dat_cp_1$init <- factor(dat_cp_1$init, levels = c('best_model', 'true_change_points','random_cp_K2'),
                        labels = c('Model Selected via the Information Criterion', 'Oracle Change Point', 'Random Change Point'))
dat_cp_2$init <- factor(dat_cp_2$init, levels = c('best_model', 'true_change_points','no_change_points'),
                        labels = c('Model Selected via the Information Criterion', 'Oracle Change Point', 'No Change Point'))
dat_cp = rbind(dat_cp_1, dat_cp_2)
dat_cp$Setting <- factor(dat_cp$Setting,levels = c('pwconst2', 'smooth'),
                         labels = c('Piecewise Constant', 'Smooth'))
dat_cp$compare = rep(c('setting1', 'setting2'), each = dim(dat_cp_1)[1])
dat_cp_processed <- gather(dat_cp, metric, value, -c('Setting', 'seed', 'init','compare'))
dat_cp_processed$metric=factor(dat_cp_processed$metric, levels = c('cp_err', 'ARI'),
                               labels =  c('CP Error', 'ARI'))
cbPalette=c('#cc0c00', '#5c88da','#84bd00', '#ffcc00', '#7c878e','#00b5e2','#00af66',"#E69F00","#660099")

p2 = ggboxplot(dat_cp_processed, x='init', y='value', fill='init',alpha=0.8,
               palette = my_colors[1:4],labs='g',
               ggtheme = theme(
                 # legend.direction="vertical",
                 legend.title = ,
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
                 axis.text.x=element_text(colour="black", size=0, angle = 0),
                 strip.text.y = element_blank(),
                 axis.text.y=element_text(colour="black", size=16),
                 plot.margin=grid::unit(c(0.3,0,40,0), "mm")
               )
               )
p2
p2=ggpar(p2, legend.title = "")


##################################################
ylims = list(c(0, 0.18),c(0.25,1), c(0, 0.20),NULL)

p_cp1 <- ggboxplot(dat_cp_processed[dat_cp_processed$compare == "setting1",],
                 x = "init", y = "value",alpha=0.8,xlab = FALSE, ylab = "Estimation Performance",
                 fill = "init", palette = my_colors[1:3],title = "Oracle Change Point+Random Change Point",
                 facet.by =c('metric','Setting'), scales = 'free_y',
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
                   text=element_text(size=18),
                   # axis.text.x = element_blank(),
                   axis.text.x=element_text(colour="white", size=0, angle = 0),
                   strip.text.y = element_blank(),
                   axis.text.y=element_text(colour="black", size=16),
                   plot.margin=grid::unit(c(0.3,0,8,0), "mm")
                 )
                 )



scale_inidividual_facet_y_axes(p_cp1, ylims = ylims)
p_cp1


# bxp+scale_fill_manual(values=cbPalette[c(1,2,5)])
# Dot plot
p_cp2 <- ggboxplot(dat_cp_processed[dat_cp_processed$compare == "setting2",],
                x = "init", y = "value", alpha=0.8,ylab=0, xlab = 0,title = "Oracle Change Point+No Change Point",
                fill = "init", palette = my_colors[c(1,2,4)],
                facet.by =c('metric','Setting'),scales = 'free',
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
                  text=element_text(size=18),
                  # axis.text.x = element_blank(),
                  axis.text.x=element_text(colour="white", size=0, angle = 0),
                  # strip.text.y = element_blank(),
                  axis.text.y=element_text(colour="white", size=0),
                  plot.margin=grid::unit(c(0.3,0,8,0), "mm")
                ))
p_cp2
scale_inidividual_facet_y_axes(p_cp2, ylims = ylims)

# Arrange
# ::::::::::::::::::::::::::::::::::::::::::::::::::
ggarrange(p_cp1, p_cp2, ncol = 2, common.legend = 1,
          legend.grob = get_legend(p2), legend = 'bottom')
ggsave(paste0("offlineIC_cp",Sys.Date(), ".pdf"), width = 14, height = 5)
