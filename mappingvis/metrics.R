
precision_recall_curve <- function(y, y_hat, n_thresh=10, low=0.1, high=0.8) {
    p_thresh <- seq(low, high, length.out = n_thresh)
    metrics <- tibble(
        threshold = p_thresh,
        precision = rep(0, n_thresh),
        recall = rep(0, n_thresh)
    )

    for (i in seq_len(n_thresh)) {
        y_ <- as.numeric(y_hat > p_thresh[i])
        intersection <- sum(y * y_)

        metrics$precision[i] <- intersection / sum(y_)
        metrics$recall[i] <- intersection / sum(y)
    }

    metrics
}


prediction_paths <- function(data_dir) {
    paths <- list()
    i <- 1
    for (data_split in c("train", "test")) {
        for (type in c("x", "y", "y_hat")) {
            paths[[i]] <- data.frame(
                split = data_split,
                type = type,
                path = dir(file.path(params$preds_dir, data_split),  str_c("^", type, "-[0-9]+"),  full = TRUE)
            )
            i <- i + 1
        }
    }
    bind_rows(paths) %>%
        mutate(ix = as.integer(str_extract(path, "[0-9]+"))) %>%
        dplyr::select(path, split, type, ix) %>%
        arrange(split, ix, type)
}


metrics_fun <- function(paths) {
    metrics <- list()

    ix <- 1
    for (i in unique(paths$ix)) {
        paths_i <- paths %>%
            filter(ix == i) %>%
            split(.$type)

        y <- np$load(paths_i[["y"]]$path[1])
        y_hat <- np$load(paths_i[["y_hat"]]$path[1])
        for (k in seq_len(nrow(y))) {
            metrics[[ix]] <- precision_recall_curve(y[k,,], y_hat[k,,])
            metrics[[ix]]$class <- k
            metrics[[ix]]$ix <- i
            metrics[[ix]]$path <- paths_i[["y_hat"]]$path[1]
            ix <- ix + 1
        }
    }

    bind_rows(metrics)
}
TF_plot<-function(paths,i=1,probability=0.4,channel=1){
    
    y_df<-paths %>% filter(ix==i,type=='y',split=='test')
    y_hat_df<-paths %>% filter(ix==i,type=='y_hat',split=='test')
    y <- np$load(y_df$path)
    y_hat <- np$load(y_hat_df$path)
    
    tp<-y[channel,,]==1&y_hat[channel,,]>=probability
    tn<-y[channel,,]==1&y_hat[channel,,]<probability
    fn<-y[channel,,]==0&y_hat[channel,,]>=probability
    fp<-y[channel,,]==0&y_hat[channel,,]<probability
    result<-1*tp+2*tn+3*fn+4*fp
    par(mar=c(4,4,4,6))
    par(xpd=TRUE)
    image(result,col=c(3,1,2,0))
    legend("right", inset=c(-0.2,0), legend=c("TP","TN","FN","FP"), pch=15, col=c(3,1,2,0),bty="n")
}  
