
#' Dice Loss for a single dimension
#'
#' @import torch
#' @export
dice_loss_ <- function(y_hat, y, smooth = 1) {
    1 - ((2 * (y_hat * y)$sum() + smooth) / (y_hat$sum() + y$sum() + smooth))
}


#' Dice loss
#' @import torch
#' @export
dice_loss <- function(y_hat, y, smooth = 1, weights = c(0.8, 1.2, 0.1)) {
    K <- dim(y)[2]
    losses <- torch_zeros(K)
    for (k in seq_len(K)) {
        res <- dice_loss_(y_hat[,k,,], y[,k,,], smooth = smooth)
        losses[k] <- res
    }
    (weights * losses)$sum()
}

#' Train over one data loader pass
#' @import torch
#' @export
train_epoch <- function(model, opt, loaders, scheduler, device) {
    model$train()
    losses <- list("train" = c(), "test" = c())
    for (xy in enumerate(loaders$train)) {
        opt$zero_grad()
        y_hat <- model(xy[[1]]$to(device = device))
        y <- xy[[2]]$to(device = device)

        loss <- dice_loss(y_hat, y)
        loss$backward()
        opt$step()
        scheduler$step()

        losses$train <- append(losses$train, loss$item())
    }

    list(model = model, opt = opt, scheduler = scheduler, loss = losses)
}
