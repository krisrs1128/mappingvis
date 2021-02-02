
#' Initialize a U-Net
#'
#' @import torch
#' @export
initialize_unet <- function(device = "cpu", ...) {

unet <- nn_module(
  "unet",

  initialize = function(channels_in = 13,
                        n_classes = 2,
                        depth = 5,
                        n_filters = 6) {

    self$down_path <- nn_module_list()

    prev_channels <- channels_in
    for (i in 1:depth) {
      self$down_path$append(down_block(prev_channels, 2 ^ (n_filters + i - 1)))
      prev_channels <- 2 ^ (n_filters + i -1)
    }

    self$up_path <- nn_module_list()

    for (i in ((depth - 1):1)) {
      self$up_path$append(up_block(prev_channels, 2 ^ (n_filters + i - 1)))
      prev_channels <- 2 ^ (n_filters + i - 1)
    }

    self$last = nn_conv2d(prev_channels, n_classes, kernel_size = 1)
  },

  forward = function(x) {

    blocks <- list()

    for (i in 1:length(self$down_path)) {
      x <- self$down_path[[i]](x)
      if (i != length(self$down_path)) {
        blocks <- c(blocks, x)
        x <- nnf_max_pool2d(x, 2)
      }
    }

    for (i in 1:length(self$up_path)) {
      x <- self$up_path[[i]](x, blocks[[length(blocks) - i + 1]]$to(device = device))
    }

    # softmax along 2nd (channel) dimension
    sigma <- nn_softmax(2)
    sigma(self$last(x))
  }
)

down_block <- nn_module(
  "down_block",

  initialize = function(in_size, out_size) {
    self$conv_block <- conv_block(in_size, out_size)
  },

  forward = function(x) {
    self$conv_block(x)
  }
)

up_block <- nn_module(
  "up_block",

  initialize = function(in_size, out_size) {

    self$up = nn_conv_transpose2d(in_size,
                                  out_size,
                                  kernel_size = 2,
                                  stride = 2)
    self$conv_block = conv_block(in_size, out_size)
  },

  forward = function(x, bridge) {

    up <- self$up(x)
    torch_cat(list(up, bridge), 2) %>%
      self$conv_block()
  }
)

conv_block <- nn_module(
  "conv_block",

  initialize = function(in_size, out_size) {

    self$conv_block <- nn_sequential(
      nn_conv2d(in_size, out_size, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_dropout(0.6),
      nn_conv2d(out_size, out_size, kernel_size = 3, padding = 1),
      nn_relu()
    )
  },

  forward = function(x){
    self$conv_block(x)
  }
)

unet(...)$to(device = device)
}
