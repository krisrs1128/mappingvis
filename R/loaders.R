
#' UNet Data Initializer
#'
#' @export
initializer <- function(x_paths, y_paths) {
    self$paths <- data.frame("x" = x_paths, "y" = y_paths)
}

#' Data Augmentation
#'
#' @importFrom purrr map
#' @importFrom torchvision transform_random_vertical_flip transform_random_horizontal_flip transform_crop
#' @export
augment <- function(x, y, imsize=512) {
    p_v <- sample(c(0, 1), 1)
    p_h <- sample(c(0, 1), 1)
    cix <- map(dim(x)[-1], ~ sample(1:(. - imsize), 1)) %>%
        as.numeric()

    list(x, y) %>%
        map(~ transform_random_vertical_flip(., p = p_v)) %>%
        map(~ transform_random_horizontal_flip(., p = p_h)) %>%
        map(~ transform_crop(., top = cix[1], left = cix[2], height = imsize, width = imsize))
}

#' Get Data Item
#'
#' @importFrom torchvision transform_to_tensor
#' @export
getitem <- function(i) {
    x <- get(load(self$paths$x[i])) %>%
        transform_to_tensor()
    y <- get(load(self$paths$y[i])) %>%
        transform_to_tensor()
    augment(x, y)
}
