
#' @importFrom EBImage normalize
#' @export
normalize_input <- function(x, ...) {
    n_layers <- dim(x)[3]
    result <- array(dim = dim(x))

    for (l in seq_len(n_layers)) {
        cur_layer <- x[,, l]
        cur_layer <- cur_layer - min(cur_layer)
        cur_layer <- cur_layer / max(cur_layer)
        result[,, l] <- EBImage::normalize(cur_layer, ...)
    }

    result
}

#' @export
impute_na <- function(x, val = 0) {
    x[is.na(x)] <- val
    x
}

#' Fast Subset Read
#'
#' @importFrom gdalUtils gdalbuildvrt
#' @export
read_subset <- function(x_path, te, band_names = NULL) {
    tmp <- tempfile()
    gdalbuildvrt(x_path, tmp, te = te)
    result <- brick(tmp)
    if (is.null(band_names)) {
      names(result) <- c("B1", "B2", "B3", "B4", "B5", "B6_VCID_1", "B6_VCID_2", "B7", "B8", "BQA", "ndvi", "ndsi", "ndwi", "elevation", "slope")
    }

    # remove outliers and return
    rmat <- cbind(-Inf, -1e8, NA)
    reclassify(result, rmat)
}

#' Write patch to file
#'
#' @import raster
#' @importFrom dplyr %>%
#' @importFrom sf st_point st_buffer st_geometry st_bbox
#' @export
generate_patch <- function(x_path, center, max_na = 0.2, subset_inputs=NULL) {
  if (is.null(subset_inputs)) {
    subset_inputs <- c(1:5, 7:9, 11:15)
  }

  point <- st_point(center, dim = "XY") %>%
    st_buffer(0.1) %>%
    st_geometry()

  x_raster <- read_subset(x_path, st_bbox(point))
  x <- as.array(x_raster)
  x <- x[,, subset_inputs]
  if (mean(is.na(x)) < max_na) {
    x <- impute_na(x) %>%
      normalize_input(ft = c(-1, 1))
  } else {
    stop("Too many missing values.")
  }
  list(x = x, meta = point, raster = x_raster)
}

#' Extract Label
#'
#' @importFrom dplyr %>%
#' @importFrom raster extent rasterize
#' @importFrom sf st_bbox st_crop st_zm
#' @importFrom abind abind
#' @importFrom purrr map
#' @export
label_mask <- function(ys, x_raster) {
  box_ <- extent(x_raster)
  box <- c(xmin = box_[1], xmax = box_[2], ymin = box_[3], ymax = box_[4])

  rasterize_ <- function(z, p) {
    if (nrow(z) == 0) {
      return (array(0, c(dim(p)[1:2], 1)))
    }
    rasterize(z, p) %>%
      as.array()
  }

  y_ <- map(ys, ~ st_crop(., box)) %>%
    map(~ rasterize_(., x_raster)) %>%
    abind()

  y_[!is.na(y_)] <- 1
  y_[is.na(y_)] <- 0
  background <- 1 - apply(y_, c(1, 2), max)
  abind(y_, background)
}

#' Write patches to file
#'
#' @importFrom sf write_sf
#' @importFrom stringr str_c
#' @importFrom reticulate import
#' @export
write_patches <- function(x_path, ys, centers, out_dir) {
  unlink(out_dir, force = TRUE)
  dir.create(out_dir, recursive = TRUE)

  j <- 1
  for (i in seq_len(nrow(centers))) {
    err <- function(e) { return(NA) }
    patch <- tryCatch({ generate_patch(x_path, centers[i, ]) }, error = err)
    y <- tryCatch({ label_mask(ys, patch$raster) }, error = err)
    if (is.na(y) || is.na(patch)) next

    # save results
    np$save(file.path(out_dir, str_c("x-", j, ".npy")), patch$x)
    np$save(file.path(out_dir, str_c("y-", j, ".npy")), y)
    write_sf(patch$meta, file.path(out_dir, str_c("geo-", j, ".geojson")))
    j <- j + 1
  }
}


#' Tensor to Raster
#'
#' @importFrom dplyr %>%
#' @importFrom raster brick
#' @export
to_raster <- function(x) {
  x %>%
      as.array() %>%
      aperm(c(2, 3, 1)) %>%
      brick()
}


#' Helper to convert npy to raster bricks
load_npy <- function(p) {
  np$load(p) %>%
    to_raster()
}

plot_rgb <- function(x, channels=NULL, ...) {
  if (is.null(channels)) {
    channels <- seq_len(dim(x)[3])
  }
  
  ggRGB(subset(x, channels), ...) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    theme(
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      axis.title = element_blank(),
      plot.margin = unit(c(0, 0, 0, 0), "cm")
    )
}