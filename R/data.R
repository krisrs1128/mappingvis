
#' @importFrom EBImage equalize
#' @export
equalize_input <- function(x, ...) {
    n_layers <- dim(x)[3]
    result <- array(dim = dim(x))

    for (l in seq_len(n_layers)) {
        cur_layer <- x[,, l]
        cur_layer <- cur_layer - min(cur_layer)
        cur_layer <- cur_layer / max(cur_layer)
        result[,, l] <- EBImage::equalize(cur_layer, ...)
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
    if (!is.null(band_names)) {
        names(result) <- band_names
    }

    result
}

#' Write patch to file
#'
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
  x <- x_raster %>%
    as.array()
  x <- x[,, subset_inputs]
  if (mean(is.na(x)) < max_na) {
    x <- impute_na(x) %>%
      equalize_input(range = c(-1, 1))
  } else {
    return()
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
  box <- extent(x_raster) %>%
    st_bbox()

  rasterize_ <- function(z, p) {
    if (nrow(z) == 0) {
      return (array(0, c(dim(p)[1:2], 1)))
    }
    rasterize(z, p) %>%
      as.array()
  }

  y_ <- map(ys, ~ st_crop(., box) %>% st_zm()) %>%
    map(~ rasterize_(., x_raster)) %>%
    abind()

  y_[!is.na(y_)] <- 1
  y_[is.na(y_)] <- 0
  y_
}

#' Write patches to file
#'
#' @importFrom sf write_sf
#' @importFrom stringr str_c
#' @export
write_patches <- function(x_path, ys, centers, out_dir) {
  unlink(out_dir)
  dir.create(out_dir)

  j <- 1
  geo <- list()
  for (i in seq_len(nrow(centers))) {
    patch <- generate_patch(x_path, centers[i, ])

    # if not too many NAs, get mask and crop
    if (!is.null(patch)) {
      y <- label_mask(ys, patch$raster)
      x <- patch$x

      # save results
      save(x, file = file.path(out_dir, str_c("x-", j, ".RData")))
      save(y, file = file.path(out_dir, str_c("y-", j, ".RData")))
      write_sf(patch$meta, file.path(out_dir, str_c("geo-", j, ".geojson")))
      j <- j + 1
    }
  }
}


#' Convert Array to EBImage Image
#'
#' @importFrom dplyr %>%
#' @importFrom EBImage Image
#' @export
to_image <- function(x) {
    as.array(x) %>%
        aperm(c(2, 3, 1)) %>%
        Image()
}

#' Convert Array to RGB Image
#' @importFrom EBImage rgbImage
#' @export
to_rgb <- function(x, ch = c(1, 2, 3)) {
    x <- as.array(x)
    rgbImage(x[ch[1],, ], x[ch[2],, ], x[ch[3],, ])
}
