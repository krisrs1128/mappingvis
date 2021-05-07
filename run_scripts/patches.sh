#!/usr/bin/env bash

git clone https://github.com/krisrs1128/mappingvis.git
((B++))
Rscript -e "rmarkdown::render('mappingvis/mappingvis/2-preprocess.Rmd', params = list(n_patches=10))"
tar -zcvf patches_${B}.tar.gz processed
mv processed_${B}.tar.gz $_CONDOR_SCRATCH_DIR/
