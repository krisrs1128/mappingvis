#!/usr/bin/env bash

git clone https://github.com/krisrs1128/mappingvis.git
Rscript -e "rmarkdown::render('mappingvis/mappingvis/2-preprocess.Rmd')"
tar -zcvf processed_${B}.tar.gz mappingvis/mappingvis/processed
mv processed_${B}.tar.gz $_CONDOR_SCRATCH_DIR/
