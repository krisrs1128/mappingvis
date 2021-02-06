#!/bin/bash

# unzip data
tar -zxvf glacier_data.tar.gz
git clone https://github.com/krisrs1128/mappingvis.git
git clone https://github.com/krisrs1128/mapping-vis.git
Rscript -e "rmarkdown::render('mapping-vis/training.Rmd')"
