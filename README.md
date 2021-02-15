# Visualizing Mapping Models

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/krisrs1128/mappingvis/HEAD)

This repository includes code for visualizing the machine learning for earth
observation workflow. It was presented during the [UW Madison Data Science
Research Bazaar](https://datascience.wisc.edu/data-science-research-bazaar/) on
data science for health and the environment. Compiled versions of all the
notebooks are linked [here](https://krisrs1128.github.io/mapping-vis/). In the
order that they are run, the notebooks are,

* [Data Preview](https://krisrs1128.github.io/mapping-vis/1-preview.html)
* [Preprocess](https://krisrs1128.github.io/mapping-vis/2-preprocess.html)
* [Train](https://krisrs1128.github.io/mapping-vis/3-train.html)
* [Save](https://krisrs1128.github.io/mapping-vis/4-save.html)
* [Evaluate](https://krisrs1128.github.io/mapping-vis/5-eval.html)

If you would like to run just one part of the pipeline, intermediate results are
saved in this [box
folder](https://uwmadison.box.com/s/lwf3rm16qroy08u1wal1kcmh9813hshb). To run
this code in an interactive environment with all packages already installed,
there is a Binder environment available
([generic](https://mybinder.org/v2/gh/krisrs1128/mappingvis/master),
[Rstudio](https://mybinder.org/v2/gh/krisrs1128/mappingvis/master?urlpath=rstudio),
[JupyterLab](https://mybinder.org/v2/gh/krisrs1128/mappingvis/master?urlpath=lab/tree/)).
If you have Docker installed and would like to run this code locally, you can
pull the `krisrs1128/mappingvis` image from DockerHub and launch an Rstudio or
JupyterLab environment in your browser. This can be accomplished by running,

```
shell > docker pull krisrs1128/mappingvis:021401
shell > mkdir ~/test-data # or whatever name you prefer
shell > docker run -p 8889:8889 -p 8787:8787 -u 0 -v ~/test-data:/home/jovyan/data/ -it krisrs1128/mappingvis:021401 bash
docker shell > jupyter lab --port 8889 --ip=0.0.0.0 --allow-root
```

from your home and docker shells, respectively. For those who are curious, `-p
xxxx:xxxx` makes sure the internal docker network port is visible externally,
`-u 0` lets you run docker as root (this is needed for JupyterLab), and -v
ensures that the `test-data` directory on your local machine is mapped to
`/home/jovyan/data` on the docker image.

Going to `localhost:8889` in your browser and entering the token printed about
by the output from the jupyter lab command should open a JupyterLab environment
with all packages available. It's also possible to use Rstudio in a similar,
though somewhat more tedious, way. The difficulty is that Rstudio doesn't like
to be run as root, so we first have to create a user with a dummy password and
home directory,


```
docker shell> useradd test
docker shell> passwd test # enter a password
docker shell> mkdir -p /home/test/.rstudio/graphics-r3
docker shell> sudo chown -R test /home/test/.rstudio/
```

Then, we can copy over the contents of this repo to the new `kris` user and
start Rstudio,

```
docker shell > cp -r ~/mappingvis /home/test/
docker shell > rstudio-server start
```

Navigating to `localhost:8787` on your home machine will now show an Rstudio
environment. Enter the username and password you created in the previous step,
and you can start running the code.
