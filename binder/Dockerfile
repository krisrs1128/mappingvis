from krisrs1128/mappingvis:020801
# install the notebook package
RUN pip3 install --no-cache --upgrade pip && \
    pip3 install --no-cache notebook

# create user with a home directory
ARG NB_USER
ARG NB_UID
ENV USER mapper
ENV HOME /home/mapper/

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${USER}
WORKDIR ${HOME}
USER ${USER}

RUN git clone git@github.com:krisrs1128/mappingvis.git
RUN mkdir /data/