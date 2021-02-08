from krisrs1128/mappingvis:020801
# install the notebook package
RUN pip3 install --no-cache --upgrade pip && \
    pip3 install --no-cache notebook

# create user with a home directory
ENV USER mapping
ENV HOME /home/mapping

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${USER}
WORKDIR ${HOME}
USER ${USER}
