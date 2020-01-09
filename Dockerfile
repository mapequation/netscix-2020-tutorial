FROM jupyter/base-notebook:1386e2046833

USER root

RUN apt-get update && \
    apt-get install -yq --no-install-recommends build-essential

COPY ./requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER
