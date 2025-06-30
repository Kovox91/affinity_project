FROM mambaorg/micromamba:1.5.6

ENV MAMBA_DOCKERFILE_ACTIVATE=1 \
    CONDA_DEFAULT_ENV=base \
    PYTHONNOUSERSITE=1

WORKDIR /build
COPY . /opt/app

RUN micromamba update -n base -c conda-forge micromamba

RUN micromamba clean --all --yes \
 && micromamba install -y -n base -c conda-forge pymol-open-source \
 && micromamba clean --all --yes

RUN chown -R mambauser:mambauser /opt/app

USER mambauser
RUN pip install --upgrade pip setuptools \
 && cd /tmp && pip install /opt/app

WORKDIR /opt/app
ENTRYPOINT ["kedro", "run"]
