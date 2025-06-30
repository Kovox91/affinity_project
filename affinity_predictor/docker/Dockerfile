FROM mambaorg/micromamba:1.5.6

ENV MAMBA_DOCKERFILE_ACTIVATE=1 \
    CONDA_DEFAULT_ENV=base \
    PYTHONNOUSERSITE=1

WORKDIR /opt/app
COPY --chown=mambauser:mambauser . /opt/app

RUN micromamba update -n base -c conda-forge micromamba

RUN micromamba clean --all --yes \
 && micromamba install -y -n base -c conda-forge pymol-open-source \
 && micromamba clean --all --yes

# <<< FIX HERE: do this as root >>>
RUN chmod -R u+rwX /opt/app

USER mambauser

RUN pip install --upgrade pip setuptools \
 && pip wheel /opt/app -w /tmp/wheels \
 && pip install /tmp/wheels/affinity_predictor-*.whl

WORKDIR /opt/app
ENTRYPOINT ["kedro", "run"]
