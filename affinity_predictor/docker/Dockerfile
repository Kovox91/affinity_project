FROM mambaorg/micromamba:1.5.6

ENV MAMBA_DOCKERFILE_ACTIVATE=1 \
    CONDA_DEFAULT_ENV=base \
    PYTHONNOUSERSITE=1

# Clean separate working dir for pip install
COPY . /opt/app
RUN chown -R mambauser:mambauser /opt/app

# Switch to non-root user before installing anything
USER mambauser
WORKDIR /opt/app

# Install dependencies from conda
RUN micromamba install -y -n base -c conda-forge pymol-open-source \
 && micromamba clean --all --yes

# Install your project
RUN pip install --upgrade pip setuptools \
 && pip install --no-build-isolation .

# Set default command for the container
ENTRYPOINT ["kedro", "run"]
