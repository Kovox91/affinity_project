FROM mambaorg/micromamba:1.5.6

ENV MAMBA_DOCKERFILE_ACTIVATE=1 \
    CONDA_DEFAULT_ENV=base \
    PYTHONNOUSERSITE=1

# Create a clean writable working dir
WORKDIR /opt/app

# Copy only project files — do not try to chown them
COPY . /opt/app

# Install pymol and Python dependencies
RUN micromamba install -y -n base -c conda-forge pymol-open-source \
 && micromamba clean --all --yes

# Set the egg-info directory to somewhere writable
ENV SETUPTOOLS_USE_DISTUTILS=stdlib

# Use --no-build-isolation to skip rebuilding env just for setuptools
RUN pip install --upgrade pip \
 && pip install --no-build-isolation .

ENTRYPOINT ["kedro", "run"]
