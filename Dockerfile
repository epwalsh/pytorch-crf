FROM python:3.5

# Install some base packages.
RUN apt-get update && \
    apt-get install \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set working dir.
WORKDIR /opt/python/app

# Install Python dependencies.
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

# Move application files into place.
COPY yapycrf ./yapycrf/
