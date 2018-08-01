FROM epwalsh/machine-learning-base:latest

# Set working dir.
WORKDIR /opt/python/app

# Install Python dependencies.
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Move application files into place.
COPY . .

ENTRYPOINT ["make", "test"]
