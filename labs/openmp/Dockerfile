# labs/openmp/Dockerfile
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    make \
 && rm -rf /var/lib/apt/lists/*

# default working directory inside container
WORKDIR /repo/labs/openmp

# no COPY here: we will mount the repo at runtime via -v
CMD ["bash"]