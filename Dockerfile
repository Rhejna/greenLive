# syntax=docker/dockerfile:1.4
FROM --platform=$BUILDPLATFORM python:3.10-alpine AS builder

WORKDIR /app

COPY requirements.txt /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

COPY . /app

ENTRYPOINT ["python3"]
CMD ["py/app.py"]

FROM builder as dev-envs

RUN apk update && \
    apk add git

RUN addgroup -S docker && \
    adduser -S --shell /bin/bash --ingroup docker vscode

# Assuming the necessary Docker tools are present in the directory /docker-tools,
# you can copy them into the image as follows. Adjust the source path as needed.
COPY --from=/docker-tools / /usr/local/bin/

# If Docker tools are to be copied from another image, use the following line instead:
# COPY --from=gloursdocker/docker / /usr/local/bin/
