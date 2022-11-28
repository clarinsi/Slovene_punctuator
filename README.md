# P&C API

Implements the P&C API for NeMo Punctuation & Capitalisation models that predict punctuation as well as capitalization. For more details about building such models, see the official [NVIDIA NeMo documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/punctuation_and_capitalization.html), [NVIDIA NeMo GitHub](https://github.com/NVIDIA/NeMo).

The API provides two endpoints `/api/healthcheck`, to retrieve the service status, and `/api/punctuate` to request a punctuation and capitalization. The service accepts either a single string or list of strings. The result will be in the same format as the request, either as a single string or list of strings. The maximal accepted text length is 5000c. Note that punctuation and capitalization of one 5000c text block on cpu will take advantage of all available cores and may take ~30s (on a system with 24 vCPU).

# Prerequisites

- docker >= 20.10.17
- docker compose >= 2.6.0
- NeMo model and `model.info`

# Model.info

The expected format for `model.info` is:
```yml
language_code: # dash saparated two-letter ISO 639-1 Langauge Code, lowercase, and ISO 3166 Country Code, uppercase, eg. sl-SI
domain: # model domain
version: # model version
info:
  build: # build time in YYYYMMDD-HHSS format
  framework: nemo:nlp:tc:pc
  ... # aditional info, optional
features: # optional
  ... # information about special features
```

The NeMo model file is expected in the same folder, named as `nlp_tc_pc.nemo`.

The Punctuation & Capitalisation model developed as part of work package 2 of the Development of Slovene in a Digital Environment, RSDO, project (https://slovenscina.eu/govorne-tehnologije), can be downloaded from http://hdl.handle.net/11356/1735.

# Deployment

Run `docker compose up -d` to deploy on cpu or `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d` to run on gpu.

# Approximate memory consumption for cpu deployment

- 3GB RAM for service and model

