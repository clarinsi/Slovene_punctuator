# silence all tqdm progress bars
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

from version import __version__
import arrow
from typing import Dict, List, Any, Union, Optional
from pydantic import BaseModel

from time import time
from glob import glob
import yaml
import os

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import torch
from nemo.core.classes.modelPT import ModelPT
from nemo.utils import logging
import contextlib


if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
    logging.info("AMP enabled!\n")
    autocast = torch.cuda.amp.autocast
else:
    @contextlib.contextmanager
    def autocast():
        yield


_TEXT_LEN_LIMIT = 5000
_use_gpu_if_available = True
_model_tag = "unknown"


class PCModel(BaseModel):
  class Config:
    arbitrary_types_allowed = True
  tag: str
  nemo: ModelPT
  platform: str
  active: int

start_time: str = None
models: Dict[str, PCModel] = {}
num_requests_processed: int = None



app = FastAPI(
  title='P&C API',
  version=__version__,
  contact={
      "name": "Vitasis Inc.",
      "url": "https://vitasis.si/",
      "email": "info@vitasis.si",
  }
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PunctuateCapitalizeRequest(BaseModel):
  text: Union[str,List[str]]

punctuateCapitalizeRequestExamples = {
  "text: string": {
    "value": {
      "text": "danes prijetno sneži jutri bo pa še lepše"
    }
  },
  "text: [string]": {
    "value": {
      "text": [
        "danes bo oblačno deževno in hladno",
        "meja sneženja bo na okoli 1600 metrih nadmorske višine",
        "padavine bodo sredi dneva od severa slabele in do večera povsod ponehale najpozneje na jugovzhodu je zapisano na spletnih straneh agencije rs za okolje (arso)"
      ]
    }
  },
}

class PunctuateCapitalizeResponse(BaseModel):
  result: Union[str,List[str]]


punctuateCapitalizeResponseExamples = {
  "result: string": {
    "value": {
      "result": "Danes prijetno sneži, jutri bo pa še lepše."
    }
  },
  "result: [string]": {
    "value": {
      "result": [
        "Danes bo oblačno, deževno in hladno.",
        "Meja sneženja bo na okoli 1600 metrih nadmorske višine.",
        "Padavine bodo sredi dneva od severa slabele in do večera povsod ponehale, najpozneje na jugovzhodu, je zapisano na spletnih straneh Agencije RS za okolje (ARSO)"
      ]
    }
  },
}


class Model(BaseModel):
  tag: str
  workers: Dict[str,Any]
  features: Optional[Dict[str,Any]]
  info: Optional[Dict[str,Any]]

class HealthCheckResponse(BaseModel):
  status: int
  start_time: Optional[str]
  models: Optional[List[Model]]
  num_requests_processed: Optional[int]

healthCheckResponseExamples = {
  "serving": {
    "value": {
      "status": 0,
      "start_time": arrow.utcnow().isoformat(),
      "models": [
        { "tag": "sl-SI:GEN:nemo-4.1", "workers": { "platform": "cpu", "active": 2 } },
      ]
    }
  },
  "failed state": {
    "value": {
      "status": 2,
    }
  },
}


@app.get(
  "/api/healthCheck",
  description="Retrieve service health info.",
  response_model=HealthCheckResponse,
  responses={ 200: { "description": "Success", "content": { "application/json": { "examples": healthCheckResponseExamples } } } }
)
def health_check():
  _SERVICE_UNAVAILABLE_ = -1
  _PASS_ = 0
  _WARN_ = 1
  _FAIL_ = 2

  status: HealthCheckResponse = {'status': _SERVICE_UNAVAILABLE_}
  if not models:
    status = {'status': _FAIL_}
  else:
    status = {'status': _PASS_}
    min_workers_available = 1 # min([ workers_info['available'] for workers_info in _response['workers_info'] ]) if len(_response['workers_info']) > 0 else 0
    if min_workers_available <= -1: # config['workers']['fail']
      status = {'status': _FAIL_}
    elif min_workers_available <= 0: # config['workers']['warn']:
      status = {'status': _WARN_}
    status['models'] = [ { "tag": models[model_tag].tag, "workers": { "platform": models[model_tag].platform, "active": models[model_tag].active } } for model_tag in models ]
    status['start_time'] = start_time
    status['num_requests_processed'] = num_requests_processed

  return status

@app.post("/api/punctuate",
  response_model=PunctuateCapitalizeResponse,
  description=f"Punctuate and capitalize text. Maximum text lenght is {_TEXT_LEN_LIMIT}c.\n\nInput: Lowercased text without punctuation.\n\nOutput: Punctuated and capitalized text.",
  responses={ 200: { "description": "Success", "content": { "application/json": { "examples": punctuateCapitalizeResponseExamples } } } }
)
def punctuate_capitalize(item: PunctuateCapitalizeRequest = Body(..., examples=punctuateCapitalizeRequestExamples)):
  time0 = time()
  if _model_tag not in models:
    raise HTTPException(status_code=404, detail=f"Model {_model_tag} unsupported")

  logging.info(f" Q: {item.text}")

  # force text to lowercase to be on the safe side
  text = [ item.text.lower() ] if isinstance(item.text, str) else [ _text.lower() for _text in item.text ]
  text_len = sum(len(_text) for _text in text)
  if text_len > _TEXT_LEN_LIMIT:
    logging.warning(f'{text}, text length exceded {text_len}c [max {_TEXT_LEN_LIMIT}c]')
    raise HTTPException(status_code=400, detail=f"Bad request.")

  logging.debug(f" Q_lower: {text}")

  models[_model_tag].active += 1
  text_with_punctuation = models[_model_tag].nemo.add_punctuation_capitalization(text)
  models[_model_tag].active -= 1

  result: PunctuateCapitalizeResponse = { "result": ' '.join(text_with_punctuation) if isinstance(item.text, str) else text_with_punctuation }

  logging.info(f' R: {result}\n')
  logging.debug(f'text_length: {text_len}c, duration: {round(time()-time0,2)}s')
  global num_requests_processed
  num_requests_processed += 1

  if num_requests_processed == 0:
    if _use_gpu_if_available and torch.cuda.is_available():
      # Force onto CPU
      models[item.src_language.lower()][item.tgt_language.lower()].nemo = models[item.src_language.lower()][item.tgt_language.lower()].nemo.cpu()
      torch.cuda.empty_cache()

  return result


def initialize():
  time0 = time()

  models: Dict[str, PCModel] = {}
  for _model_info_path in glob(f"./models/**/model.info",recursive=True):
    with open(_model_info_path) as f:
      _model_info = yaml.safe_load(f)

    global _model_tag
    _model_tag = f"{_model_info['language_code']}:{_model_info['domain']}:{_model_info['version']}"
    _model_platform = "gpu" if _use_gpu_if_available and torch.cuda.is_available() else "cpu"
    _model_path = f"{os.path.dirname(_model_info_path)}/{_model_info['info']['framework'].partition(':')[-1].replace(':','_')}.{_model_info['info']['framework'].partition(':')[0]}"

    model = ModelPT.restore_from(_model_path,map_location="cuda" if _model_platform == "gpu" else "cpu")
    model.freeze()
    model.eval()

    models[_model_tag] = PCModel(
      tag = _model_tag,
      nemo = model,
      platform = _model_platform,
      active = 0,
    )

  logging.info(f'Loaded models {[ (models[model_tag].tag,models[model_tag].platform) for model_tag in models ]}')
  logging.info(f'Initialization finished in {round(time()-time0,2)}s')

  start_time = arrow.utcnow().isoformat()
  num_requests_processed = 0
  return start_time, models, num_requests_processed

def start_service():
  uvicorn.run(app, host="0.0.0.0", port=5000)

if __name__ == "__main__":
  logging.setLevel(logging.DEBUG)
  start_time, models, num_requests_processed = initialize()
  start_service()
