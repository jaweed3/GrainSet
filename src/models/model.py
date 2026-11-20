import os
import logging
import torch

from collections import OrderedDict
from .backbones.resnet import seresnet50
from timm.models import vgg19, inception_v3, vit_base_patch16_224
from timm.models.resnet import resnet50, resnet152

__all__ = ['get_model','load_pretrain_model']

import timm

models = {
    'seresnet50':seresnet50,
    'vgg19':vgg19,
    'inception_v3':inception_v3,
    'vit_base':vit_base_patch16_224,
    'resnet50':resnet50,
    'resnet152':resnet152,
    'mobilenetv4_small': lambda **kw: timm.create_model('mobilenetv4_conv_small', **kw),
    'mobilenetv3_large': lambda **kw: timm.create_model('mobilenetv3_large_100', **kw),
    'efficientnet_b0': lambda **kw: timm.create_model('efficientnet_b0', **kw),
    'edgenext_x_small': lambda **kw: timm.create_model('edgenext_x_small', **kw)
}


def get_model(cfg, **kwargs):
  model_name = cfg.MODEL.NAME.lower()
  if model_name in models:
      model = models[model_name](**kwargs)
  elif timm.is_model(model_name):
      pretrained = getattr(cfg.MODEL, 'PRETRAINED', True)
      model = timm.create_model(model_name, pretrained=pretrained, **kwargs)
  else:
      raise ValueError("model: {} is not supported, please check model name...".format(cfg.MODEL.NAME))
  # model = models[model_name.lower()]()
  resume = cfg.MODEL.RESUME_PATH
  if os.path.exists(resume):
    model = torch.load(resume, weights_only=False).module
    print('**********Load model from:',resume)
    model.inter_layer=False
  return model
  
    

def load_pretrain_model(model, resume):
  assert os.path.isfile(resume), "Error: check the checkpoint path...."
  logging.warning("load pretrain model from {}".format(resume))
  
  state_dict_pretrain = torch.load(resume, weights_only=False)    
  
  is_best = True
  if 'best' not in resume:
    is_best = False
  
  if is_best:
    state_dict_to_load = state_dict_pretrain
  else:
    state_dict_to_load = state_dict_pretrain['state_dict']

  msg = model.load_state_dict(state_dict_to_load, strict=False)
  logging.info(msg)

  '''
  state_dict = model.state_dict()
  keys_wrong_shape = []
  state_dict_suitable = OrderedDict()

  for k, v in state_dict_to_load.items():
    if v.shape == state_dict[k].shape:
      state_dict_suitable[k] = v
    else:
      keys_wrong_shape.append(k)
  logging.info('Shape unmatched weights: {}'.format(keys_wrong_shape))      
  msg = model.load_state_dict(state_dict_suitable, strict=False)
  '''
  
  


