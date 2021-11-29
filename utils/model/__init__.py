import torch

from utils.model.model_ostr import OSnet_free, OSnet_frozen


models = {
    'ResNet50_free': OSnet_free,
    'ResNet50_frozen': OSnet_frozen
}
