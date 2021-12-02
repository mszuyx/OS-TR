import torch

from utils.model.model_ostr import OSnet_free, OSnet_frozen
from utils.model.mobilenetv3 import mobilenetv3_large, mobilenetv3_small
from utils.model.model_ostr_mbn3 import OSnet_mb_frozen, OSnet_mb_free

models = {
    'ResNet50_free': OSnet_free,
    'ResNet50_frozen': OSnet_frozen,
    'mobilenetv3_large': mobilenetv3_large,
    'mobilenetv3_small': mobilenetv3_small,
    'OSnet_mb_frozen': OSnet_mb_frozen,
    'OSnet_mb_free': OSnet_mb_free
}
