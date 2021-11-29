# from utils.data.pascal import mydataset_pascal
from utils.data.my_dataset_dtd import dtd_embedding
from utils.data.my_dataset_tcd import tcd_embedding

datasets = {
    # 'pascal': mydataset_pascal,
    'dtd': dtd_embedding,
    'tcd': tcd_embedding
}