# from utils.data.pascal import mydataset_pascal
from utils.data.my_dataset_dtd import dtd_embedding
from utils.data.my_dataset_tcd import tcd_embedding
from utils.data.my_dataset_tcd_alot import tcd_alot_embedding
from utils.data.my_dataset_tcd_alot_dtd import tcd_alot_dtd_embedding

datasets = {
    # 'pascal': mydataset_pascal,
    'dtd': dtd_embedding,
    'tcd': tcd_embedding,
    'tcd_alot': tcd_alot_embedding,
    'tcd_alot_dtd': tcd_alot_dtd_embedding
}