from mmengine.registry import Registry

DATASETS = Registry('datasets')

def build_dataset(cfg):
    """Build dataset."""
    return DATASETS.build(cfg)


