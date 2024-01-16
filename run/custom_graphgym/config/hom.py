from torch_geometric.graphgym.register import register_config
from rooted_hom_count.count_hom import patterns as original_patterns


@register_config('hom')
def set_cfg_example(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    cfg.dataset.add_counts = False

    cfg.dataset.count_type = 'HOM'

    cfg.dataset.patterns = ','.join(original_patterns.keys())