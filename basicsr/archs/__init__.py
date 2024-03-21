import importlib
from copy import deepcopy
from os import path as osp
from torch.nn import init
from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import ARCH_REGISTRY
import logging
import functools


logger = logging.getLogger('base')

__all__ = ['build_network']

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
# import all the arch modules
_arch_modules = [importlib.import_module(f'basicsr.archs.{file_name}') for file_name in arch_filenames]


def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net

def build_ddpmnetwork(opt):
    opt = deepcopy(opt)
    opt_unet = opt['unet']
    opt_netg = opt['netg']
    network_type = opt_netg.pop('type')
    unet_type = opt_unet.pop('type')
    unet = ARCH_REGISTRY.get(unet_type)(**opt_unet)
    net = ARCH_REGISTRY.get(network_type)(unet, **opt_netg)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    if opt['phase'] == 'train':
        # init_weights(netG, init_type='kaiming', scale=0.1)
        init_weights(net, init_type='orthogonal')
    return net


# def build_refineddpmnetwork(opt):
#     opt = deepcopy(opt)
#     opt_unet = opt['unet']
#     opt_netg = opt['netg']
#     opt_opn = opt['opn']
#     network_type = opt_netg.pop('type')
#     unet_type = opt_unet.pop('type')
#     opn_type = opt_opn.pop('type')
#     unet = ARCH_REGISTRY.get(unet_type)(**opt_unet)
#     opn= ARCH_REGISTRY.get(opn_type)(**opt_opn)
#     net = ARCH_REGISTRY.get(network_type)(unet, opn, **opt_netg)
#     logger = get_root_logger()
#     logger.info(f'Network [{net.__class__.__name__}] is created.')
#     if opt['phase'] == 'train':
#         # init_weights(netG, init_type='kaiming', scale=0.1)
#         init_weights(net, init_type='orthogonal')
#     return net

def dynamic_instantiation(modules, cls_type, opt):
    """Dynamically instantiate class.

    Args:
        modules (list[importlib modules]): List of modules from importlib
            files.
        cls_type (str): Class type.
        opt (dict): Class initialization kwargs.

    Returns:
        class: Instantiated class.
    """

    for module in modules:
        cls_ = getattr(module, cls_type, None)
        if cls_ is not None:
            break
    if cls_ is None:
        raise ValueError(f'{cls_type} is not found.')
    return cls_(**opt)


def define_network(opt):
    network_type = opt.pop('type')
    net = dynamic_instantiation(_arch_modules, network_type, opt)
    return net



def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))