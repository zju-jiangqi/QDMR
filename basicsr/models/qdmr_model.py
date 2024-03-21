from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torchvision.utils as tvu
from basicsr.metrics import calculate_metric
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
# from basicsr.models.receptive_cal import *
import copy


@MODEL_REGISTRY.register()
class UDAVQGAN(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

         # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)


        self.LQ_stage = self.opt['network_g'].get('LQ_stage', False) 
        if self.LQ_stage:
            load_path = self.opt['path'].get('pretrain_network_dmc', None)
            assert load_path is not None, 'Need to specify hq prior model path in LQ stage'

            hq_opt = self.opt['network_g'].copy()
            hq_opt['LQ_stage'] = False
            self.net_hq = build_network(hq_opt)
            self.net_hq = self.model_to_device(self.net_hq)
            self.load_network(self.net_hq, load_path, self.opt['path']['strict_load'])

            self.load_network(self.net_g, load_path, False)
            frozen_module_keywords = self.opt['network_g'].get('frozen_module_keywords', None) 
            if frozen_module_keywords is not None:
                for name, module in self.net_g.named_modules():
                    for fkw in frozen_module_keywords:
                        if fkw in name:
                            for p in module.parameters():
                                p.requires_grad = False
                            break

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        logger = get_root_logger()
        if load_path is not None:
            logger.info(f'Loading net_g from {load_path}')
            self.load_network(self.net_g, load_path, self.opt['path']['strict_load'])
            
        if self.is_train:
            self.init_training_settings()
            self.use_dis = (self.opt['train']['gan_opt']['loss_weight'] != 0) 
            self.net_d_best = copy.deepcopy(self.net_d)
        
        self.net_g_best = copy.deepcopy(self.net_g)

    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']
        self.net_g.train()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.net_dt = build_network(self.opt['network_d'])
        self.net_dt = self.model_to_device(self.net_dt)
        self.net_df = build_network(self.opt['network_df'])
        self.net_df = self.model_to_device(self.net_df)
        # load pretrained d models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        # print(load_path)
        if load_path is not None:
            logger.info(f'Loading net_d from {load_path}')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))
            
        self.net_d.train()
        self.net_dt.train()
        self.net_df.train()



    
        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
            self.model_to_device(self.cri_perceptual)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        if train_opt.get('gan_opt_f'):
            self.cri_gan_f = build_loss(train_opt['gan_opt_f']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
    
    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            optim_params.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_g = optim_class(optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        optim_params_d = []
        for k, v in self.net_d.named_parameters():
            optim_params_d.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        for k, v in self.net_dt.named_parameters():
            optim_params_d.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        for k, v in self.net_df.named_parameters():
            optim_params_d.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_d = optim_class(optim_params_d, **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)


    def feed_data(self, data):
        self.lqsrc = data['lq'].to(self.device)
        if 'trg' in data:
            self.lqtrg = data['trg'].to(self.device)
            self.gttrg = data['trg'].to(self.device)
        if 'gt' in data:
            self.gtsrc = data['gt'].to(self.device)


    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']

        for p in self.net_d.parameters():
            p.requires_grad = False
        for p in self.net_dt.parameters():
            p.requires_grad = False
        for p in self.net_df.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()

    #src
        if self.LQ_stage:
            with torch.no_grad():
                self.gt_rec, _, _, gt_indices = self.net_hq(self.gtsrc)

            self.output, l_codebook, l_semantic, _ = self.net_g(self.lqsrc, gt_indices) 
        else:
            self.output, l_codebook, l_semantic, _, src_quant_feat = self.net_g(self.lqsrc) 

        l_g_total = 0
        loss_dict = OrderedDict()

        # ===================================================
        # codebook loss
        if train_opt.get('codebook_opt', None):
            l_codebook *= train_opt['codebook_opt']['loss_weight'] 
            l_g_total += l_codebook.mean()
            loss_dict['l_codebook'] = l_codebook.mean()

        # semantic cluster loss, only for LQ stage!
        if train_opt.get('semantic_opt', None) and isinstance(l_semantic, torch.Tensor):
            l_semantic *= train_opt['semantic_opt']['loss_weight'] 
            l_semantic = l_semantic.mean()
            l_g_total += l_semantic
            loss_dict['l_semantic'] = l_semantic

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gtsrc)
            l_g_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gtsrc)
            if l_percep is not None:
                l_g_total += l_percep.mean()
                loss_dict['l_percep'] = l_percep.mean()
            if l_style is not None:
                l_g_total += l_style
                loss_dict['l_style'] = l_style
        
        # gan loss
        if self.use_dis and current_iter > train_opt['net_d_init_iters']:
            fake_g_pred = self.net_d(self.output)

            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan
        l_g_total = 1*l_g_total
        l_g_total.mean().backward()


    #trg
        if self.LQ_stage:
            with torch.no_grad():
                self.gt_rec, _, _, gt_indices = self.net_hq(self.gttrg)

            self.toutput, l_codebookt, l_semantict, _ = self.net_g(self.lqtrg, gt_indices) 
        else:
            self.toutput, l_codebookt, l_semantict, _, trg_quant_feat = self.net_g(self.lqtrg) 

        l_g_totalt = 0
        # loss_dict = OrderedDict()

        # ===================================================
        # codebook loss
        if train_opt.get('codebook_opt', None):
            l_codebookt *= train_opt['codebook_opt']['loss_weight'] 
            l_g_totalt += l_codebookt.mean()
            loss_dict['l_codebookt'] = l_codebookt.mean()

        # semantic cluster loss, only for LQ stage!
        if train_opt.get('semantic_opt', None) and isinstance(l_semantic, torch.Tensor):
            l_semantict *= train_opt['semantic_opt']['loss_weight'] 
            l_semantict = l_semantict.mean()
            l_g_totalt += l_semantict
            loss_dict['l_semantict'] = l_semantict

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.toutput, self.gttrg)
            l_g_totalt += l_pix
            loss_dict['l_pixt'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.toutput, self.gttrg)
            if l_percep is not None:
                l_g_totalt += l_percep.mean()
                loss_dict['l_percept'] = l_percep.mean()
            if l_style is not None:
                l_g_totalt += l_style
                loss_dict['l_stylet'] = l_style
                
        l_g_totalt.mean().backward()

        self.optimizer_g.step()

        # optimize net_d
        self.fixed_disc = self.opt['train'].get('fixed_disc', False)
        if not self.fixed_disc and self.use_dis and current_iter > train_opt['net_d_init_iters']:
            for p in self.net_d.parameters():
                p.requires_grad = True
            for p in self.net_dt.parameters():
                p.requires_grad = True
            for p in self.net_df.parameters():
                p.requires_grad = True
            self.optimizer_d.zero_grad()
            # real
            real_d_pred = self.net_d(self.gttrg)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()

            fake_d_pred = self.net_d(self.output.detach())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()

            self.optimizer_d.step()
        
        self.log_dict = self.reduce_loss_dict(loss_dict)
        
    def test(self):
        self.net_g.eval()
        net_g = self.get_bare_model(self.net_g)
        min_size = 8000 * 8000 # use smaller min_size with limited GPU memory
        lq_input = self.lqsrc
        _, _, h, w = lq_input.shape
        if h*w < min_size:
            self.output = net_g.test(lq_input)
        else:
            self.output = net_g.test_tile(lq_input)
        self.net_g.train()
        
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            if self.opt['val'].get('grids', False):
                self.grids()
            self.test()
            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gtsrc

            # tentative for out of GPU memory
            del self.lqsrc
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], str(current_iter),dataset_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if name =='niqe' or name =='piqe' or name =='brisque':
                        metric_data = dict(img=sr_img)
                    else:
                        metric_data = dict(img1=sr_img, img2=gt_img)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
    
    def vis_single_code(self, up_factor=2):
        net_g = self.get_bare_model(self.net_g)
        codenum = self.opt['network_g']['codebook_params'][0][1]
        with torch.no_grad():
            code_idx = torch.arange(codenum).reshape(codenum, 1, 1, 1)
            code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
            output_img = net_g.decode_indices(code_idx) 
            output_img = tvu.make_grid(output_img, nrow=32)

        return output_img.unsqueeze(0)

    def get_current_visuals(self):
        vis_samples = 16
        out_dict = OrderedDict()
        out_dict['lq'] = self.lqsrc.detach().cpu()[:vis_samples]
        out_dict['result'] = self.output.detach().cpu()[:vis_samples]
        if not self.LQ_stage:
            out_dict['codebook'] = self.vis_single_code()
        if hasattr(self, 'gt_rec'):
            out_dict['gt_rec'] = self.gt_rec.detach().cpu()[:vis_samples]
        if hasattr(self, 'gtsrc'):
            out_dict['gt'] = self.gtsrc.detach().cpu()[:vis_samples]
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_network(self.net_d, 'net_dt', current_iter)
        self.save_network(self.net_d, 'net_df', current_iter)
        self.save_training_state(epoch, current_iter)

@MODEL_REGISTRY.register()
class QDMR_UDAModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

         # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        
        self.net_s = build_network(opt['network_s'])
        self.net_s = self.model_to_device(self.net_s)


        # load pre-trained dmc ckpt, frozen decoder and codebook 
        self.LQ_stage = self.opt['network_g'].get('LQ_stage', False) 
        if self.LQ_stage:
            load_path = self.opt['path'].get('pretrain_network_dmc', None)
            load_path_s = self.opt['path'].get('pretrain_network_s', None)
            assert load_path is not None, 'Need to specify hq prior model path in LQ stage'

            self.load_network(self.net_g, load_path, False)
            frozen_module_keywords = self.opt['network_g'].get('frozen_module_keywords', None) 
            if frozen_module_keywords is not None:
                for name, module in self.net_g.named_modules():
                    for fkw in frozen_module_keywords:
                        if fkw in name:
                            for p in module.parameters():
                                p.requires_grad = False
                            break
                        
            self.load_network(self.net_s, load_path_s, False)
            frozen_module_keywords = self.opt['network_s'].get('frozen_module_keywords', None) 
            if frozen_module_keywords is not None:
                for name, module in self.net_s.named_modules():
                    for fkw in frozen_module_keywords:
                        if fkw in name:
                            for p in module.parameters():
                                p.requires_grad = False
                            break

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        logger = get_root_logger()
        if load_path is not None:
            logger.info(f'Loading net_g from {load_path}')
            self.load_network(self.net_g, load_path, self.opt['path']['strict_load'])
            
        if self.is_train:
            self.init_training_settings()
            self.use_dis = (self.opt['train']['gan_opt']['loss_weight'] != 0) 
            self.net_d_best = copy.deepcopy(self.net_d)
        
        self.net_g_best = copy.deepcopy(self.net_g)

    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']
        self.net_g.train()
        # self.net_s.train()
        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        
        self.net_dt = build_network(self.opt['network_dt'])
        self.net_dt = self.model_to_device(self.net_dt)
        # load pretrained d models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        # print(load_path)
        if load_path is not None:
            logger.info(f'Loading net_d from {load_path}')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))
            
        self.net_d.train()
        self.net_dt.train()
        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
            self.model_to_device(self.cri_perceptual)
        else:
            self.cri_perceptual = None


        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
            
        if train_opt.get('gan_opt_t'):
            self.cri_gan_t = build_loss(train_opt['gan_opt_t']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
    
    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            optim_params.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_g = optim_class(optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        optim_params_d = []
        for k, v in self.net_d.named_parameters():
            optim_params_d.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        for k, v in self.net_dt.named_parameters():
            optim_params_d.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_d = optim_class(optim_params_d, **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, datasrc, datatrg=None):
        self.lqsrc = datasrc['lq'].to(self.device)
        # self.lq = self.lqsrc
        if datatrg is not None:
            self.lqtrg = datatrg['lq'].to(self.device)
        if 'gt' in datasrc:
            self.gtsrc = datasrc['gt'].to(self.device)
            if datatrg is not None:
                self.gttrg = datatrg['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']

        for p in self.net_d.parameters():
            p.requires_grad = False
        for p in self.net_dt.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()
        
        #s2t
        with torch.no_grad():
            self.s2t, _, _, _= self.net_s(self.lqsrc)
            # min_max = (0, 1)
            # self.s2t = self.s2t.clamp_(*min_max)
        #src_forward
        self.output_residual, l_codebook, _, feat_quant= self.net_g(self.lqsrc)
        #s2t_forward
        self.output_s2t, l_codebook_s2t, _, feat_quant_s2t= self.net_g(self.s2t)

        
        
        l_g_total = 0
        loss_dict = OrderedDict()



        # ===================================================
        # codebook loss (can be discarded)
        if train_opt.get('codebook_opt', None):
            l_codebook *= train_opt['codebook_opt']['loss_weight'] 
            l_g_total += l_codebook.mean()
            loss_dict['l_codebook'] = l_codebook.mean()
            l_g_total += l_codebook_s2t.mean()
            loss_dict['l_codebook_s2t'] = l_codebook_s2t.mean()


        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output_residual, self.gtsrc)
            l_g_total += l_pix
            loss_dict['l_pix'] = l_pix
            l_pix_s2t = self.cri_pix(self.output_s2t, self.gtsrc)
            l_g_total += l_pix_s2t
            loss_dict['l_pix_s2t'] = l_pix_s2t


        
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output_residual, self.gtsrc)
            l_percep_s2t, l_style_s2t = self.cri_perceptual(self.output_s2t, self.gtsrc)
            if l_percep is not None:
                l_g_total += l_percep.mean()
                loss_dict['l_percep'] = l_percep.mean()
                l_g_total += l_percep_s2t.mean()
                loss_dict['l_percep_s2t'] = l_percep_s2t.mean()
            if l_style is not None:
                l_g_total += l_style
                loss_dict['l_style'] = l_style
        # l_g_total.mean().backward()
        
        
        self.toutput_residual, tl_codebook, _, tfeat_quant = self.net_g(self.lqtrg)
        l_g_total += 0.1*tl_codebook.mean()
        loss_dict['tl_codebook'] = 0.1*tl_codebook.mean()

        if self.use_dis and current_iter > train_opt['net_d_init_iters']:

            trg_fuse_pred = self.net_d(feat_quant)
            l_g_gan_trgfuse = self.cri_gan(trg_fuse_pred, 'half', is_disc=False)
            l_g_total += l_g_gan_trgfuse
            loss_dict['l_g_gan_srcfuse'] = l_g_gan_trgfuse

            trgs_fuse_pred = self.net_d(tfeat_quant)
            l_g_gan_trgsfuse = 0.1*self.cri_gan(trgs_fuse_pred, 'half', is_disc=False)
            l_g_total += l_g_gan_trgsfuse
            loss_dict['l_g_gan_trgfuse'] = l_g_gan_trgsfuse
            
            out_fake_g_preds = self.net_dt(self.output_residual)
            l_g_gans = self.cri_gan_t(out_fake_g_preds, True, is_disc=False)
            l_g_total += l_g_gans
            loss_dict['l_g_gans'] = l_g_gans
        
        
            out_fake_g_predst = self.net_dt(self.output_s2t)
            l_g_ganst = self.cri_gan_t(out_fake_g_predst, True, is_disc=False)
            l_g_total += l_g_ganst
            loss_dict['l_g_ganst'] = l_g_ganst
            
        l_g_total.mean().backward()
        self.optimizer_g.step()

        # optimize net_d
        self.fixed_disc = self.opt['train'].get('fixed_disc', False)
        if not self.fixed_disc and self.use_dis and current_iter > train_opt['net_d_init_iters']:
            for p in self.net_d.parameters():
                p.requires_grad = True
            for p in self.net_dt.parameters():
                p.requires_grad = True
            self.optimizer_d.zero_grad()

            trg_fuse_real = self.net_d(tfeat_quant.detach())
            l_d_trgreal = self.cri_gan(trg_fuse_real, True, is_disc=True)
            loss_dict['l_d_trgreal'] = l_d_trgreal
            loss_dict['out_d_trgreal'] = torch.mean(trg_fuse_real.detach())
            l_d_trgreal.backward()

            src_fuse_fake = self.net_d(feat_quant.detach())
            l_d_srcfake= self.cri_gan(src_fuse_fake, False, is_disc=True)
            loss_dict['l_d_srcfake'] = l_d_srcfake
            loss_dict['out_d_srcfake'] = torch.mean(src_fuse_fake.detach())
            l_d_srcfake.backward()

        
            real_d_pred = self.net_dt(self.gtsrc)
            l_d_reals = self.cri_gan_t(real_d_pred, True, is_disc=True)
            loss_dict['l_d_reals'] = l_d_reals
            loss_dict['out_d_reals'] = torch.mean(real_d_pred.detach())
            l_d_reals.backward()
            
            fake_d_pred = self.net_dt(self.output_residual.detach())
            l_d_fake = self.cri_gan_t(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()
                   
            fake_d_predst = self.net_dt(self.output_s2t.detach())
            l_d_fakest = self.cri_gan_t(fake_d_predst, False, is_disc=True)
            loss_dict['l_d_fakest'] = l_d_fakest
            loss_dict['out_d_fakest'] = torch.mean(fake_d_predst.detach())
            l_d_fakest.backward()
            
            self.optimizer_d.step()

        
        self.log_dict = self.reduce_loss_dict(loss_dict)
        
    def test(self):
        self.net_g.eval()
        net_g = self.get_bare_model(self.net_g)
        min_size = 8000 * 8000 # use smaller min_size with limited GPU memory
        lq_input = self.lqsrc
        _, _, h, w = lq_input.shape
        if h*w < min_size:
            self.output = net_g.test(lq_input)
        else:
            self.output = net_g.test_tile(lq_input)
        self.net_g.train()




    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            if self.opt['val'].get('grids', False):
                self.grids()
            self.test()
            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            if 'gtsrc' in visuals:
                gt_img = tensor2img([visuals['gtsrc']])
                del self.gtsrc

            # tentative for out of GPU memory
            del self.lqsrc
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], str(current_iter),dataset_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if name =='niqe' or name =='piqe' or name =='brisque':
                        metric_data = dict(img=sr_img)
                    else:
                        metric_data = dict(img1=sr_img, img2=gt_img)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
    
    def vis_single_code(self, up_factor=2):
        net_g = self.get_bare_model(self.net_g)
        codenum = self.opt['network_g']['codebook_params'][0][1]
        with torch.no_grad():
            code_idx = torch.arange(codenum).reshape(codenum, 1, 1, 1)
            code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
            output_img = net_g.decode_indices(code_idx) 
            output_img = tvu.make_grid(output_img, nrow=32)

        return output_img.unsqueeze(0)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lqsrc'] = self.lqsrc.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gtsrc'):
            out_dict['gtsrc'] = self.gtsrc.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_network(self.net_dt, 'net_dt', current_iter)
        self.save_network(self.net_s, 'net_s', current_iter)
        self.save_training_state(epoch, current_iter)

@MODEL_REGISTRY.register()
class QDMR_BaseModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

         # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)


        # load pre-trained DMC ckpt, frozen decoder and codebook 
        self.LQ_stage = self.opt['network_g'].get('LQ_stage', False) 
        if self.LQ_stage:
            load_path = self.opt['path'].get('pretrain_network_dmc', None)
            if load_path is not None:
                assert load_path is not None, 'Need to specify dmc prior model path in LQ stage'
                self.load_network(self.net_g, load_path, False)
                
            frozen_module_keywords = self.opt['network_g'].get('frozen_module_keywords', None) 
            if frozen_module_keywords is not None:
                for name, module in self.net_g.named_modules():
                    for fkw in frozen_module_keywords:
                        if fkw in name:
                            for p in module.parameters():
                                p.requires_grad = False
                            break

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        logger = get_root_logger()
        if load_path is not None:
            logger.info(f'Loading net_g from {load_path}')
            self.load_network(self.net_g, load_path, self.opt['path']['strict_load'])
            
        if self.is_train:
            self.init_training_settings()
            self.use_dis = (self.opt['train']['gan_opt']['loss_weight'] != 0) 
            self.net_d_best = copy.deepcopy(self.net_d)
        
        self.net_g_best = copy.deepcopy(self.net_g)

    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']
        self.net_g.train()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        # load pretrained d models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        # print(load_path)
        if load_path is not None:
            logger.info(f'Loading net_d from {load_path}')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))
            
        self.net_d.train()
    
        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
            self.model_to_device(self.cri_perceptual)
        else:
            self.cri_perceptual = None


        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
    
    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            optim_params.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_g = optim_class(optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_d = optim_class(self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, datasrc, datatrg=None):
        self.lqsrc = datasrc['lq'].to(self.device)
        if datatrg is not None:
            self.lqtrg = datatrg['lq'].to(self.device)
        if 'gt' in datasrc:
            self.gtsrc = datasrc['gt'].to(self.device)
            if datatrg is not None:
                self.gttrg = datatrg['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']

        for p in self.net_d.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()

    #src
        if self.LQ_stage:
            # with torch.no_grad():
            #     self.gt_rec, _, _, gt_indices = self.net_hq(self.gtsrc)
            self.output, self.output_residual, l_codebook, l_semantic, quant_g, _, _ = self.net_g(self.lqsrc)
            # self.output_residual, quant_s = self.net_g(self.lqsrc)


        l_g_total = 0
        loss_dict = OrderedDict()

        # ===================================================
        # codebook loss
        if train_opt.get('codebook_opt', None):
            l_codebook *= train_opt['codebook_opt']['loss_weight'] 
            l_g_total += l_codebook.mean()
            loss_dict['l_codebook'] = l_codebook.mean()

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output_residual, self.gtsrc)
            l_g_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output_residual, self.gtsrc)
            if l_percep is not None:
                l_g_total += l_percep.mean()
                loss_dict['l_percep'] = l_percep.mean()
            if l_style is not None:
                l_g_total += l_style
                loss_dict['l_style'] = l_style
        
        # gan loss
        if self.use_dis and current_iter > train_opt['net_d_init_iters']:

            out_fake_g_pred = self.net_d(self.output_residual)
            # print(fake_g_pred.size())
            l_g_gan = self.cri_gan(out_fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan
            
        l_g_total.mean().backward()

        self.optimizer_g.step()

        # optimize net_d
        self.fixed_disc = self.opt['train'].get('fixed_disc', False)
        if not self.fixed_disc and self.use_dis and current_iter > train_opt['net_d_init_iters']:
            for p in self.net_d.parameters():
                p.requires_grad = True
            self.optimizer_d.zero_grad()
            
            real_d_pred = self.net_d(self.gtsrc)
            l_d_reals = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_reals'] = l_d_reals
            loss_dict['out_d_reals'] = torch.mean(real_d_pred.detach())
            l_d_reals.backward()
            
            fake_d_pred = self.net_d(self.output_residual.detach())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()

            self.optimizer_d.step()
        
        self.log_dict = self.reduce_loss_dict(loss_dict)
        
    def test(self):
        self.net_g.eval()
        net_g = self.get_bare_model(self.net_g)
        min_size = 8000 * 8000 # use smaller min_size with limited GPU memory
        lq_input = self.lqsrc
        _, _, h, w = lq_input.shape
        if h*w < min_size:
            self.output = net_g.test(lq_input)
        else:
            self.output = net_g.test_tile(lq_input)
        self.net_g.train()

        
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            if self.opt['val'].get('grids', False):
                self.grids()
            self.test()
            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            if 'gtsrc' in visuals:
                gt_img = tensor2img([visuals['gtsrc']])
                del self.gtsrc

            # tentative for out of GPU memory
            del self.lqsrc
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], str(current_iter),dataset_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if name =='niqe' or name =='piqe' or name =='brisque':
                        metric_data = dict(img=sr_img)
                    else:
                        metric_data = dict(img1=sr_img, img2=gt_img)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
    
    def vis_single_code(self, up_factor=2):
        net_g = self.get_bare_model(self.net_g)
        codenum = self.opt['network_g']['codebook_params'][0][1]
        with torch.no_grad():
            code_idx = torch.arange(codenum).reshape(codenum, 1, 1, 1)
            code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
            output_img = net_g.decode_indices(code_idx) 
            output_img = tvu.make_grid(output_img, nrow=32)

        return output_img.unsqueeze(0)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lqsrc'] = self.lqsrc.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gtsrc'):
            out_dict['gtsrc'] = self.gtsrc.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)


