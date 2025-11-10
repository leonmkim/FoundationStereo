#%%
from pathlib import Path
import sys
from typing import Union
# import path to foundation_stereo
root_foundation_stereo_path = Path(__file__).parent.parent.parent
assert root_foundation_stereo_path.exists(), f"FoundationStereo path does not exist: {root_foundation_stereo_path}"
sys.path.append(str(root_foundation_stereo_path))
import torchvision
import einops
#%%
from FoundationStereo.core.utils.utils import InputPadder
from FoundationStereo.Utils import *
from FoundationStereo.core.foundation_stereo import *
#%%
from omegaconf import OmegaConf

set_logging_format()

class FoundationStereoWrapper:
    def __init__(self, ckpt_path: Path, camera_intrinsic_left: Union[torch.Tensor, np.ndarray], baseline: Union[torch.Tensor, float], 
                 scale: float=1.0, hiera: int=1, valid_iters: int=32, remove_invisible: int=1):
        self.ckpt_path = ckpt_path
        if isinstance(camera_intrinsic_left, np.ndarray):
            assert isinstance(baseline, float), "If camera_intrinsic_left is numpy array, baseline must be float"
            assert camera_intrinsic_left.ndim == 2 and camera_intrinsic_left.shape == (3,3), "camera_intrinsic_left must be 3x3 numpy array"
            self.batch_inference_mode = False
        elif isinstance(camera_intrinsic_left, torch.Tensor):
            assert isinstance(baseline, torch.Tensor), "If camera_intrinsic_left is torch tensor, baseline must be torch tensor"
            self.batch_inference_mode = True
            camera_intrinsic_left = camera_intrinsic_left.cuda().float()
            baseline = baseline.cuda().float()
        else:
            raise TypeError("camera_intrinsic_left must be either numpy array or torch tensor")
        
        self.camera_intrinsic_left = camera_intrinsic_left
        self.baseline = baseline

        assert ckpt_path.exists(), f"Checkpoint does not exist: {ckpt_path}"
        assert scale<=1.0, "scale must be <=1.0"
        self.camera_intrinsic_left[:2] *= scale
        # with open(args['intrinsic_file'], 'r') as f:
        #     lines = f.readlines()
        #     K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
        #     baseline = float(lines[1])
        args_dict = dict(
            scale=scale, # 'downsize the image by scale, must be <=1'
            hiera=hiera, # 'hierarchical inference (only needed for high-resolution images (>1K))'
            valid_iters=valid_iters, # 'number of flow-field updates during forward pass'
            remove_invisible=remove_invisible, # 'remove non-overlapping observations between left and right images
            # z_far=10, # 'max depth to clip in point cloud'
            # get_pc=1, # 'save point cloud output'
            # denoise_cloud=1, # 'whether to denoise the point cloud'
            # denoise_nb_points=30, # 'number of points to consider for radius outlier
            # denoise_radius=0.03, # 'radius to use for outlier removal'
        )

        set_seed(0)
        torch.autograd.set_grad_enabled(False)

        config_path = ckpt_path.parent / 'cfg.yaml'
        cfg = OmegaConf.load(config_path)
        if 'vit_size' not in cfg:
            cfg['vit_size'] = 'vitl'
        for k, v in args_dict.items():
            cfg[k] = v
        self.args = OmegaConf.create(cfg)
        # logging.info(f"args:\n{args}")
        logging.info(f"Using pretrained model from {ckpt_path}")

        self.model = FoundationStereo(self.args)

        ckpt = torch.load(ckpt_path)
        logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
        self.model.load_state_dict(ckpt['model'])

        self.model.cuda()
        self.model.eval()

    @torch.inference_mode()
    def inference(self, img0, img1):
        # Inference code here
        assert img0.shape == img1.shape, "Left and right images must have the same shape"

        if isinstance(img0, np.ndarray):
            assert img0.ndim == 3, "img0 must be HWC numpy array"
            assert img0.shape[2] == 3, "img0 must have 3 channels"
            # put numpy array to torch tensor on GPU
            img0 = einops.rearrange(torch.as_tensor(img0).cuda().float(), 'h w c -> () c h w')
        if isinstance(img1, np.ndarray):
            img1 = einops.rearrange(torch.as_tensor(img1).cuda().float(), 'h w c -> () c h w')

        
        if self.args.scale != 1.0:
            img0 = torchvision.transforms.v2.functional.resize(img0, size=None, scale_factor=self.args.scale, interpolation=torchvision.transforms.InterpolationMode.LANCZOS, antialias=True)
            img1 = torchvision.transforms.v2.functional.resize(img1, size=None, scale_factor=self.args.scale, interpolation=torchvision.transforms.InterpolationMode.LANCZOS, antialias=True)
        
        # img0 = img0.float()
        # img1 = img1.float()
        assert img0.shape[1] == 3, "Input images must have 3 channels"

        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        with torch.amp.autocast(enabled=True, dtype=torch.bfloat16, device_type='cuda'):
            if not self.args.hiera:
                disp = self.model.forward(img0, img1, iters=self.args.valid_iters, test_mode=True)
            else:
                disp = self.model.run_hierachical(img0, img1, iters=self.args.valid_iters, test_mode=True, small_ratio=0.5)
        disp = padder.unpad(disp.float())
        # disp = disp.data.cpu().numpy().reshape(H,W)

        if self.args.remove_invisible:
            yy,xx = torch.meshgrid(torch.arange(disp.shape[-2], device=disp.device), torch.arange(disp.shape[-1], device=disp.device), indexing='ij')
            us_right = xx - disp
            invalid = us_right < 0
            disp[invalid] = float('inf')
            # yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
            # us_right = xx-disp
            # invalid = us_right<0
            # disp[invalid] = np.inf
        
        if not self.batch_inference_mode:
            left_depth_image = (self.camera_intrinsic_left[0,0]*self.baseline/disp)/1000.0  # in meters
            left_depth_image = left_depth_image[0,0].cpu().numpy()
        else:
            # camera_intrinsic_left: Bx3x3, baseline: B tensors
            # disp: Bx1xHxW
            to_depth_scalars = einops.rearrange((self.camera_intrinsic_left[:,0,0]*self.baseline)/1000.0, 'b -> b () () ()')  # B tensors, in meters
            left_depth_image = to_depth_scalars * (1.0/disp)  # in meters

        return left_depth_image

    def batch_inference(self, batch_img0, batch_img1):
        raise NotImplementedError("Batch inference is not implemented yet.")
