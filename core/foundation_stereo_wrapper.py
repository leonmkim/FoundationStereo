#%%
from pathlib import Path
import sys
# import path to foundation_stereo
root_foundation_stereo_path = Path(__file__).parent.parent.parent
assert root_foundation_stereo_path.exists(), f"FoundationStereo path does not exist: {root_foundation_stereo_path}"
sys.path.append(str(root_foundation_stereo_path))
#%%
from FoundationStereo.core.utils.utils import InputPadder
from FoundationStereo.Utils import *
from FoundationStereo.core.foundation_stereo import *
#%%
from omegaconf import OmegaConf

set_logging_format()

class FoundationStereoWrapper:
    def __init__(self, ckpt_path: Path, camera_intrinsic_left: np.ndarray, baseline: float, 
                 scale: float=1.0, hiera: int=1, valid_iters: int=32, remove_invisible: int=1):
        self.ckpt_path = ckpt_path
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


    def infer(self, img0, img1):
        # Inference code here
        img0 = cv2.resize(img0, fx=self.args.scale, fy=self.args.scale, dsize=None)
        img1 = cv2.resize(img1, fx=self.args.scale, fy=self.args.scale, dsize=None)
        H,W = img0.shape[:2]
        img0_ori = img0.copy()
        logging.info(f"img0: {img0.shape}")

        img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
        img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        with torch.cuda.amp.autocast(True):
            if not self.args.hiera:
                disp = self.model.forward(img0, img1, iters=self.args.valid_iters, test_mode=True)
            else:
                disp = self.model.run_hierachical(img0, img1, iters=self.args.valid_iters, test_mode=True, small_ratio=0.5)
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H,W)
        vis = vis_disparity(disp)
        vis = np.concatenate([img0_ori, vis], axis=1)

        if self.args.remove_invisible:
            yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
            us_right = xx-disp
            invalid = us_right<0
            disp[invalid] = np.inf
        
        left_depth_image = (self.camera_intrinsic_left[0,0]*self.baseline/disp)/1000.0  # in meters
        return left_depth_image
