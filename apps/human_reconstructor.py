import os
import glob
from torch import nn
from apps.data_loader import DataLoader, init_semantic_labels
from depth_predictor.external_interface import HumanRecon
import cv2

# facade class for reconstructing humans
class HumanReconstructorFacade(nn.Module):
    def __init__(self, params, cam_params):
        super(HumanReconstructorFacade, self).__init__()
        self.params = params
        self.cam_params = cam_params
        
        self.path2image = os.path.join(self.params.path2data, self.params.dataset.color_path)
        self.path2mask = os.path.join(self.params.path2data, self.params.dataset.mask_path)
        self.path2depth = os.path.join(self.params.path2data, self.params.dataset.depth_path)
        self.path2smpl = os.path.join(self.params.path2data, self.params.dataset.smpl_path)
        self.path2results = os.path.join(self.params.path2data, self.params.dataset.recon_path)
        
        self.esr_ckpt = os.path.join(self.params.dataset.data_dir, self.params.recon.esr_ckpt)
        self.lbs_ckpt = os.path.join(self.params.dataset.data_dir, self.params.recon.lbs_ckpt)
        self.recon_ckpt = os.path.join(self.params.dataset.data_dir, self.params.recon.recon_ckpt)
        self.color_ckpt = os.path.join(self.params.dataset.data_dir, self.params.recon.color_ckpt)
        
        self.v_label = init_semantic_labels(os.path.join(self.params.dataset.data_dir, self.params.recon.smpl_semantic))

        self.data_dirs = sorted(glob.glob(os.path.join(self.path2image, '*')))
        self.data_list = []
        for data_dir in self.data_dirs:
            tmp_data_list = sorted(os.listdir(data_dir))
            for data_list in tmp_data_list:
                self.data_list.append(os.path.join(data_dir, data_list))
                
        self.recon_models = HumanRecon(result_path=self.path2results,
                                       ckpt_path=self.recon_ckpt,
                                       color_ckpt_path=self.color_ckpt,
                                       model_name=self.params.recon.model_name,
                                       model_C_name=self.params.recon.model_C_name,
                                       esr_path=self.esr_ckpt,
                                       lbs_ckpt=self.lbs_ckpt,
                                       v_label=self.v_label,
                                       cam_params=self.cam_params,
                                       params=self.params.recon,)

    def fetch_next_data(self):
        assert len(self.data_list) > 0, 'No data left in the queue'
        data = self.data_list.pop(0)
        self.data_dir = data.split('/')[-2]
        self.data_name = data.split('/')[-1]
        self.ext = self.data_name.split('.')[-1]
        self.save_dir = os.path.join(self.path2results, self.data_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        
        image = DataLoader.load_image(data, pred_res=self.cam_params['width'])
        mask = DataLoader.load_mask(os.path.join(self.path2mask, self.data_dir, self.data_name), pred_res=self.cam_params['width'])
        
        depth_front, depth_back = DataLoader.load_depth(os.path.join(self.path2depth, self.data_dir, self.data_name), pred_res=self.cam_params['width'])
        smpl_param = DataLoader.load_smpl_params(os.path.join(self.path2smpl, self.data_dir, self.data_dir + '.json'))
        return image, mask, depth_front, depth_back, data, smpl_param

    def forward(self, image, mask=None, depth_front=None, depth_back=None, smpl_params=None, cam_params=None):
        lbs_posed_mesh, color_posed_mesh, color_front, color_back, lbs_front, lbs_back, process_time, process_time_full \
            = self.recon_models(image, depth_front=depth_front, depth_back=depth_back, mask=mask, smpl_params=smpl_params, smpl_resource=self.params.recon.smpl_resource, cam_params=cam_params)

        cv2.imwrite(os.path.join(self.save_dir, self.data_name.replace(self.ext, '_color_front.' + self.ext)), color_front)
        cv2.imwrite(os.path.join(self.save_dir, self.data_name.replace(self.ext, '_color_back.' + self.ext)), color_back)
        
        cv2.imwrite(os.path.join(self.save_dir, self.data_name.replace(self.ext, '_lbs_front.' + self.ext)), lbs_front)
        cv2.imwrite(os.path.join(self.save_dir, self.data_name.replace(self.ext, '_lbs_back.' + self.ext)), lbs_back)        

        lbs_posed_mesh.export(os.path.join(self.save_dir, self.data_name.replace(self.ext, '_lbs.obj')))

        color_posed_mesh.export(os.path.join(self.save_dir, self.data_name.replace(self.ext, '_color.obj')))
        return process_time, process_time_full
