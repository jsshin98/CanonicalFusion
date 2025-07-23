import os
import open3d as o3d
import PIL.Image as pil_img
from .keypoints_utils import *
from smpl_optimizer.externals.expose.data.transforms import build_transforms
from smpl_optimizer.externals.expose.data.targets import BoundingBox
from smpl_optimizer.externals.expose.utils.plot_utils import HDRenderer
from smpl_optimizer.externals.expose.data.utils.bbox import bbox_to_center_scale
from collections import OrderedDict, defaultdict
from torchvision.models.detection import keypointrcnn_resnet50_fpn

Vec3d = o3d.utility.Vector3dVector
Vec3i = o3d.utility.Vector3iVector


def undo_img_normalization(image, mean, std, add_alpha=True):
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy().squeeze()

    out_img = (image * std[np.newaxis, :, np.newaxis, np.newaxis] +
               mean[np.newaxis, :, np.newaxis, np.newaxis])
    if add_alpha:
        out_img = np.pad(
            out_img, [[0, 0], [0, 1], [0, 0], [0, 0]],
            mode='constant', constant_values=1.0)
    return out_img


def weak_persp_to_blender(
        target,
        camera_scale,
        camera_transl,
        H, W,
        sensor_width=36,
        focal_length=5000):
    ''' Converts weak-perspective camera to a perspective camera
    '''
    if torch.is_tensor(camera_scale):
        camera_scale = camera_scale.detach().cpu().numpy()
    if torch.is_tensor(camera_transl):
        camera_transl = camera_transl.detach().cpu().numpy()

    output = defaultdict(lambda: [])
    orig_bbox_size = target.get_field('orig_bbox_size')
    bbox_center = target.get_field('orig_center')
    z = 2 * focal_length / (camera_scale * orig_bbox_size)

    transl = [
        camera_transl[0, 0].item(), camera_transl[0, 1].item(),
        z.item()]
    shift_x = - (bbox_center[0] / W - 0.5)
    shift_y = (bbox_center[1] - 0.5 * H) / W
    focal_length_in_mm = focal_length / W * sensor_width
    output['shift_x'].append(shift_x)
    output['shift_y'].append(shift_y)
    output['transl'].append(transl)
    output['focal_length_in_mm'].append(focal_length_in_mm)
    output['focal_length_in_px'].append(focal_length)
    output['center'].append(bbox_center)
    output['sensor_width'].append(sensor_width)
    output['scale'].append(camera_scale[0])
    for key in output:
        output[key] = np.stack(output[key], axis=0)
    return output

def rcnn_process(image_list, exp_cfg, device):
    scale_factor = 1.2
    min_score = 0.5

    rcnn_model = keypointrcnn_resnet50_fpn(weights=True)
    rcnn_model.eval()
    rcnn_model = rcnn_model.to(device=device)
    output = rcnn_model(image_list)

    bboxes = []
    fullimg_list = []
    cropimg_list = []
    target_list = []
    for ii, x in enumerate(output):
        for n, bbox in enumerate(output[ii]['boxes']):
            bbox = bbox.detach().cpu().numpy()
            if output[ii]['scores'][n].item() < min_score:
                continue
            bboxes.append(bbox)

    dataset_cfg = exp_cfg.get('datasets', {})
    body_dsets_cfg = dataset_cfg.get('body', {})
    body_transfs_cfg = body_dsets_cfg.get('transforms', {})
    transforms = build_transforms(body_transfs_cfg, is_train=False)

    for idx in range(len(image_list)):
        img = image_list[idx].permute(1, 2, 0).cpu()
        bbox = bboxes[idx]
        target = BoundingBox(bbox, size=img.shape)
        center, scale, bbox_size = bbox_to_center_scale(
            bbox, dset_scale_factor=scale_factor)
        target.add_field('bbox_size', bbox_size)
        target.add_field('orig_bbox_size', bbox_size)
        target.add_field('orig_center', center)
        target.add_field('center', center)
        target.add_field('scale', scale)
        if transforms is not None:
            full_img, cropped_image, target = transforms(img, target)
        fullimg_list.append(full_img)
        cropimg_list.append(cropped_image)
        target_list.append(target)
    return fullimg_list, cropimg_list, target_list


def rotmat2vec(rotmats):
    """
    Convert rotation matrices to angle-axis using opencv's Rodrigues formula.
    Args:
        rotmats: A np array of shape (..., 3, 3)

    Returns:
        A np array of shape (..., 3)
    """
    assert rotmats.shape[-1] == 3 and rotmats.shape[-2] == 3 and len(rotmats.shape) >= 3, 'invalid input dimension'
    orig_shape = rotmats.shape[:-2]
    rots = np.reshape(rotmats, [-1, 3, 3])
    aas = np.zeros([rots.shape[0], 3])
    for i in range(rots.shape[0]):
        aas[i] = np.squeeze(cv2.Rodrigues(rots[i])[0])
    return np.reshape(aas, orig_shape + (3,))


def smplx_fitting(save_path,
                  exp_cfg,
                  model,
                  img,
                  cropped_img,
                  target,
                  save_vis=False,
                  save_mesh=False,
                  save_params=False,
                  show=False,
                  device=None):

    sensor_width = 36.0
    focal_length = 5000.0

    means = np.array(exp_cfg['datasets']['body']['transforms']['mean'])
    std = np.array(exp_cfg['datasets']['body']['transforms']['std'])
    render = save_vis or show
    body_crop_size = exp_cfg.get('datasets', {}).get('body', {}).get(
        'transforms').get('crop_size', 256)

    if render:
        hd_renderer = HDRenderer(img_size=body_crop_size)

    full_imgs = img.to(device=device)
    body_imgs = cropped_img.to(device=device)
    body_targets = target.to(device)

    torch.cuda.synchronize()
    model_output = model(body_imgs.unsqueeze(0), body_targets, full_imgs=full_imgs.unsqueeze(0),
                                    device=device)
    torch.cuda.synchronize()
    hd_imgs = full_imgs.detach().cpu().numpy().squeeze()
    _, H, W = full_imgs.shape

    if render:
        hd_imgs = np.transpose(undo_img_normalization(hd_imgs, means, std),
                               [0, 2, 3, 1])
        hd_imgs = np.clip(hd_imgs, 0, 1.0)

    body_output = model_output.get('body', {})
    num_stages = body_output.get('num_stages', 3)
    stage_n_out = body_output.get(f'stage_{num_stages - 1:02d}', {})
    model_vertices = stage_n_out.get('vertices', None)

    if stage_n_out is not None:
        model_vertices = stage_n_out.get('vertices', None)

    faces = stage_n_out['faces']
    if model_vertices is not None:
        model_vertices = model_vertices.detach().cpu().numpy()
        camera_parameters = body_output.get('camera_parameters', {})
        camera_scale = camera_parameters['scale'].detach()
        camera_transl = camera_parameters['translation'].detach()

    out_img = OrderedDict()
    final_model_vertices = None
    final_out = model_output.get('body', {}).get('final', {})

    stage_n_out['hd_proj_joints'] = model_output.get('body', {}).get('hd_proj_joints', {})
    stage_n_out['faces'] = torch.Tensor(faces.astype(int)).unsqueeze(0)
    stage_n_out['img_width'] = full_imgs.shape[1]
    stage_n_out['img_height'] = full_imgs.shape[2]

    if final_out is not None:
        final_model_vertices = final_out.get('vertices', None)
    if final_model_vertices is not None:
        final_model_vertices = final_model_vertices.detach().cpu().numpy()
        camera_parameters = model_output.get('body', {}).get(
            'camera_parameters', {})
        camera_scale = camera_parameters['scale'].detach()
        camera_transl = camera_parameters['translation'].detach()

    hd_params = weak_persp_to_blender(
        body_targets,
        camera_scale=camera_scale,
        camera_transl=camera_transl,
        H=H, W=W,
        sensor_width=sensor_width,
        focal_length=focal_length,
    )

    if save_vis:
        bg_hd_imgs = np.transpose(hd_imgs, [0, 3, 1, 2])
        out_img['hd_imgs'] = bg_hd_imgs

    # Render the initial predictions on the original image resolution
    if render:
        hd_orig_overlays = hd_renderer(
            model_vertices,
            faces,
            focal_length=hd_params['focal_length_in_px'],
            camera_translation=hd_params['transl'],
            camera_center=hd_params['center'],
            bg_imgs=bg_hd_imgs,
            return_with_alpha=True,
        )
        out_img['hd_orig_overlay'] = hd_orig_overlays

    # Render the overlays of the final prediction
    if render:
        hd_overlays = hd_renderer(
            final_model_vertices,
            faces,
            focal_length=hd_params['focal_length_in_px'],
            camera_translation=hd_params['transl'],
            camera_center=hd_params['center'],
            bg_imgs=bg_hd_imgs,
            return_with_alpha=True,
            body_color=[0.4, 0.4, 0.7]
        )
        out_img['hd_overlay'] = hd_overlays

    if save_vis:
        for key in out_img.keys():
            out_img[key] = np.clip(
                np.transpose(
                    out_img[key], [0, 2, 3, 1]) * 255, 0, 255).astype(
                np.uint8)

    if save_vis:
        for name, curr_img in out_img.items():
            pil_img.fromarray(curr_img[0]).save(
                os.path.join(save_path, f'{name}.png'))

    if save_mesh:
        # Store the mesh predicted by the body-crop network
        naive_mesh = o3d.geometry.TriangleMesh()
        naive_mesh.vertices = Vec3d(
            (model_vertices[0] + hd_params['transl'][0]))
        naive_mesh.triangles = Vec3i(faces)

        # Store the final mesh
        expose_mesh = o3d.geometry.TriangleMesh()
        expose_mesh.vertices = Vec3d(
            (final_model_vertices[0] + hd_params['transl'][0]))
        expose_mesh.triangles = Vec3i(faces)

        lbs = np.asarray(model.smplx.body_model.lbs_weights.cpu())
        colormap = np.random.rand(55, 3)
        new_color = np.matmul(lbs, colormap)
        skinning_mesh = trimesh.Trimesh(vertices=np.asarray(naive_mesh.vertices),
                                        faces=faces,
                                        vertex_colors=new_color,
                                        maintain_order=True,
                                        process=False)

    if save_params:
        params_fname = os.path.join(save_path, f'expose_params.npz')
        out_params = dict(fname='target')
        for key, val in stage_n_out.items():
            if key == 'body_pose' or key == 'left_hand_pose' or \
                    key == 'right_hand_pose' or key == 'jaw_pose' or key == 'global_orient':
                rodg = rotmat2vec(val.detach().cpu().numpy())
                val = torch.Tensor(rodg)
            if torch.is_tensor(val):
                val = val.detach().cpu().numpy()[0]
            out_params[key] = val

        for key, val in hd_params.items():
            if torch.is_tensor(val):
                val = val.detach().cpu().numpy()
            if np.isscalar(val[0]):
                out_params[key] = val[0].item()
            else:
                out_params[key] = val[0]

    # skinning_mesh.vertices -= skinning_mesh.bounding_box.centroid
    # skinning_mesh.vertices *= 2 / np.max(skinning_mesh.bounding_box.extents)

    # print(gt_model.bounding_box.centroid)
    # print(2 / np.max(gt_model.bounding_box.extents))

    # gt_model.vertices -= gt_model.bounding_box.centroid
    # gt_model.vertices *= 2 / np.max(gt_model.bounding_box.extents)

    return skinning_mesh, out_params

if __name__ == '__main__':
    voxel_resolution = 512
    width = 256
    height = 256
    fov = 60
    cam_res = 256

    RGB_MAX = [255.0, 255.0, 255.0]
    DEPTH_MAX = 255.0

    mesh = trimesh.load ('./15500_AAM_M/pred_mesh/result_19.obj')
    color_front = cv2.imread ('./15500_AAM_M/pred_color/color_19_front.png', cv2.IMREAD_COLOR)
    color_back = cv2.imread ('./15500_AAM_M/pred_color/color_19_back.png', cv2.IMREAD_COLOR)
    depth_front = cv2.imread ('./15500_AAM_M/pred_depth/depth_19_front.png', cv2.IMREAD_ANYDEPTH)