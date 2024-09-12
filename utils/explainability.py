import numpy as np
import torch
import cv2
from typing import Callable, List, Tuple, Optional
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
# from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class GradCAM(BaseCAM): # Overwrite official GradCAM to apply to video
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.mean(grads, axis=(3, 4))
    
    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads)
        
        weighted_activations = weights[:, :, :, None, None] * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> np.ndarray:
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_depth_height_width(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            cam = np.maximum(cam, 0)
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer
    
    def get_target_depth_height_width(self, input_tensor: torch.Tensor) -> Tuple[int, int, int]:
        depth, height, width = input_tensor.size(-3), input_tensor.size(-2), input_tensor.size(-1)
        return depth, height, width
    
    def scale_cam_image(self, cam, target_size=None):
        result = []
        print(target_size)
        for clip in cam:
            key_frames = []
            for img in clip:
                img = img - np.min(img)
                img = img / (1e-7 + np.max(img))
                if target_size is not None:
                    img = cv2.resize(img, target_size[1:3])
                key_frames.append(img)
            key_frames = np.float32(key_frames)
            if key_frames.shape[0]==1:
                key_frames = np.repeat(key_frames, target_size[0], axis=0)
            result.append(key_frames)
        result = np.float32(result)

        return result