#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    depth_culling_mod : int

    
class GaussianRasterizerRenderCulling():
    def __init__(self, raster_settings):
        self.raster_settings = raster_settings


    def render(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, tile_depthmap = None):
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if tile_depthmap is None:
            tile_depthmap = torch.Tensor([])

        args = (
            self.raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            tile_depthmap,
            self.raster_settings.scale_modifier,
            cov3D_precomp,
            self.raster_settings.viewmatrix,
            self.raster_settings.projmatrix,
            self.raster_settings.tanfovx,
            self.raster_settings.tanfovy,
            self.raster_settings.image_height,
            self.raster_settings.image_width,
            shs,
            self.raster_settings.sh_degree,
            self.raster_settings.campos,
            self.raster_settings.prefiltered,
            self.raster_settings.depth_culling_mod
        )

        # Invoke C++/CUDA rasterizer
        num_rendered, time_ms, color, depth_map, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        return color, depth_map, time_ms, num_rendered
        



