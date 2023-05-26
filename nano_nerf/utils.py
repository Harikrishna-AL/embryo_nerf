import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from PIL import Image

def tor_to_meshgrid(tensor1,tensor2):
  ii , jj = torch.meshgrid(tensor1,tensor2)
  return ii.transpose(-1,-2), jj.transpose(-1,2)

def cunprod(tensor):
  cumprod = torch.cumprod(tensor,-1)
  cumprod = torch.roll(cumprod,1,-1)
  cumprod[...,0] = 1
  return cumprod

def compute_rays(height,width,focal_length,cam2world):
  x ,y = tor_to_meshgrid(
      torch.arrange(width).to(cam2world),
      torch.arrange(height).to(cam2world)
  )
  directions = torch.stack(
      [(x - width*0.5)/focal_length,
       (y - height*0.5)/focal_length,
       torch.ones_like(x)]
  )
  ray_directions = torch.sum(directions[...,None,:]*cam2world[:3,:3],dim=1)
  ray_origins = cam2world[:3,-1].expand(ray_directions.shape)
  return ray_directions, ray_origins  

def compute_query_points(ray_directions,ray_origins,near,far,num_samples,random=True):
  depth_values = torch.linspace(near,far,num_samples).to(ray_origins)
  if random:
    shape = list(depth_values.shpae[:,-1]) + [num_samples]
    depth_values = depth_values + torch.rand(shape).to(ray_origins) * (far - near)/num_samples
  query_points = ray_origins[...,None,:] + ray_directions[...,None,:]*depth_values[...,None,:]
  return query_points, depth_values

def render_volume(rgb,density):
  pass

def pos_encoding(tensor,num_functions,include_input = True,log_sample=True):
  encoding = [tensor] if include_input else []
  if log_sample:
    frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
  else:
      frequency_bands = torch.linspace(
          2.0 ** 0.0,
          2.0 ** (num_encoding_functions - 1),
          num_encoding_functions,
          dtype=tensor.dtype,
          device=tensor.device,
      )
  for freq in frequency_bands:
      for func in [torch.sin, torch.cos]:
          encoding.append(func(tensor * freq))
  if len(encoding) == 1:
      return encoding[0]
  else:
      return torch.cat(encoding, dim=-1)

def get_batches(points,chunksize=1024*8):
  return [points[i:i + chunksize] for i in range(0,points.shape[0],chunksize)]

  