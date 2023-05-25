from utils import compute_rays, compute_query_points, encoding_fun
def train_iter(height,
               width,
               focal_len,
               cam2world,
               near,
               far,
               samples,
               encoding_fun,
               model):
  ray_directions, ray_origins = compute_rays(height,
  width,
  focal_len,
  cam2world)
  query_points, depth_values = compute_query_points(
      ray_origins,
      ray_directions,
      near,
      far,
      samples)
  query_points = query_points.reshape((-1,3))
  encoded_points = encoding_fun(query_points)