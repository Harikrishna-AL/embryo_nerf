from nano_nerf.utils import (
    compute_rays,
    compute_query_points,
    pos_encoding,
    get_batches,
    render_volume,
)
import torch


def train_iter(
    height, width, focal_len, cam2world, near, far, samples, encoding_fun, model
):
    ray_directions, ray_origins = compute_rays(height, width, focal_len, cam2world)
    query_points, depth_values = compute_query_points(
        ray_origins, ray_directions, near, far, samples
    )
    query_points_flattened = query_points.reshape((-1, 3))
    encoded_points = encoding_fun(query_points_flattened)
    batches = get_batches(encoded_points)
    predictions = []
    for batch in batches:
        predictions.append(model(batch))
    radiance_field = torch.cat(predictions, dim=0)
    shape = list(query_points.shape[:-1]) + [4]
    radiance_field = torch.reshape(radiance_field, shape)
    rgb_pred, _, _ = render_volume(radiance_field, ray_origins, depth_values)

    return rgb_pred
