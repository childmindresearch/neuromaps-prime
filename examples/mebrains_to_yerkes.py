from neuromaps_nhp.config import config
from neuromaps_nhp.resources.fetch_resource import fetch_resource, fetch_atlas, fetch_transform, search_resources
from neuromaps_nhp.graph.transform_graph import BrainAtlasTransformGraph
from neuromaps_nhp.utils.niwrap_wrappers import surface_sphere_project_unproject, metric_resample, surface_resample

from pathlib import Path


graph = BrainAtlasTransformGraph()

input = Path("/home/bshrestha/projects/Tfunck/testdata/mebrains_oxot.func.gii")

source = "mebrains"
target = "yerkes19"
density = "101k"
hemisphere = "left"

# Find the shortest path
path = graph.find_shortest_path(source, target)
print(f"Shortest path from {source} to {target}: {path}")

current_sphere_available = search_resources(resource_type="atlas", source=path[0].lower(), hemisphere=hemisphere, density=density, resource_name="sphere")
current_sphere = current_sphere_available[0].filepath

new_sphere_available = search_resources(resource_type="transform", source=path[0].lower(), target=path[1].lower(), density=density, hemisphere=hemisphere, resource_name="sphere")
new_sphere = new_sphere_available[0].filepath
target_density = new_sphere_available[0].density

print(f"Available {len(current_sphere_available)} current_sphere resources: {current_sphere_available}")
print(f"Available {len(new_sphere_available)} new_sphere resources: {new_sphere_available}")



#Apply the resulting transform to the input data
result = metric_resample(
    metric_in=str(input.resolve()),
    current_sphere=str(current_sphere.resolve()),
    new_sphere=str(new_sphere.resolve()),
    metric_out=str(config.data_dir / f"out_{source}_to_{target}_den-{target_density}_hemi-{hemisphere}_data.func.gii"),
    method="BARYCENTRIC"
)
print(f"Data resampled and saved at: {result}")

