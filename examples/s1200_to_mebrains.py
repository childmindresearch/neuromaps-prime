from neuromaps_nhp.config import config
from neuromaps_nhp.resources.fetch_resource import fetch_resource, fetch_atlas, fetch_transform, search_resources
from neuromaps_nhp.graph.transform_graph import BrainAtlasTransformGraph
from neuromaps_nhp.utils.niwrap_wrappers import surface_sphere_project_unproject, metric_resample, surface_resample

from pathlib import Path


graph = BrainAtlasTransformGraph()

input = Path("/home/bshrestha/projects/niwrap/outputs/output/pipeline_cpac_abcd-options/sub-NDARINV003RTV85/ses-baselineYear1Arm1/func/sub-NDARINV003RTV85_ses-baselineYear1Arm1_task-rest_run-01_hemi-R_space-fsLR_den-32k_desc-atlasroi_bold.func.gii")
# mid thickness works too

source = "s1200"
target = "mebrains"
density = "32k"
hemisphere = "left"

# Find the shortest path
path = graph.find_shortest_path(source, target)
print(f"Shortest path from {source} to {target}: {path}")
# s1200 - yerkes19 - mebrains

sphere_in_available = search_resources(resource_type="transform", source=path[0].lower(), target=path[1].lower(), density=density, hemisphere=hemisphere, resource_name="sphere")
sphere_in = sphere_in_available[0].filepath #  s1200- yerkes19, 32

sphere_project_to_available = search_resources(resource_type="atlas", source=path[2].lower(), hemisphere=hemisphere, density="101k", resource_name="sphere")
sphere_project_to = sphere_project_to_available[0].filepath # yerkes19, 32

sphere_unproject_from_available = search_resources(resource_type="transform", source=path[2].lower(), target=path[1].lower(), density="101k", hemisphere=hemisphere, resource_name="sphere")
sphere_unproject_from = sphere_unproject_from_available[0].filepath # yerkes19 - mebrains, 32
target_density = sphere_unproject_from_available[0].density

print(f"Available {len(sphere_in_available)} sphere_in resources: {sphere_in_available}")
print(f"Available {len(sphere_project_to_available)} sphere_project_to resources: {sphere_project_to_available}")
print(f"Available {len(sphere_unproject_from_available)} sphere_unproject_from resources: {sphere_unproject_from_available}")

output_surface = config.data_dir / f"out_{source}_to_{target}_den-{target_density}_hemi-{hemisphere}.surf.gii"


resulting_transform = surface_sphere_project_unproject(
    sphere_in=str(sphere_in.resolve()),
    sphere_project_to=str(sphere_project_to.resolve()),
    sphere_unproject_from=str(sphere_unproject_from.resolve()),
    sphere_out=str(output_surface.resolve())
)

#Apply the resulting transform to the input data
result = metric_resample(
    metric_in=str(input.resolve()),
    current_sphere=str(sphere_in.resolve()),
    new_sphere=str(output_surface.resolve()),
    metric_out=str(config.data_dir / f"out_{source}_to_{target}_den-{target_density}_hemi-{hemisphere}_data.func.gii"),
    method="BARYCENTRIC"
)
print(f"Data resampled and saved at: {result}")

