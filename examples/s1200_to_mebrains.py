from neuromaps_nhp.config import config
from neuromaps_nhp.resources.fetch_resource import fetch_resource, fetch_atlas, fetch_transform, search_resources
from neuromaps_nhp.graph.transform_graph import BrainAtlasTransformGraph
from neuromaps_nhp.utils.niwrap_wrappers import surface_sphere_project_unproject, metric_resample, surface_resample
from neuromaps_nhp.utils.gifti_utils import get_density
from pathlib import Path
from niwrap import workbench


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

sphere_in = fetch_resource(resource_type="transform", source=path[0].lower(), target=path[1].lower(), density=density, hemisphere=hemisphere, resource_name="sphere")
#  s1200- yerkes19, 32

sphere_project_to = fetch_resource(resource_type="atlas", source=path[1].lower(), hemisphere=hemisphere, density=density, resource_name="sphere")
 # yerkes19, 32

sphere_unproject_from = fetch_resource(resource_type="transform", source=path[1].lower(), target=path[2].lower(), density=density, hemisphere=hemisphere, resource_name="sphere")
 # yerkes19 - mebrains, 32


output_surface = config.data_dir / f"out_{source}_to_{target}_den-{density}_hemi-{hemisphere}.surf.gii"

print(f" \nsphere_in: {sphere_in}, \nsphere_project_to: {sphere_project_to}, \nsphere_unproject_from: {sphere_unproject_from}, \noutput_surface: {output_surface}")

resulting_transform = surface_sphere_project_unproject(
    sphere_in=sphere_in,
    sphere_project_to=sphere_project_to,
    sphere_unproject_from=sphere_unproject_from,
    sphere_out=str(output_surface)
)

#Apply the resulting transform to the input data
target_surface = search_resources(resource_type="atlas", source=path[2].lower(), hemisphere=hemisphere, resource_name="midthickness")
target_density = get_density(target_surface[0].filepath)

new_sphere = fetch_resource(resource_type="atlas", source=path[2].lower(), hemisphere=hemisphere, density=target_density, resource_name="sphere")

current_area = fetch_resource(resource_type="atlas", source=path[0].lower(), hemisphere=hemisphere, density=density, resource_name="midthickness")

new_area = fetch_resource(resource_type="atlas", source=path[2].lower(), hemisphere=hemisphere, density=target_density, resource_name="midthickness")

area_surfs = workbench.metric_resample_area_surfs_params(current_area=current_area, new_area=new_area)

print(f"\nnew_sphere: {new_sphere}, \ncurrent_area: {current_area}, \nnew_area: {new_area}, \narea_surfs: {area_surfs}")

result = workbench.metric_resample(
    metric_in=input,
    current_sphere=resulting_transform,
    new_sphere=new_sphere,
    metric_out=str(config.data_dir / f"out_{source}_to_{target}_den-{density}_hemi-{hemisphere}_data.func.gii"),
    method="ADAP_BARY_AREA",
    area_surfs= area_surfs
)
print(f"Data resampled and saved at: {result}")

