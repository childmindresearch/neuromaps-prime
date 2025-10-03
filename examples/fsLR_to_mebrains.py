from neuromaps_nhp.config import config
from neuromaps_nhp.resources.fetch_resource import fetch_resource, fetch_atlas, fetch_transform, search_resources
from neuromaps_nhp.graph.transform_graph import BrainAtlasTransformGraph
from neuromaps_nhp.utils.niwrap_wrappers import surface_sphere_project_unproject, metric_resample

from neuromaps.datasets import fetch_all_atlases
from pathlib import Path


graph = BrainAtlasTransformGraph()

fetch_all_atlases()

input = Path("/home/bshrestha/projects/niwrap/outputs/output/pipeline_cpac_abcd-options/sub-NDARINV003RTV85/ses-baselineYear1Arm1/func/sub-NDARINV003RTV85_ses-baselineYear1Arm1_task-rest_run-01_hemi-R_space-fsLR_den-32k_desc-atlasroi_bold.func.gii")

source = "s1200"
target = "mebrains"
density = "32k"
hemisphere = "left"

# Find the shortest path
path = graph.find_shortest_path(source, target)
print(f"Shortest path from {source} to {target}: {path}")
# output : ['s1200', 'yerkes19', 'mebrains']

# find atlas
source = path[0].lower()
target = path[1].lower()
resources_available = search_resources(resource_type="atlas", source=source, hemisphere=hemisphere, resource_name="sphere")
for resource in resources_available:
    if resource.density == density:
        input_surface = fetch_resource( resource_type="atlas", source=resource.source, target=None, density=resource.density, hemisphere=resource.hemisphere, resource_name="sphere")
        print(f"Input surface found at: {input_surface}")

# build transform path
source = path[0].lower()
target = path[1].lower()
resources_available = search_resources( source = source, target = target, hemisphere = hemisphere, resource_name = "sphere")
for resource in resources_available:
    if resource.density == density:
        surface_project_to = fetch_resource( resource_type="transform", source=resource.source, target=resource.target, density=resource.density, hemisphere=resource.hemisphere, resource_name="sphere")
        print(f"Surface project to found at: {surface_project_to}")

source = path[1].lower()
target = path[2].lower()
resources_available = search_resources( source = source, target = target, hemisphere = hemisphere, resource_name = "sphere")
for resource in resources_available:
    if resource.density == density:
        surface_unproject_from = fetch_resource( resource_type="transform", source=resource.source, target=resource.target, density=resource.density, hemisphere=resource.hemisphere, resource_name="sphere")
        print(f"Surface unproject from found at: {surface_unproject_from}"
        )
output_surface = config.data_dir / f"out_fsLR_to_mebrains_den-{density}_hemi-{hemisphere}.surf.gii"


resulting_transform = surface_sphere_project_unproject(
    sphere_in=str(input_surface.resolve()),
    sphere_project_to=str(surface_project_to.resolve()),
    sphere_unproject_from=str(surface_unproject_from.resolve()),
    sphere_out=str(output_surface.resolve())
)

#Apply the resulting transform to the input data
result = metric_resample(
    metric_in=str(input.resolve()),
    current_sphere=str(input_surface.resolve()),
    new_sphere=str(output_surface.resolve()),
    metric_out=str(config.data_dir / f"out_fsLR_to_mebrains_den-{density}_hemi-{hemisphere}_data.func.gii"),
    method="BARYCENTRIC"
)
print(f"Data resampled and saved at: {result}")



import nibabel as nib
import matplotlib.pyplot as plt

# Load metric
metric = nib.load(result)
data = metric.darrays[0].data  # shape: (n_vertices, )

# Simple histogram
plt.hist(data, bins=50)
plt.savefig(config.data_dir / "metric_histogram.png")