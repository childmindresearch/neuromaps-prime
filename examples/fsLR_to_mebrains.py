import pandas as pd
import os
import tempfile
import subprocess
from pathlib import Path
import nibabel as nib

from neuromaps_nhp.config import config
from neuromaps_nhp.resources.fetch_resource import fetch_resource, fetch_atlas, fetch_transform
from neuromaps_nhp.graph.transform_graph import BrainAtlasTransformGraph

transforms_df = config.transforms_df

# Example input sphere (you'll need to provide the actual path)
input_sphere = "/path/to/your/input_fsLR_sphere.surf.gii"  # Replace with actual path

# Output path
output_sphere = config.data_dir / "transformed_fsLR_to_MEBRAINS_sphere.surf.gii"

try:
    final_sphere, intermediates = transform_data_sphere(
        input_sphere_path=input_sphere,
        source_space=source,
        target_space=target,
        density=density,
        hemisphere=hemisphere,
        output_path=output_sphere
    )
    
    print(f"\n✓ Transformation complete!")
    print(f"Final transformed sphere: {final_sphere}")
    print(f"Intermediate spheres: {intermediates}")
    
except Exception as e:
    print(f"❌ Transformation failed: {e}")
    
    # Show available transforms for debugging
    print("\nAvailable transforms:")
    for (src, tgt), group in transforms_df.groupby(['source', 'target']):
        densities = group['density'].unique()
        print(f"  {src} -> {tgt}: {sorted(densities)}")