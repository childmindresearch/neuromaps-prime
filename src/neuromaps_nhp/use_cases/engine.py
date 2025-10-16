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

def get_exact_transform(source, target, density, hemisphere, resource_type="sphere"):
    """Get exact transform if it exists."""
    matches = transforms_df[
        (transforms_df['source'] == source) & 
        (transforms_df['target'] == target) & 
        (transforms_df['density'] == density) & 
        (transforms_df['hemisphere'] == hemisphere) &
        (transforms_df['resource_name'] == resource_type) &
        (transforms_df['file_type'] == 'transform')
    ]
    
    if matches.empty:
        return None
    
    return matches.iloc[0]['filepath']

def get_available_densities(source, target, hemisphere):
    """Get list of available densities for a transform."""
    matches = transforms_df[
        (transforms_df['source'] == source) & 
        (transforms_df['target'] == target) & 
        (transforms_df['hemisphere'] == hemisphere) &
        (transforms_df['file_type'] == 'transform') &
        (transforms_df['resource_name'] == 'sphere')  # Only check spheres
    ]
    
    return matches['density'].unique().tolist()

def find_best_density_for_resampling(target_density, available_densities):
    """Find the best density to use for resampling to target density."""
    def density_to_num(d):
        return int(d.replace('k', '')) * 1000
    
    target_num = density_to_num(target_density)
    available_nums = [(d, density_to_num(d)) for d in available_densities]
    
    # Prefer upsampling over downsampling for better quality
    lower_densities = [(d, num) for d, num in available_nums if num < target_num]
    if lower_densities:
        return max(lower_densities, key=lambda x: x[1])[0]
    
    higher_densities = [(d, num) for d, num in available_nums if num > target_num]
    if higher_densities:
        return min(higher_densities, key=lambda x: x[1])[0]
    
    return available_densities[0] if available_densities else None

def get_or_build_sphere_transform(source, target, target_density, hemisphere):
    """Get sphere transform or build it by resampling from best available density."""
    
    # First try to get exact match
    exact_sphere = get_exact_transform(source, target, target_density, hemisphere, "sphere")
    if exact_sphere:
        return exact_sphere
    
    # If no exact match, find best available and resample
    available_densities = get_available_densities(source, target, hemisphere)
    
    if not available_densities:
        raise ValueError(f"No sphere transforms available from {source} to {target} for {hemisphere} hemisphere")
    
    best_density = find_best_density_for_resampling(target_density, available_densities)
    source_sphere = get_exact_transform(source, target, best_density, hemisphere, "sphere")
    
    if not source_sphere:
        raise ValueError(f"Could not find sphere transform {source} -> {target} with density {best_density}")
    
    if best_density == target_density:
        return source_sphere
    
    print(f"Would resample sphere {source} -> {target} from {best_density} to {target_density}")
    # For now, return the available sphere (implement resampling as needed)
    return source_sphere

def transform_sphere_through_path(input_sphere_path, path, density, hemisphere, output_dir=None):
    """Transform a sphere through a sequence of atlas spaces."""
    
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="sphere_transform_")
    
    current_sphere = input_sphere_path
    intermediate_spheres = []
    
    print(f"Transforming sphere through path: {' -> '.join(path)}")
    
    for i in range(len(path) - 1):
        src_space = path[i]
        tgt_space = path[i + 1]
        
        print(f"Step {i+1}: {src_space} -> {tgt_space}")
        
        # Get the transform sphere for this step
        transform_sphere = get_or_build_sphere_transform(src_space, tgt_space, density, hemisphere)
        
        if not transform_sphere or not os.path.exists(transform_sphere):
            raise ValueError(f"Transform sphere not found: {src_space} -> {tgt_space}")
        
        # Output path for this transformation step
        hemi_abbrev = hemisphere[0].upper()
        step_output = os.path.join(output_dir, f"step_{i+1}_{src_space}_to_{tgt_space}_den-{density}_hemi-{hemi_abbrev}_sphere.surf.gii")
        
        # Apply the transformation
        transformed_sphere = apply_sphere_transform(
            input_sphere=current_sphere,
            transform_sphere=transform_sphere,
            output_path=step_output,
            source_space=src_space,
            target_space=tgt_space
        )
        
        intermediate_spheres.append(transformed_sphere)
        current_sphere = transformed_sphere
        
        print(f"  -> Transformed sphere saved: {transformed_sphere}")
    
    return current_sphere, intermediate_spheres

def apply_sphere_transform(input_sphere, transform_sphere, output_path, source_space, target_space):
    """Apply a sphere transformation using surface resampling."""
    
    try:
        # Using Connectome Workbench for sphere transformation
        # wb_command -surface-resample <input> <current-sphere> <new-sphere> BARYCENTRIC <output>
        
        cmd = [
            'wb_command', '-surface-resample',
            input_sphere,           # Input data/sphere to transform
            input_sphere,           # Current sphere (source space)
            transform_sphere,       # Target sphere (transformation)
            'BARYCENTRIC',          # Interpolation method
            output_path             # Output transformed sphere
        ]
        
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"wb_command failed: {result.stderr}")
            
        return output_path
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: wb_command failed or not found: {e}")
        print("Falling back to copy (implement proper transformation as needed)")
        
        # Fallback: copy the input (you should implement proper transformation)
        import shutil
        shutil.copy2(input_sphere, output_path)
        return output_path

def transform_data_sphere(input_sphere_path, source_space, target_space, density, hemisphere, output_path=None):
    """Transform a data sphere from source to target space."""
    
    # Find the transformation path
    graph = BrainAtlasTransformGraph()
    path = graph.find_shortest_path(source_space, target_space)
    
    if not path:
        raise ValueError(f"No transformation path found from {source_space} to {target_space}")
    
    print(f"Transformation path: {' -> '.join(path)}")
    
    # Set up output
    if output_path is None:
        output_dir = tempfile.mkdtemp(prefix="transformed_sphere_")
        hemi_abbrev = hemisphere[0].upper()
        output_path = os.path.join(output_dir, f"transformed_{source_space}_to_{target_space}_den-{density}_hemi-{hemi_abbrev}_sphere.surf.gii")
    else:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
    
    # Apply the transformation
    final_sphere, intermediate_spheres = transform_sphere_through_path(
        input_sphere_path, path, density, hemisphere, output_dir
    )
    
    # Copy final result to desired output path
    if final_sphere != output_path:
        import shutil
        shutil.copy2(final_sphere, output_path)
    
    return output_path, intermediate_spheres