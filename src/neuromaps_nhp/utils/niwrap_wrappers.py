import os
from pathlib import Path
from niwrap import workbench as wb


def surface_sphere_project_unproject(sphere_in, sphere_project_to, sphere_unproject_from, sphere_out) -> Path:
    """Project and unproject a surface from one sphere to another.

    Parameters
    ----------
    sphere_in : str
        Path to input spherical surface.
    sphere_project_to : str
        Path to spherical surface to project to.
    sphere_unproject_from : str
        Path to spherical surface to unproject from.
    sphere_out : str
        Path to output spherical surface.
    """
    wb.surface_sphere_project_unproject(sphere_in, sphere_project_to, sphere_unproject_from, sphere_out)
    return Path(sphere_out)


def metric_resample(metric_in, current_sphere, new_sphere, method, metric_out,
                   area_surfs=None, area_metrics=None, current_roi=None, 
                   valid_roi_out=None, largest=False, bypass_sphere_check=False) -> Path:
    """Resample a metric file to a different mesh.

    Parameters
    ----------
    metric_in : Path
        The metric file to resample.
    current_sphere : Path
        A sphere surface with the mesh that the metric is currently on.
    new_sphere : Path
        A sphere surface that is in register with current_sphere and has 
        the desired output mesh.
    method : str
        The resampling method. Must be one of 'ADAP_BARY_AREA' or 'BARYCENTRIC'.
    metric_out : Path
        Path to the output metric file.
    area_surfs : tuple of (Path, Path), optional
        Tuple of (current_area, new_area) surface files for vertex area correction.
        current_area: relevant anatomical surface with current_sphere mesh
        new_area: relevant anatomical surface with new_sphere mesh
    area_metrics : tuple of (Path, Path), optional
        Tuple of (current_area, new_area) metric files with vertex areas.
        current_area: metric file with vertex areas for current_sphere mesh
        new_area: metric file with vertex areas for new_sphere mesh
    current_roi : Path, optional
        ROI metric file to exclude non-data vertices on current mesh.
    valid_roi_out : Path, optional
        Output path for ROI of vertices that got data from valid source vertices.
    largest : bool, optional
        Use only the value of the vertex with the largest weight. Default False.
    bypass_sphere_check : bool, optional
        ADVANCED: allow current and new 'spheres' to have arbitrary shape.
        Default False.

    Returns
    -------
    str
        Path to the output metric file.

    Notes
    -----
    - ADAP_BARY_AREA method is recommended for ordinary metric data
    - If ADAP_BARY_AREA is used, exactly one of area_surfs or area_metrics must be specified
    - The largest option results in nearest vertex behavior when used with BARYCENTRIC
    - When resampling binary metrics, consider thresholding at 0.5 after resampling
      rather than using largest=True

    Examples
    --------
    Basic resampling:
    >>> metric_resample('input.func.gii', 'sphere_32k.surf.gii', 'sphere_10k.surf.gii', 
    ...                'BARYCENTRIC', 'output.func.gii')

    With area correction using surfaces:
    >>> metric_resample('input.func.gii', 'sphere_32k.surf.gii', 'sphere_10k.surf.gii',
    ...                'ADAP_BARY_AREA', 'output.func.gii', 
    ...                area_surfs=('midthick_32k.surf.gii', 'midthick_10k.surf.gii'))

    With ROI masking:
    >>> metric_resample('input.func.gii', 'sphere_32k.surf.gii', 'sphere_10k.surf.gii',
    ...                'BARYCENTRIC', 'output.func.gii', 
    ...                current_roi='cortex_roi.func.gii')
    """
    
    # Validate method
    valid_methods = ['ADAP_BARY_AREA', 'BARYCENTRIC']
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}, got {method}")
    
    # Validate area correction options for ADAP_BARY_AREA
    if method == 'ADAP_BARY_AREA':
        area_options_count = sum([area_surfs is not None, area_metrics is not None])
        if area_options_count != 1:
            raise ValueError("For ADAP_BARY_AREA method, exactly one of area_surfs or area_metrics must be specified")
    
    # Build arguments list
    args = [metric_in, current_sphere, new_sphere, method, metric_out]
    
    # Add optional arguments
    if area_surfs is not None:
        current_area, new_area = area_surfs
        args.extend(['-area-surfs', current_area, new_area])
    
    if area_metrics is not None:
        current_area, new_area = area_metrics
        args.extend(['-area-metrics', current_area, new_area])
    
    if current_roi is not None:
        args.extend(['-current-roi', current_roi])
    
    if valid_roi_out is not None:
        args.extend(['-valid-roi-out', valid_roi_out])
    
    if largest:
        args.append('-largest')
    
    if bypass_sphere_check:
        args.append('-bypass-sphere-check')
    
    # Call the workbench function
    wb.metric_resample(*args)
    
    return metric_out


def surface_resample(surface_in, current_sphere, new_sphere, method, surface_out,
                    area_surfs=None, area_metrics=None, current_roi=None,
                    valid_roi_out=None, largest=False, bypass_sphere_check=False) -> Path:
    """Resample a surface file to a different mesh.
    
    This is a wrapper around wb_command -surface-resample with similar interface
    to metric_resample for consistency.

    Parameters
    ----------
    surface_in : Path
        The surface file to resample.
    current_sphere : Path
        A sphere surface with the mesh that the surface is currently on.
    new_sphere : Path
        A sphere surface that is in register with current_sphere and has
        the desired output mesh.
    method : str
        The resampling method. Must be one of 'ADAP_BARY_AREA' or 'BARYCENTRIC'.
    surface_out : Path
        Path to the output surface file.
    area_surfs : tuple of (Path, Path), optional
        Tuple of (current_area, new_area) surface files for vertex area correction.
    area_metrics : tuple of (Path, Path), optional
        Tuple of (current_area, new_area) metric files with vertex areas.
    current_roi : Path, optional
        ROI metric file to exclude non-data vertices on current mesh.
    valid_roi_out : Path, optional
        Output path for ROI of vertices that got data from valid source vertices.
    largest : bool, optional
        Use only the value of the vertex with the largest weight. Default False.
    bypass_sphere_check : bool, optional
        ADVANCED: allow current and new 'spheres' to have arbitrary shape.
        Default False.

    Returns
    -------
    Path
        Path to the output surface file.
    """
    
    # Validate method
    valid_methods = ['ADAP_BARY_AREA', 'BARYCENTRIC'] 
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}, got {method}")
    
    # Build arguments list
    args = [surface_in, current_sphere, new_sphere, method, surface_out]
    
    # Add optional arguments (same logic as metric_resample)
    if area_surfs is not None:
        current_area, new_area = area_surfs
        args.extend(['-area-surfs', current_area, new_area])
    
    if area_metrics is not None:
        current_area, new_area = area_metrics  
        args.extend(['-area-metrics', current_area, new_area])
    
    if current_roi is not None:
        args.extend(['-current-roi', current_roi])
    
    if valid_roi_out is not None:
        args.extend(['-valid-roi-out', valid_roi_out])
    
    if largest:
        args.append('-largest')
    
    if bypass_sphere_check:
        args.append('-bypass-sphere-check')
    
    # Call the workbench function
    wb.surface_resample(*args)

    return Path(surface_out)