import pandas as pd
from pathlib import Path
from typing import List, Optional
from neuromaps_nhp.config import config

class Resource:
    """A class to represent a brain atlas or transform resource."""
    
    def __init__(self, **kwargs):
        """Initialize resource from CSV row data."""
        self.resource_name = kwargs.get('resource_name')
        self.filename = kwargs.get('filename')
        self.filepath = Path(kwargs.get('filepath'))
        self.source = kwargs.get('source')
        
        # Handle NaN values for target
        target_val = kwargs.get('target')
        self.target = None if pd.isna(target_val) else target_val
        
        self.density = kwargs.get('density')
        self.hemisphere = kwargs.get('hemisphere')
        self.description = kwargs.get('description')
        self.file_type = kwargs.get('file_type')
        
        # Set resource_type based on file_type or target presence
        if self.file_type == 'atlas' or self.target is None:
            self.resource_type = 'atlas'
        else:
            self.resource_type = 'transform'
    
    @property
    def is_atlas(self) -> bool:
        """Check if this resource is an atlas."""
        return self.resource_type == 'atlas'
    
    @property
    def is_transform(self) -> bool:
        """Check if this resource is a transform."""
        return self.resource_type == 'transform'
    
    def __repr__(self):
        return f"resource_type: {self.resource_type}, resource_name: {self.resource_name}, source: {self.source}, target: {self.target}, density: {self.density}, hemisphere: {self.hemisphere})"



class ResourceManager:
    """Manager class for searching and fetching brain atlas resources."""
    
    def __init__(self):
        """Initialize ResourceManager with data from CSV files."""
        self.resources_dir = Path(__file__).parent
        self.atlas_df = pd.read_csv(self.resources_dir / "atlas_paths.csv")
        self.transforms_df = pd.read_csv(self.resources_dir / "transforms_paths.csv")
        
        # Combine both dataframes for unified searching
        self.all_resources_df = pd.concat([self.atlas_df, self.transforms_df], ignore_index=True)
    
    def search(self, 
               resource_type: Optional[str] = None,
               source: Optional[str] = None, 
               target: Optional[str] = None, 
               density: Optional[str] = None, 
               hemisphere: Optional[str] = None, 
               resource_name: Optional[str] = None) -> List[Resource]:
        """
        Search for resources based on criteria.
        
        Parameters
        ----------
        resource_type : str, optional
            Type of resource: 'atlas' or 'transform'
        source : str, optional
            Source atlas name
        target : str, optional
            Target atlas name (only for transforms)
        density : str, optional
            Density specification (e.g., '32k', '41k')
        hemisphere : str, optional
            Hemisphere: 'left' or 'right'
        resource_name : str, optional
            Resource name (e.g., 'sphere', 'midthickness')
            
        Returns
        -------
        List[Resource]
            List of matching resources
        """
        df = self.all_resources_df.copy()
        
        # Apply filters
        if resource_type is not None:
            df = df[df['file_type'] == resource_type]
        
        if source is not None:
            # Case-insensitive matching
            df = df[df['source'].str.lower() == source.lower()]
        
        if target is not None:
            # For atlas resources, target should be None/empty
            if resource_type == 'atlas':
                df = df[df['target'].isna() | (df['target'] == '')]
            else:
                df = df[df['target'].str.lower() == target.lower()]
        
        if density is not None:
            df = df[df['density'] == density]
        
        if hemisphere is not None:
            df = df[df['hemisphere'] == hemisphere]
        
        if resource_name is not None:
            df = df[df['resource_name'] == resource_name]
        
        # Convert to Resource objects
        resources = []
        for _, row in df.iterrows():
            resources.append(Resource(**row.to_dict()))
        
        return resources
    
    def get_filepath(self, 
                     resource_type: str,
                     source: Path, 
                     target: Path = None, 
                     density: str = None, 
                     hemisphere: str = None, 
                     resource_name: str = None) -> Optional[Path]:
        """
        Get the filepath for a specific resource.
        
        Returns the first matching filepath or None if not found.
        """
        resources = self.search(
            resource_type=resource_type,
            source=source,
            target=target,
            density=density,
            hemisphere=hemisphere,
            resource_name=resource_name
        )
        
        if resources:
            return resources[0].filepath
        return None

# Create a global instance
resource_manager = ResourceManager()

def fetch_resource(resource_type: str, 
                   source: Path, 
                   target: Path = None, 
                   density: str = None, 
                   hemisphere: str = None, 
                   resource_name: str = None) -> Optional[Path]:
    """
    Fetch a resource filepath based on search criteria.
    
    Parameters
    ----------
    resource_type : str
        Type of resource: 'atlas' or 'transform'
    source : Path
        Source atlas name
    target : Path, optional
        Target atlas name (only for transforms)
    density : str, optional
        Density specification (e.g., '32k', '41k')
    hemisphere : str, optional
        Hemisphere: 'left' or 'right'
    resource_name : str, optional
        Resource name (e.g., 'sphere', 'midthickness')
        
    Returns
    -------
    Path or None
        Filepath to the resource, or None if not found
        
    Examples
    --------
    >>> # Get a Yerkes19 sphere for left hemisphere at 32k density
    >>> fetch_resource('atlas', 'Yerkes19', density='32k', hemisphere='left', resource_name='sphere')
    '/path/to/src-Yerkes19_den-32k_hemi-L_sphere.surf.gii'
    
    >>> # Get a transform from Yerkes19 to D99
    >>> fetch_resource('transform', 'Yerkes19', 'D99', density='32k', hemisphere='left', resource_name='sphere')
    '/path/to/src-Yerkes19_to-D99_den-32k_hemi-L_sphere.surf.gii'
    """
    return resource_manager.get_filepath(
        resource_type=resource_type,
        source=source,
        target=target,
        density=density,
        hemisphere=hemisphere,
        resource_name=resource_name
    )

def fetch_atlas(resource_name: str, source: Path, density: Path, hemisphere: str) -> Optional[Path]:
    """
    Fetch atlas resource filepath.
    
    Parameters
    ----------
    resource_name : str
        Resource name (e.g., 'sphere', 'midthickness')
    source : Path  
        Source atlas name
    density : Path
        Density specification (e.g., '32k', '41k')
    hemisphere : str
        Hemisphere: 'left' or 'right'
        
    Returns
    -------
    Path or None
        Filepath to the atlas resource
    """
    return fetch_resource('atlas', source, None, density, hemisphere, resource_name)

def fetch_transform(resource_name: str, source: Path, target: Path, density: str, hemisphere: str) -> Optional[Path]:
    """
    Fetch transform resource filepath.
    
    Parameters
    ----------
    resource_name : str
        Resource name (e.g., 'sphere', 'midthickness')
    source : Path
        Source atlas name
    target : Path
        Target atlas name
    density : str
        Density specification (e.g., '32k', '41k')
    hemisphere : str
        Hemisphere: 'left' or 'right'
        
    Returns
    -------
    Path or None
        Filepath to the transform resource
    """
    return fetch_resource('transform', source, target, density, hemisphere, resource_name)

def search_resources(**kwargs) -> List[Resource]:
    """
    Search for resources with flexible criteria.
    
    Parameters
    ----------
    **kwargs
        Any combination of: resource_type, source, target, density, hemisphere, resource_name
        
    Returns
    -------
    List[Resource]
        List of matching resources
        
    Examples
    --------
    >>> # Find all Yerkes19 resources
    >>> search_resources(source='Yerkes19')
    
    >>> # Find all left hemisphere spheres
    >>> search_resources(hemisphere='left', resource_name='sphere')
    """
    return resource_manager.search(**kwargs)