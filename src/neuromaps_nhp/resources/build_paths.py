from pathlib import Path
import pandas as pd
import re

class Paths:
    
    def __init__(self, base_share: Path = None):
        if base_share is None:
            base_share = Path("/home/bshrestha/projects/Tfunck/neuromaps-nhp-prep/share").resolve()
        self.base_share = base_share
        self.resources_dir = Path(__file__).parent.resolve()

        self.atlas_df = self.create_atlas_info_csv()
        self.transforms_df = self.create_transforms_info_csv()

    def get_atlas_paths(self) -> list[Path]:
        """Get all atlas files from Inputs directory with folder structure source/files"""
        inputs_dir = self.base_share / "Inputs"
        atlas_paths = []
        
        # Look for folders in Inputs (these are sources)
        for source_folder in inputs_dir.iterdir():
            if source_folder.is_dir():
                # Get all .surf.gii files in this source folder
                atlas_paths.extend(list(source_folder.glob("*.surf.gii")))
        
        return atlas_paths
    
    def get_transforms_paths(self) -> list[Path]:
        """Get all transform files from Outputs directory with folder structure target-source/"""
        outputs_dir = self.base_share / "Outputs"
        transforms_paths = []
        
        # Look for folders with target-source pattern
        for folder in outputs_dir.glob("*-*"):
            if folder.is_dir():
                # Get all .surf.gii files in this folder
                transforms_paths.extend(list(folder.glob("*.surf.gii")))
        
        return transforms_paths
    
    def parse_filename_info(self, filepath: Path, file_type: str = "atlas") -> dict:
        """
        Parse filename to extract source, density, hemisphere, etc.
        file_type can be 'atlas' or 'transform'
        """
        filename = filepath.stem  # Remove .surf.gii extension
        
        # Initialize default values
        info = {
            'filename': filepath.name,
            'filepath': str(filepath),
            'source': None,
            'target': None,
            'density': None,
            'hemisphere': None,
            'description': None,
            'file_type': file_type
        }
        
        if file_type == "transform":
            # Parse transform files like: src-D99_to-CIVETNMT_den-41k_hemi-L_midthickness.surf.gii
            # Also extract target and source from folder structure
            folder_name = filepath.parent.name
            if "-" in folder_name:
                target_source = folder_name.split("-", 1)
                if len(target_source) == 2:
                    info['target'] = target_source[0]
                    info['source'] = target_source[1]
            
            # Extract source from filename (src-XXX)
            src_match = re.search(r'src-([A-Za-z0-9]+)', filename)
            if src_match:
                info['source'] = src_match.group(1)
            
            # Extract target from filename (to-XXX)
            target_match = re.search(r'to-([A-Za-z0-9]+)', filename)
            if target_match:
                info['target'] = target_match.group(1)
            
            # Extract density (den-XXX)
            den_match = re.search(r'den-([0-9]+k?)', filename)
            if den_match:
                info['density'] = den_match.group(1)
            
            # Extract hemisphere (hemi-X)
            hemi_match = re.search(r'hemi-([LR])', filename)
            if hemi_match:
                hemi = hemi_match.group(1)
                info['hemisphere'] = 'left' if hemi == 'L' else 'right'
            
            # Extract description (everything after hemi)
            desc_match = re.search(r'hemi-[LR]_(.+)', filename)
            if desc_match:
                info['description'] = desc_match.group(1)
        
        else:  # atlas files
            # Extract source from folder structure for atlas files
            info['source'] = filepath.parent.name
            
            # Extract hemisphere with multiple patterns for atlas files
            # Pattern 1: With separators
            hemi_match = re.search(r'[._-](L|R|left|right|lh|rh)[._-]', filename, re.IGNORECASE)
            if not hemi_match:
                # Pattern 2: At beginning or end
                hemi_match = re.search(r'^(L|R|left|right|lh|rh)[._-]|[._-](L|R|left|right|lh|rh)$', filename, re.IGNORECASE)
            if not hemi_match:
                # Pattern 3: Simple match anywhere
                hemi_match = re.search(r'(L|R|left|right|lh|rh)', filename, re.IGNORECASE)
            
            if hemi_match:
                # Get the matched group (could be group 1 or 2 depending on pattern)
                hemi = hemi_match.group(1) if hemi_match.group(1) else hemi_match.group(2)
                hemi = hemi.lower()
                if hemi in ['l', 'left', 'lh']:
                    info['hemisphere'] = 'left'
                elif hemi in ['r', 'right', 'rh']:
                    info['hemisphere'] = 'right'
            
            density_match = re.search(r'(\d+)k', filename, re.IGNORECASE)
            if density_match:
                info['density'] = f"{density_match.group(1)}k"
        
        return info
    
    def create_atlas_info_csv(self, output_path: Path = None) -> pd.DataFrame:
        """Create a CSV file with parsed information from atlas filenames."""
        if output_path is None:
            output_path = self.resources_dir / "atlas_paths.csv"
        
        atlas_paths = self.get_atlas_paths()
        
        # Parse each filename
        atlas_info = []
        for path in atlas_paths:
            info = self.parse_filename_info(path, file_type="atlas")
            atlas_info.append(info)
        
        # Create DataFrame
        df = pd.DataFrame(atlas_info)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Atlas information saved to: {output_path}")
        
        return df
    
    def create_transforms_info_csv(self, output_path: Path = None) -> pd.DataFrame:
        """Create a CSV file with parsed information from transform filenames."""
        if output_path is None:
            output_path = self.resources_dir / "transforms_paths.csv"
        
        transforms_paths = self.get_transforms_paths()
        
        # Parse each filename
        transforms_info = []
        for path in transforms_paths:
            info = self.parse_filename_info(path, file_type="transform")
            transforms_info.append(info)
        
        # Create DataFrame
        df = pd.DataFrame(transforms_info)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Transforms information saved to: {output_path}")
        
        return df
    
