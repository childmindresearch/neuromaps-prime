# neuromaps-PRIME Annotation Files

This document describes the annotation files and the naming convention used to identify them across the dataset.

---

## Naming Convention

Filenames follow the BIDS specification as closely as possible.

### Surface Files
```text
src-{space}_den-{den}k_hemi-{hemi}_{subLabelCode}-{subLabel}_desc-{annotationCode}_annot.{extension}
```

### Volume Files
```text
src-{space}_res-{res}mm_desc-{annotationCode}_annot.{extension}
```

> **Note:** Volume files use a voxel resolution (`res-{res}mm`) instead of surface mesh density, and may use BIDS-defined suffixes where clearly defined instead of `desc-{code}_annot`.

### Field Breakdown

| Field | Meaning |
| :--- | :--- |
| **space** | The reference/coordinate space the data is aligned to (e.g., `Yerkes19`, `fsLR`, `fsaverage`). |
| **den** | Surface mesh density in thousands of vertices (e.g., `10k`, `32k`, `164k`) — *surface files only*. |
| **res** | Voxel resolution in mm (e.g., `0p40mm` = 0.4mm) — *volume files only*. |
| **hemi** | Hemisphere (`L` = left, `R` = right). |
| **subLabelCode-subLabel** | A short code identifying the specific source/variant of the data (e.g., dataset, parcellation name, atlas name), followed by its descriptive name. See **Sub-Label Naming Resources** below. |
| **annotationCode** | A short code identifying which type of annotation the file contains (see table below). |
| **extension** | File format (e.g., `.shape.gii`, `.label.gii`, `.nii.gz`, `.txt`). |

---

## Annotation Type Codes

These codes appear after `desc-` in the filename and indicate the kind of data stored in the file. 

| Code | Annotation Type | Description |
| :--- | :--- | :--- |
| **AE** | Area Expansion | Relative surface area expansion. |
| **BM** | Brain Masks | Binary or probabilistic maps defining which surface vertices or voxels correspond to valid cortical tissue (used for masking and analysis inclusion; exception uses `mask` suffix). |
| **CH** | Cortical Hierarchy | Anatomical cortical hierarchy estimates (e.g., hierarchy indices complementing intrinsic timescales). |
| **CT** | Cortical Thickness | Vertex-wise estimates of the distance between the white matter and pial surfaces, reflecting local cortical thickness. |
| **CV** | Curvature | Vertex-wise measures of cortical surface geometry (e.g., mean or Gaussian curvature) indicating gyral and sulcal folding patterns. |
| **DM** | Dendritic Morphology | Microstructural neuronal features, such as layer III pyramidal cell dendritic tree size or spine counts. |
| **FG** | Functional Connectivity Gradients | Spatial gradients representing low-dimensional embeddings of functional connectivity profiles (often stored as multi-maps). |
| **FHI** | Functional Homology Index | Functional connectivity homology index representing cross-species similarity of FC gradient profiles (e.g., high in sensory, low in association cortex). |
| **HTR1A** | Gene Expression | Specific gene expression maps (e.g., serotonin 1A receptor gene expression mapped across species). |
| **IT** | Intrinsic Timescale | Regional estimates of neural temporal autocorrelation (e.g., decay constants from fMRI time series), reflecting how quickly activity fluctuates over time. |
| **MM** | Myelin Maps | Vertex-wise estimates of cortical myelin content, commonly derived from MRI contrasts such as T1w/T2w ratio and reflecting intracortical myelination. |
| **ND** | Neuron Density | Quantitative spatial distribution and estimates of neuronal density across cortical areas. |
| **PC** | Parcellations | Discrete atlas-based assignments mapping each surface vertex or voxel to a labeled cortical region, often used to define analysis units. |
| **RG** | Receptor Gradients | Low-dimensional spatial gradients derived from neurotransmitter receptor distribution profiles (e.g., principal receptor gradients across non-human primate cortex). |
| **RM** | Receptor Maps | Spatial maps of neurotransmitter receptor density or binding potential, derived from PET or autoradiography imaging (e.g., radioligand binding densities per-neuron or raw). |
| **SD** | Sulcal Depth | Vertex-wise measure of the depth of cortical folds, defined as the distance between the cortical surface and a reference surface. |
| **SMM** | Smoothed Myelin Maps | Spatially smoothed versions of myelin maps (e.g., using surface-based kernels) to improve signal-to-noise ratio and emphasize large-scale gradients. |
| **TPL** | Templates | Reference anatomical volumes or surfaces defining a standard coordinate space to which other data are aligned (exception uses bare modality suffix, e.g., `T1w`). |
| **TPM** | Tissue Probability Maps | Voxel-wise probabilistic maps indicating the likelihood that a given location belongs to a specific tissue class (e.g., gray matter, white matter, CSF; exception uses `probseg` suffix). |

---

## Sub-Label Naming Resources

The `subLabelCode-subLabel` portion of the filename identifies the specific source of the data (e.g., which dataset, atlas, or parcellation it comes from). When naming or interpreting sub-labels, refer to the **BIDS Specification: Entities**.

### File Naming Examples

* **Surface File:**  
  `src-Yerkes19_den-10k_hemi-L_atlas-Yeo7Networks_desc-PC_annot.label.gii`
  * **Space:** Yerkes19
  * **Density:** 10k vertices
  * **Hemisphere:** Left
  * **Atlas:** Yeo7Networks
  * **Annotation Type:** Parcellation (`PC`)

* **Volume File:**  
  `src-NCBR_res-0p40mm_desc-T1w_mask.nii`
  * **Space:** NCBR
  * **Resolution:** 0.40mm resolution
  * **Annotation Type:** Brain Mask (`BM` exception)
