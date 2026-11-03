# Adding Local or Private Data to the Graph

This guide shows how to describe a whole dataset in a graph YAML file, so it loads with the graph every time instead of being added to the graph step by step in code. The worked example is a small subset of the [BigMac](https://open.oxcin.ox.ac.uk/pages/amyh/bigmacdocumentation) macaque postmortem dataset.

This guide is about **how to add data**. The process is the same for volumetric and surface resources: describe the files under a node (`volumes` or `surfaces`), and describe transforms under an edge (`volume_to_volume` or `surface_to_surface`). The BigMac example only has volumes, so it uses the volume layout. Surface resources are keyed by density and hemisphere instead of resolution.

The YAML structure is defined by the JSON schemas in [`schemas/`](../../../schemas). Refer to them for the exact keys, allowed patterns and required fields:

| Schema | Describes |
| :--- | :--- |
| `neuromaps_nodes.schema.json` | Nodes (spaces) and their resources |
| `neuromaps_volumes.schema.json` | Volume resources, keyed by resolution |
| `neuromaps_surfaces.schema.json` | Surface resources, keyed by density, type and hemisphere |
| `neuromaps_volume_edges.schema.json` | Volume-to-volume transforms |
| `neuromaps_surface_edges.schema.json` | Surface-to-surface transforms |
| `neuromaps_references.schema.json`, `neuromaps_notes.schema.json` | `references` and `notes` fields |

Use this guide when your data is:

- **Local**: files already on disk, not downloadable from OSF, GitHub or GIN.
- **Private**: data that may require a data-use agreement, or data that has not yet been released. The data files stay on your machine; only the YAML describing them is shared.

For the annotation filename convention and the annotation type codes (`BM`, `TPM`, `PC`, ...), see [`README.md`](README.md).

---

## How the graph reads a YAML file

`NeuromapsGraph` builds from one YAML file with two top-level keys, `nodes` and `edges`:

```yaml
nodes:
  - <SpaceName>:        # one single-key mapping per node
      species: ...
      description: ...
      references: [...]
      surfaces: {...}
      volumes: {...}
edges:
  surface_to_surface: [...]
  volume_to_volume: [...]
```

Each `uri` can be a remote URL or a **local path**. When the path exists on disk, the graph uses it in place and never downloads or copies it.

> [!IMPORTANT]
> Local and private data are passed as the **full file path**. Relative paths resolve against the working directory, so they break when the graph is built from elsewhere.
>
> The YAML is fine to commit. The graph cannot use the node or its edges unless every referenced file exists at that path on the machine running it.

Passing `yaml_file=` to `NeuromapsGraph` replaces the bundled graph with the contents of your file. To add your dataset *to* the bundled graph, load the bundled graph first and then load your file with the builder:

```python
from pathlib import Path

from neuromaps_prime.graph import NeuromapsGraph

graph = NeuromapsGraph()
graph._builder.build_from_yaml(graph, Path("/data/bigmac/bigmac_graph.yaml"))
```

---

## Step 1: Decide what becomes what

Sort every file in the dataset into one of four groups.

| Dataset content | Graph concept | YAML location |
| :--- | :--- | :--- |
| Images defining a space (structural scans) | Volume atlas | `nodes` → `volumes` → `<res>` → `<type>` |
| Masks, tissue classes, parcellations in that space | Volume annotation | `nodes` → `volumes` → `<res>` → `annotation` |
| Warps from this space to another node | Volume transform | `edges` → `volume_to_volume` |
| Logs, scripts, READMEs, intermediate outputs | Not represented | skip |

Key naming rules enforced by the schemas:

- **Resolution key** matches `^[0-9]+(u|m)m$`, for example `250um` or `1mm`. Use `um` for sub-millimetre data (`0.6mm` is `600um`).
- **Resource type key** (such as `T1w`) uses only letters, digits and `_`. Hyphens are not allowed.
- **Annotation key** uses only letters, digits and `_`. Start it with an annotation type code from the [annotation type codes table](README.md#annotation-type-codes) (`BM`, `TPM`, `PC`, ...).

---

## Step 2: Map the BigMac tree

BigMac `Postmortem/struct/` contains two acquisitions (`bSSFP` and `MGE`). This guide maps only the MGE brain image, its brain mask and the MGE-to-F99 transforms into one node, `BigMac`.

| File | Graph entry | Name |
| :--- | :--- | :--- |
| `MGE/data/struct_brain.nii.gz` | volume atlas | `MGE` |
| `MGE/masks/brain_mask.nii.gz` | annotation (`BM`, [brain mask](README.md#annotation-type-codes)) | `BM` |
| `MGE/reg/struct_2_F99_fnirt/struct_2_F99_fnirt_field.nii.gz` | volume transform `BigMac` → `F99` | `fnirt_field` |
| `MGE/reg/struct_2_F99_flirt/struct_2_F99_flirt.mat` | volume transform `BigMac` → `F99` | `flirt_affine` |

Composite warps are preferred for volume transforms. The FSL field and affine above are kept as-is; see [Known limitations](#known-limitations).

Not added:

- The remaining `MGE/data` and `MGE/masks` files, and everything under `bSSFP/`. Add them the same way if you need them.
- `createwarp.sh*` (scripts and job logs) and `reg/README`.
- `struct_2_F99_*.nii.gz` warped images, `*_coeffs.nii.gz` and `*_jac.nii.gz`. These are outputs or intermediates, not transforms.
- `struct_2_dwi_*`. DWI space is not a node in the graph. Add a `BigMacDWI` node first if you need these.

---

## Step 3: Write the YAML

Save this as, for example, `/data/bigmac/bigmac_graph.yaml`. Replace `ROOT` with your absolute path to `Postmortem/struct` and `<RES>` with the real voxel size (see Step 1 for the key format).

```yaml
nodes:
  - BigMac:
      species: macaque
      description: BigMac postmortem macaque brain (ex vivo MRI)
      references:
        - citation: Howard AFD, et al. 2023. eLife.
          doi: 10.7554/eLife.84680
      volumes:
        <RES>:
          MGE: ROOT/MGE/data/struct_brain.nii.gz
          annotation:
            BM:
              uri: ROOT/MGE/masks/brain_mask.nii.gz
              notes:
                - Brain mask of the MGE acquisition.
edges:
  surface_to_surface: []
  volume_to_volume:
    - from: BigMac
      to: F99
      volumes:
        BigMac:                 # provider name
          <RES>:                # resolution key of the F99 reference grid
            fnirt_field: ROOT/MGE/reg/struct_2_F99_fnirt/struct_2_F99_fnirt_field.nii.gz
            flirt_affine: ROOT/MGE/reg/struct_2_F99_flirt/struct_2_F99_flirt.mat
```

Notes:

- Annotation names become part of the resource name, so keep them unique within a resolution block.
- Transform blocks accept `references` but not `notes`, so the composite-warp preference is documented here and under Known limitations.
- A local `uri` may use a different file name from the key. The key, not the file name, is what you query by.

---

## Step 4: Validate

The schemas in `schemas/` validate node and edge definitions separately, the way the bundled files are laid out: one node per file (the node name is the top-level key), and one list of edges per file. Split your combined file into those pieces, then run:

```bash
uv run --only-group validate check-jsonschema \
  --schemafile schemas/neuromaps_nodes.schema.json bigmac_node.yaml
uv run --only-group validate check-jsonschema \
  --schemafile schemas/neuromaps_volume_edges.schema.json bigmac_volume_edges.yaml
```

Then check that the files resolve:

```python
node = graph.get_node_data("BigMac")
for atlas in node.volumes:
    print(atlas.name, atlas.fetch())      # raises FileNotFoundError if the path is wrong
for annot in node.volume_annotations:
    print(annot.name, annot.fetch())
```

---

## Known limitations

- **Transform format.** Composite warps are preferred. `VolumeTransformOps` applies a transform with ANTs (`antsApplyTransforms`, single transform file). FSL FNIRT fields and FLIRT `.mat` files are not ANTs format. They can be registered in the graph, but warping with them through the graph will not work until they are converted (for example to an ANTs composite warp) and listed as `composite`.
- **Direction.** The edge above only goes `BigMac` → `F99`. The reverse needs its own inverted warp.
- **Missing files.** A node or edge whose files are absent from disk is unusable. Make sure the paths exist on each machine, and keep unpublished references out of a shared YAML.
