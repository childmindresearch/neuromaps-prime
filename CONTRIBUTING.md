# Neuromaps-PRIME contributing guidelines

Pull requests are always welcome, and we appreciate any help you give. Note that a code of conduct applies to all spaces managed by the neuromaps-prime project, including issues and pull requests. Please see the [Code of Conduct](CODE_OF_CONDUCT.md) for details.

When submitting a pull request, we ask you to check the following:

1. **Unit tests**, **documentation**, and **code style** are in order.
   See the Continuous Integration for up to date information on the current code style, tests, and any other requirements.

   It is also OK to submit work in progress if you're unsure of what this exactly means, in which case you'll likely be asked to make some further changes.

2. The contributed code will be **licensed under the same [license](LICENSE) as the rest of the repository**, If you did not write the code yourself, you must ensure the existing license is compatible and include the license information in the contributed files, or obtain permission from the original author to relicense the contributed code.

## Contributing resources

We welcome contributions of new resources (annotations, atlases, transforms, etc.). There are two ways to contribute:

1. **Resource already has a public URI**

   If your resource is already hosted publicly, you can contribute it directly:
   - Locate the appropriate YAML file under `src/neuromaps_prime/resources/` — resources are organized by type (e.g. `nodes/`, `edges/`) and species (e.g. `human/`, `macaque/`)
   - Add your resource following the structure of the corresponding schema under [`/schemas`](https://github.com/childmindresearch/neuromaps-prime/tree/main/schemas)
      - For example: node entries follow `neuromaps_nodes.schema.json`
   - Ensure all required fields are included (e.g. `description`, `species`, and surface/volume URIs where applicable)
   - Open a [PR](https://github.com/childmindresearch/neuromaps-prime/pulls) following the guidelines above

2. **Resource needs to be uploaded**

   If your resource is not yet hosted anywhere:
   - Open a resource request issue using the [provided template](https://github.com/childmindresearch/neuromaps-prime/issues/new?template=resource_request.yaml)
   - Include any relevant details, including:
      - Resource description
      - Type (annotation, atlas, transform, or other)
      - Source link or reference (if available)
      - Publications or DOIs
      - Access restrictions or licensing information
      - Any additional notes/caveats
   - Maintainers will review the request and reach out with any questions or next steps
   - After upload, open a [PR](https://github.com/childmindresearch/neuromaps-prime/pulls) to add the resource to the appropriate YAML file(s)

> [!NOTE]
> When contributing a resource, please include where possible:
> - A direct URI or source link
> - Any relevant references (DOIs, publications)
> - Notes or caveats about the resource (e.g. acquisition method, version, species mapping)
