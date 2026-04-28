# splineslicer

[![License](https://img.shields.io/pypi/l/splineslicer.svg?color=green)](https://github.com/alvillars/splineslicer/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/splineslicer.svg?color=green)](https://pypi.org/project/splineslicer)
[![Python Version](https://img.shields.io/pypi/pyversions/splineslicer.svg?color=green)](https://python.org)
[![tests](https://github.com/alvillars/splineslicer/actions/workflows/test_and_deploy.yml/badge.svg)](https://github.com/alvillars/splineslicer/actions)

A python package for slicing nD images along splines. This tool is designed to help align and measure curved structures in 3D/nD biological images by "unrolling" them along a central axis (spline).

---

*Note: This README has been updated and structured with the assistance of AI to ensure accuracy and clarity.*

## Project Structure & Features

The project is organized into several modules, each handling a specific step in the image processing pipeline:

- **`skeleton`**: Tools for binarizing images and extracting a "skeleton" or centerline of the structure of interest. It includes a GUI for interactive pruning of branches.
- **`spline`**: Functions to convert skeletons into smooth mathematical splines (B-splines) that can be sampled continuously.
- **`slice`**: The core logic for "slicing" the original nD image volume perpendicularly to the spline, effectively straightening the structure.
- **`rotate`**: Tools to align the orientation of the slices (e.g., aligning the dorso-ventral axis) so that the structure is consistently oriented across the entire stack.
- **`measure`**: Functions for measuring boundaries, aligning slices, and extracting quantitative data from the straightened images.
- **`sub_slicing`**: Specialized logic for further refining the slicing process.
- **`correction`**: A GUI widget for manually correcting spline points or measurement results.
- **`view`**: Result viewers for inspecting the output of the slicing and measurement pipeline.
- **`io`**: Handlers for reading and writing data in various formats, including HDF5, Ilastik projects, and JSON splines.

## Installation

You can install `splineslicer` via [pip]:

    pip install splineslicer

For the latest development version:

    pip install git+https://github.com/alvillars/splineslicer.git

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"splineslicer" is free and open source software.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/alvillars/splineslicer/issues
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
