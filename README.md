# <img src="Icon_Ext.jpg" width="45" valign="middle"> &nbsp; TSF-85 Tactile Sensor

## Table of Contents
- [Overview](#overview)
- [Isaac Sim Compatibility](#isaac-sim-compatibility)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
- [Features](#features)
- [Usage](#usage)
  - [Assets](#assets)
  - [Interactive — Isaac Sim GUI](#interactive--isaac-sim-gui)
  - [Headless — standalone Python script](#headless--standalone-python-script)
    - [Example 1 — Touch objects](#example-1--touch-objects)
    - [Example 2 — Grasp and lift objects](#example-2--grasp-and-lift-objects)
  - [Notes](#notes)
- [Citation](#citation)
- [Contact](#contact)
- [More Information](#more-information)

## Overview
The TSF-85 Isaac Sim extension provides a custom user interface panel for generating
synthetic tactile maps for the Robotiq tactile sensor **TSF-85**. It was developed in
collaboration with the Control and Robotics Laboratory (CoRo) at the École de
technologie supérieure (ÉTS) in Montréal.

<p align="center">
  <img width="80%" alt="TSF-85 sensor and example scene with a Robotiq gripper" src="Scenes.jpg">
</p>

## Isaac Sim Compatibility
This extension is compatible with:
* NVIDIA Isaac Sim 5.1.0 (developed and tested on this version)
* NVIDIA Isaac Sim 6.0.0 — support in progress
* Tested on Linux (Ubuntu 22.04)

### Development environment
The extension was developed and validated on the following machine. Deformable-body
simulation and ONNX/cuDNN inference are both GPU-bound, so the GPU and driver/CUDA
versions are the most relevant figures to match.

| Component | Specification |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS |
| CPU | 12th Gen Intel Core i9-12900H |
| GPU | NVIDIA RTX A2000 8GB Laptop GPU (8 GB VRAM) |
| RAM | 32 GB |
| NVIDIA driver | 580.159.03 |
| CUDA (driver) | 13.0 |
| cuDNN | 9.7.1 (bundled with Isaac Sim) |
| Isaac Sim | 5.1.0 |
| Python | 3.11.13 (bundled with Isaac Sim) |

## Installation
### Prerequisites
* Python 3.11.13 (Python runtime bundled with Isaac Sim)
* [NumPy](https://pypi.org/project/numpy/) for array and math operations
* [onnxruntime](https://pypi.org/project/onnxruntime/) for running the CNN model
* [Pillow](https://pypi.org/project/pillow/) for rendering the tactile-map heatmap images

> NumPy and Pillow usually ship with the Python runtime bundled in Isaac Sim, so in a
> standard install you typically only need to add `onnxruntime`. Install into Isaac Sim's
> bundled Python (e.g. `./python.sh -m pip install onnxruntime`), not your system Python.

### Steps
A short summary of the install is below. For the full walkthrough, see
📖 [docs/installation.md](docs/installation.md).

1. Clone the repository.
2. Add the extension's search path in Isaac Sim (point it at the folder containing the
   extension) and enable **TSF_85_Ext** — either through the Extension Manager (GUI) or
   via carb settings in a Python launcher (see [Headless](#headless--standalone-python-script)).

## Features
* User interface panel to interact with the sensor prim.
* Supports up to two sensors per environment.
* Data generation rate matches the simulation refresh rate.
* Automatic generation of CSV files containing essential information.
* User-defined output location for generated files.
* Real-time visualization of synthetic tactile maps.

## Usage
The extension can be driven two ways: **interactively** through the Isaac Sim GUI, or
**headless** from a standalone Python script that wires the tactile-data pipeline
directly into your own simulation.

### Assets
The `assets/` folder contains the USD files used by the example scenes:

* **Robotiq 2F-85 gripper (modified)** — a 2F-85 adaptive gripper modified to mount the
  TSF-85 tactile sensors in place of the standard fingertips.
* **TSF-85 sensor** — the tactile sensor USD, including the deformable sensing mesh that
  the extension reads to compute deformation and predict the tactile map.

These are referenced by the example scenes in `examples/scenes/`. For step-by-step
instructions on mounting the TSF-85 sensors onto the gripper, see
📖 [docs/attaching_sensors.md](docs/attaching_sensors.md).

### Interactive — Isaac Sim GUI

📖 For the full step-by-step workflow and a walkthrough of every panel control, see
[docs/gui_guide.md](docs/gui_guide.md).

### Headless — standalone Python script
The extension can run entirely from Python, with no GUI. This lets you implement the
tactile-data generation pipeline directly inside your own standalone script: you
configure the sensor(s), register the extension, and step physics yourself, and the
extension records the same CSV files it would produce in the GUI.

To wire the pipeline into a script, set the extension's carb settings **before**
enabling it, then register the extension path and enable it:

```python
import carb.settings

settings = carb.settings.get_settings()
settings.set("/exts/TSF_85_Ext/headless",      True)            # drop the GUI/panel gate
settings.set("/exts/TSF_85_Ext/sensor_root",   SENSOR_ROOT)     # sensor 1 case prim path (-> _s1)
settings.set("/exts/TSF_85_Ext/sensor_root_2", SENSOR_ROOT_2)   # optional sensor 2 (-> _s2)
settings.set("/exts/TSF_85_Ext/output_dir",    OUTPUT_DIR)      # where the CSVs go
settings.set("/exts/TSF_85_Ext/base_name",     BASE_NAME)       # output filename prefix
settings.set("/exts/TSF_85_Ext/log_dz",        True)            # write *_deformations.csv
settings.set("/exts/TSF_85_Ext/log_pred",      True)            # write *_tactile_maps.csv
settings.set("/exts/TSF_85_Ext/log_mesh",      True)            # write *_mesh_state.csv
settings.set("/exts/TSF_85_Ext/record_active", False)           # start gated OFF (see below)

from omni.kit.app import get_app
ext_mgr = get_app().get_extension_manager()
ext_mgr.add_path(EXT_SEARCH_PATH)   # folder containing the extension, so Kit can discover it
ext_mgr.set_extension_enabled_immediate("TSF_85_Ext", True)
```

**Two sensors.** Setting `sensor_root_2` in addition to `sensor_root` enables
two-sensor mode (e.g. the left and right pads of a gripper). Sensor 1 produces files
infixed with `_s1`, sensor 2 with `_s2`. With a single sensor only `sensor_root` is set.

**Sensor-root paths.** Give the sensor prim path in its authored form (e.g.
`/World/robot_gripper_adapter_sensor/TSF_85_right/TSF_85`). When the robot/sensor USD is
brought in as a reference, its contents can end up nested one level deeper at runtime;
the extension resolves the configured path to the prim that actually exists on the live
stage, so the authored path keeps working without manual adjustment.

**Choosing when to record (`record_active`).** Recording is gated by the
`/exts/TSF_85_Ext/record_active` carb setting (default **OFF**). While it is `False` the
extension still finds the mesh, computes deformation, and runs inference (so its state
stays warm), but writes **no** CSV rows. Flip it `True` at the moment you want recording
to begin and back to `False` when you want it to stop — for a grasp, typically `True`
just before the gripper starts to close and `False` once it has fully opened again. The
frame numbers written are the real simulation frame indices (they are not reset to zero),
so a recording that starts mid-run begins at the simulation frame where the gate opened:

```python
# ... approach / descent (not recorded) ...
settings.set("/exts/TSF_85_Ext/record_active", True)    # start of close
#   close -> hold -> open
settings.set("/exts/TSF_85_Ext/record_active", False)   # after open finishes
# ... ascent / retreat (not recorded) ...
```

#### Example 1 — Touch objects
`examples/touch_cylinder.py`

A Robotiq gripper fitted with two TSF-85 tactile sensors closes on an object and reads
its tactile signature. The gripper only **squeezes** — there is no lift. The object stays
on the table the whole time, so the sensors capture the normal-force pressure
distribution of the object's shape against the pads.

This example **depends on [cuRobo](https://curobo.org/)** for motion planning: cuRobo
generates the arm trajectory to approach the object, a straight-line Z descent to the
grasp pose, and the straight-line ascent afterwards. The example enables the extension
headless with both sensor roots set, and toggles `record_active` so that only the
close → hold → open window is written to the CSVs. Each run produces the three files per
sensor (`_s1` and `_s2`): `*_deformations.csv`, `*_tactile_maps.csv`, and
`*_mesh_state.csv`.

### Notes
* This extension is best suited for scenes or tasks where **static friction at the soft
  contact is not involved, or does not play a critical role** (for example, squeeze /
  normal-force experiments). The simulator's deformable-body solver models only kinetic
  friction at the soft-body ↔ rigid-body contact, so tasks that depend on a static-
  friction "stick" phase (such as lifting an object by friction alone) cannot be
  reproduced faithfully without a workaround like the grasp aid in Example 2.
* Each sensor produces **three** CSV files, all built from the base name. `_deformations`
  holds the per-node sensor-mesh deformation, `_tactile_maps` holds the generated tactile
  maps, and `_mesh_state` holds the full per-node-per-frame mesh state. In two-sensor mode
  the files are additionally infixed with `_s1` (sensor 1) and `_s2` (sensor 2). Individual
  files can be turned off with the `log_dz` / `log_pred` / `log_mesh` carb settings.
* The extension can also run in headless mode, driven entirely by a standalone
  Python script without launching the Isaac Sim GUI.

## Citation
If you use this extension in your research, please cite the following paper:
```bibtex
@article{delacruz2025hybrid,
  title   = {A hybrid elastic-hyperelastic approach for simulating soft tactile sensors},
  author  = {De la Cruz S{\'a}nchez, Berith Atemoztli and Roberge, Jean-Philippe},
  journal = {Frontiers in Robotics and AI},
  volume  = {12},
  pages   = {1639524},
  year    = {2025},
  publisher = {Frontiers Media SA}
}
```

## Contact
For any questions, suggestions, or feedback, please feel free to reach out:

**Lead Maintainer**
**Berith De la Cruz Sánchez**
Email: [berithcruzs@gmail.com](mailto:berithcruzs@gmail.com)
GitHub: [BerithCS](https://github.com/BerithCS)

**Project Supervisor**
**Prof. Jean-Philippe Roberge**
Control and Robotics Laboratory (CoRo), École de technologie supérieure (ÉTS)
Email: [jean-philippe.roberge@etsmtl.ca](mailto:jean-philippe.roberge@etsmtl.ca)

## More Information
We are currently developing a **hybrid co-simulation** approach to compensate for the
limitations of deformable bodies in the simulator, with the goal of recovering the
physical behavior (including static friction at the soft contact) that the current
deformable-body solver cannot reproduce. This section will be updated as that work
matures.

