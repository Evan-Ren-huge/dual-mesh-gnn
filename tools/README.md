# Abaqus ODB → NPZ Export Tool

This tool converts Abaqus `.odb` simulation results into compressed `.npz` files
for downstream machine-learning workflows.

The script must be executed using **Abaqus/CAE Python**, as it depends on `odbAccess`.

---

## What This Script Does

For each Abaqus `.odb` file, the exporter:

- Reads nodal geometry and mesh connectivity.
- Extracts time-dependent field outputs.
- Filters valid frames to ensure consistent temporal data.
- Converts element-level quantities to nodal representations.
- Saves all results into a single compressed `.npz` file.

Each `.odb` file produces exactly **one** `.npz` file.

---

## Output NPZ Schema

Each exported `.npz` file contains the following keys:

- `node_labels` : `(N,) int32`
- `node_coords` : `(N, 3) float32`
- `connectivity`: `(E, 8) int32` (C3D8 elements)

- `disp` : `(T, N, 3) float32`  
  Nodal displacement (`U`)

- `s` : `(T, N) float32`  
  Nodal von Mises stress, averaged from adjacent elements

- `peeq` : `(T, N) float32`  
  Nodal equivalent plastic strain, averaged from adjacent elements

- `frame_times` : `(T,) float32`  
  Simulation time values for retained frames

- `head_u2` : `(T,) float32`  
  Mean displacement component `U2` over a specified node set

- `rf2` : `(T,) float32`  
  Reaction force `RF2`, interpolated to `frame_times`

- `SURF1_NODE_LABELS` : `(Ns,) int32`  
  Node labels defining a surface or region of interest

---

## Frame Selection Rules

Only frames that contain **all** of the following field outputs are retained:

- `U`
- `S`
- `PEEQ`

Frames missing any of these outputs are skipped.
This guarantees that all exported arrays share the same temporal length `T`.

---

## Stress and Plastic Strain Mapping

Abaqus stores stress (`S`) and plastic strain (`PEEQ`) primarily at the
element or integration-point level.

This exporter converts them to nodal values by:

- Building a node → adjacent element mapping.
- Averaging element values over all elements connected to each node.

This mapping strategy is applied consistently across all cases.

---

## Reaction Force (`RF2`) Handling

The exporter searches step history outputs for `RF2`
(typically associated with a reference point or node set).

- If found, `RF2` is linearly interpolated to match `frame_times`.
- If not found, `rf2` is saved as zeros and a warning message is printed.

---

## SURF1_NODE_LABELS (PAD A / PAD B Selection)

`SURF1_NODE_LABELS` is constructed as the union of two node groups
(PAD A and PAD B), selected using geometric windows in the model
coordinate system.

Optional filename-based shifts (e.g., tokens such as `a+50`, `b-25`)
may be parsed from the `.odb` filename to translate PAD regions
on a per-case basis.

All PAD-related parameters are configurable via command-line arguments.

---

## Usage

### Basic batch export

```bash
abaqus python tools/export_odb_to_npz.py \
  --odb_dir <ODB_DIR> \
  --out_dir <NPZ_DIR>
