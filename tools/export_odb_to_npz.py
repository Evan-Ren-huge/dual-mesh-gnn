# -*- coding: utf-8 -*-

from odbAccess import openOdb
import os
import glob
import re
import argparse
import numpy as np


# ---------------------- PAD selection helpers ----------------------
def parse_shifts_from_fname(path):
    """
    Parse shiftA/shiftB from filename tokens like:
      Job-new8a50b0.odb  -> (50, 0)
      Job-a-25b+10.odb   -> (-25, +10)
    """
    fname = os.path.basename(path)
    ma = re.search(r'a\s*([+-]?\d+(?:\.\d+)?)', fname, re.IGNORECASE)
    mb = re.search(r'b\s*([+-]?\d+(?:\.\d+)?)', fname, re.IGNORECASE)
    sa = float(ma.group(1)) if ma else 0.0
    sb = float(mb.group(1)) if mb else 0.0
    return sa, sb


def pick_pad_nodes(asm, inst_name, shiftA, shiftB,
                   base_xmin, base_xmax,
                   base_azmin, base_azmax,
                   base_bzmin, base_bzmax,
                   y_plane, y_tol, xz_pad,
                   restrict_nset=""):
    """
    Pick PAD A/B node labels by geometric windows (same logic as original).
    """
    if inst_name not in asm.instances:
        raise ValueError("Instance '%s' not found in ODB." % inst_name)
    inst = asm.instances[inst_name]

    xmin  = base_xmin - xz_pad
    xmax  = base_xmax + xz_pad
    azmin = base_azmin + shiftA - xz_pad
    azmax = base_azmax + shiftA + xz_pad
    bzmin = base_bzmin + shiftB - xz_pad
    bzmax = base_bzmax + shiftB + xz_pad

    print("PAD A window: x=[%.3f, %.3f], z=[%.3f, %.3f], y=%.3f±%.3f"
          % (xmin, xmax, azmin, azmax, y_plane, y_tol))
    print("PAD B window: x=[%.3f, %.3f], z=[%.3f, %.3f], y=%.3f±%.3f"
          % (xmin, xmax, bzmin, bzmax, y_plane, y_tol))

    restrict_ids = None
    if restrict_nset:
        ns = None
        if restrict_nset in inst.nodeSets:
            ns = inst.nodeSets[restrict_nset]
        elif restrict_nset in asm.nodeSets:
            ns = asm.nodeSets[restrict_nset]
        if ns is None:
            print("WARNING: restrict node set '%s' not found; ignoring." % restrict_nset)
        else:
            restrict_ids = set([n.label for n in ns.nodes])
            print("Restricting to node set '%s' (%d nodes)" % (restrict_nset, len(restrict_ids)))

    A_nodes = []
    B_nodes = []
    for n in inst.nodes:
        if restrict_ids is not None and n.label not in restrict_ids:
            continue
        x, y, z = n.coordinates
        if abs(y - y_plane) > y_tol:
            continue
        if (z >= azmin) and (z <= azmax):
            A_nodes.append(n.label)
            continue
        if (z >= bzmin) and (z <= bzmax):
            B_nodes.append(n.label)
            continue

    print("Picked PAD nodes: A=%d, B=%d" % (len(A_nodes), len(B_nodes)))
    return sorted(A_nodes), sorted(B_nodes)


# ---------------------- Core export ----------------------
def export_one_odb(odb_path, out_dir,
                   step_name, inst_name, head_nset_name,
                   base_xmin, base_xmax,
                   base_azmin, base_azmax,
                   base_bzmin, base_bzmax,
                   y_plane, y_tol, xz_pad,
                   restrict_nset=""):
    print("Processing:", odb_path)
    odb = None
    try:
        odb = openOdb(path=odb_path)
        asm = odb.rootAssembly

        if inst_name not in asm.instances:
            raise ValueError("Instance '%s' not found in rootAssembly." % inst_name)
        inst = asm.instances[inst_name]

        if head_nset_name not in asm.nodeSets:
            raise ValueError("Node set '%s' not found in rootAssembly.nodeSets." % head_nset_name)
        head_nset = asm.nodeSets[head_nset_name]

        if step_name not in odb.steps:
            raise ValueError("Step '%s' not found in ODB." % step_name)
        step = odb.steps[step_name]

        # --- SURF1_NODE_LABELS (PAD A ∪ PAD B) ---
        shiftA, shiftB = parse_shifts_from_fname(odb_path)
        print("Parsed shifts: A=%+g, B=%+g" % (shiftA, shiftB))
        padA, padB = pick_pad_nodes(
            asm, inst_name, shiftA, shiftB,
            base_xmin, base_xmax,
            base_azmin, base_azmax,
            base_bzmin, base_bzmax,
            y_plane, y_tol, xz_pad,
            restrict_nset=restrict_nset
        )
        surf_nodes = sorted(set(padA) | set(padB))
        SURF1_NODE_LABELS = np.array(surf_nodes, dtype=np.int32)
        print("SURF1_NODE_LABELS size (A∪B):", SURF1_NODE_LABELS.size)

        # --- Static info ---
        node_labels = np.array([n.label for n in inst.nodes], dtype=np.int32)               # (N,)
        node_coords = np.array([n.coordinates for n in inst.nodes], dtype=np.float32)       # (N,3)
        connectivity = np.array([e.connectivity for e in inst.elements], dtype=np.int32)    # (E,8)

        # node -> elements mapping (same as original)
        node2elems = {int(lbl): [] for lbl in node_labels.tolist()}
        elem_labels = [e.label for e in inst.elements]
        for e_lbl, conn in zip(elem_labels, connectivity):
            for n_lbl in conn:
                node2elems[int(n_lbl)].append(int(e_lbl))

        # --- Dynamic containers ---
        disp_list, s_list, peeq_list = [], [], []
        head_u2, frame_times = [], []

        for frame in step.frames:
            # keep only frames that contain U/S/PEEQ
            if not all(k in frame.fieldOutputs for k in ('U', 'S', 'PEEQ')):
                continue

            # U at nodes
            u_vals = frame.fieldOutputs['U'].getSubset(region=inst).values
            disp_list.append(
                np.array([u_vals[i].data for i in range(len(node_labels))], dtype=np.float32)
            )  # (N,3)

            # element-level mises and peeq
            s_elem_vm = {v.elementLabel: v.mises
                         for v in frame.fieldOutputs['S'].getSubset(region=inst).values}
            p_elem = {v.elementLabel: v.data
                      for v in frame.fieldOutputs['PEEQ'].getSubset(region=inst).values}

            # nodal average over adjacent elements
            s_node_vm = np.zeros(len(node_labels), dtype=np.float32)
            peeq_node = np.zeros(len(node_labels), dtype=np.float32)
            for i, lbl in enumerate(node_labels.tolist()):
                elems = node2elems[int(lbl)]
                if elems:
                    s_node_vm[i] = sum([float(s_elem_vm.get(e, 0.0)) for e in elems]) / float(len(elems))
                    peeq_node[i] = sum([float(p_elem.get(e, 0.0)) for e in elems]) / float(len(elems))

            s_list.append(s_node_vm)       # (N,)
            peeq_list.append(peeq_node)    # (N,)

            # head U2 (average Y displacement over head_nset)
            u2_vals = [v.data[1] for v in frame.fieldOutputs['U'].getSubset(region=head_nset).values]
            head_u2.append(sum(u2_vals) / float(len(u2_vals)))

            frame_times.append(frame.frameValue)

        if len(frame_times) == 0:
            raise RuntimeError("No valid frames kept (missing U/S/PEEQ in all frames?).")

        disp_arr = np.stack(disp_list).astype(np.float32)          # (T,N,3)
        s_arr = np.stack(s_list).astype(np.float32)                # (T,N)
        peeq_arr = np.stack(peeq_list).astype(np.float32)          # (T,N)
        frame_times_arr = np.array(frame_times, dtype=np.float32)  # (T,)

        # --- RF2 history -> interpolate to kept frames ---
        rf2_on_frames = np.zeros_like(frame_times_arr, dtype=np.float32)
        try:
            rf2_times_all = None
            rf2_vals_all = None
            for hkey, hreg in step.historyRegions.items():
                if 'set-rp1' in hkey.lower():
                    if 'RF2' in hreg.historyOutputs:
                        rf2_hist = hreg.historyOutputs['RF2']
                        rf2_times_all = np.array([pt[0] for pt in rf2_hist.data], dtype=np.float32)
                        rf2_vals_all = np.array([pt[1] for pt in rf2_hist.data], dtype=np.float32)
                        print("Found RF2 history in:", hkey)
                    break

            if rf2_times_all is not None and len(rf2_times_all) > 0:
                rf2_on_frames = np.interp(frame_times_arr, rf2_times_all, rf2_vals_all).astype(np.float32)
            else:
                print("WARNING: RF2 history not found; using zeros.")
        except Exception as e:
            print("WARNING: RF2 extraction failed; using zeros. Error:", e)

        # --- Save NPZ ---
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        out_path = os.path.join(out_dir, os.path.basename(odb_path).replace('.odb', '.npz'))
        np.savez_compressed(
            out_path,
            node_labels=node_labels,
            node_coords=node_coords,
            connectivity=connectivity,
            disp=disp_arr,
            s=s_arr,
            peeq=peeq_arr,
            head_u2=np.array(head_u2, dtype=np.float32),
            frame_times=frame_times_arr,
            SURF1_NODE_LABELS=SURF1_NODE_LABELS,
            rf2=rf2_on_frames
        )
        print("Saved:", out_path, "| frames kept:", len(frame_times_arr))

    finally:
        if odb is not None:
            odb.close()


def parse_args():
    p = argparse.ArgumentParser("Export Abaqus ODB -> NPZ (Abaqus Python)")
    p.add_argument("--odb_dir", type=str, required=True, help="Folder containing .odb files")
    p.add_argument("--pattern", type=str, default="*.odb", help="Glob pattern under odb_dir (default: *.odb)")
    p.add_argument("--out_dir", type=str, required=True, help="Output folder for .npz files")

    p.add_argument("--step_name", type=str, default="Step-1")
    p.add_argument("--inst_name", type=str, default="COCRET-1")
    p.add_argument("--head_nset", type=str, default="SET-7")

    # PAD selection parameters (same defaults as your current script)
    p.add_argument("--base_xmin", type=float, default=0.0)
    p.add_argument("--base_xmax", type=float, default=150.0)
    p.add_argument("--base_azmin", type=float, default=1700.0)
    p.add_argument("--base_azmax", type=float, default=1800.0)
    p.add_argument("--base_bzmin", type=float, default=900.0)
    p.add_argument("--base_bzmax", type=float, default=1000.0)

    p.add_argument("--y_plane", type=float, default=250.0)
    p.add_argument("--y_tol", type=float, default=1.0)
    p.add_argument("--xz_pad", type=float, default=0.0)

    p.add_argument("--restrict_nset", type=str, default="", help="Optional: restrict PAD picking to a node set")
    return p.parse_args()


def main():
    args = parse_args()
    odb_pattern = os.path.join(args.odb_dir, args.pattern)
    odb_files = sorted(glob.glob(odb_pattern))
    if not odb_files:
        raise FileNotFoundError("No ODB files matched: %s" % odb_pattern)

    for odb_path in odb_files:
        export_one_odb(
            odb_path, args.out_dir,
            args.step_name, args.inst_name, args.head_nset,
            args.base_xmin, args.base_xmax,
            args.base_azmin, args.base_azmax,
            args.base_bzmin, args.base_bzmax,
            args.y_plane, args.y_tol, args.xz_pad,
            restrict_nset=args.restrict_nset
        )

    print("All done!")


if __name__ == "__main__":
    main()
