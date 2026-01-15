"""MOUSE_to_HUMAN_NIFTI.py

Project mouse ROI-wise map values (e.g., Map_1 / Map_2 from an SVD template) onto human MNI atlases
using a mouse→human ROI mapping table.

Outputs
-------
1) A 4D NIfTI in MNI space (one volume per map column; e.g., Map_1, Map_2, ...)
2) A ROI-wise CSV summary of the filled values per human ROI

Atlases
-------
This script is designed to work with standard atlases distributed with FSL.
If FSL is installed, you can typically find them under:
    $FSLDIR/data/atlases

You can point the script to that location via `--atlas-dir`.

Example
-------
```bash
python figures/fig3_4_sleep_transitions/scripts/cross_species/MOUSE_to_HUMAN_NIFTI.py \
  --roi-csv "/path/to/roi_by_map_values.csv" \
  --mapping-csv "/path/to/Selected_ROI_HUMAN_MOUSE.xlsx" \
  --atlas-dir "$FSLDIR/data/atlases" \
  --out-nifti "/path/to/Mouse_to_Human_MAPS.nii.gz" \
  --out-csv "/path/to/Mouse_to_Human_MAPS_ROIwise.csv"

"""
import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd

INDEX_SHIFT_BY_ATLAS = {
    "HarvardOxford-Cortical": 1,
    "HarvardOxford-Subcortical": 1,
    # MorelAtlasMNI152: no shift
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Project mouse ROI-wise map values onto human MNI atlases using a mouse–human ROI mapping table. "
            "Outputs a 4D NIfTI (one volume per map) and a ROI-wise CSV summary."
        )
    )

    p.add_argument(
        "--roi-csv",
        required=True,
        help=(
            "Path to roi_by_map_values.csv (rows=mouse ROI names, columns include Map_1, Map_2, ...)."
        ),
    )
    p.add_argument(
        "--mapping-csv",
        required=True,
        help=(
            "Path to mouse–human mapping table (CSV or XLSX). Must contain columns: "
            "Atlas, Mapped_ROI, Animal_ROI, and Index."
        ),
    )

    # Atlases: either provide atlas_dir (default) or explicit atlas paths.
    p.add_argument(
        "--atlas-dir",
        default=None,
        help=(
            "Base directory for atlases (e.g., $FSLDIR/data/atlases). If provided and explicit atlas paths are not, "
            "the script will look for the default FSL locations for HarvardOxford cortical/subcortical and Morel."
        ),
    )
    p.add_argument(
        "--atlas-cortical",
        default=None,
        help="Path to HarvardOxford cortical maxprob atlas NIfTI (2mm). Overrides --atlas-dir default.",
    )
    p.add_argument(
        "--atlas-subcortical",
        default=None,
        help="Path to HarvardOxford subcortical maxprob atlas NIfTI (2mm). Overrides --atlas-dir default.",
    )
    p.add_argument(
        "--atlas-morel",
        default=None,
        help="Path to Morel atlas NIfTI (2mm). Overrides --atlas-dir default.",
    )

    p.add_argument(
        "--out-nifti",
        required=True,
        help="Output path for the 4D NIfTI (e.g., Mouse_to_Human_MAPS.nii.gz).",
    )
    p.add_argument(
        "--out-csv",
        required=True,
        help="Output path for ROI-wise CSV summary (e.g., Mouse_to_Human_MAPS_ROIwise.csv).",
    )

    p.add_argument(
        "--map-cols",
        default="Map_1,Map_2",
        help="Comma-separated list of map column names in roi CSV (default: Map_1,Map_2).",
    )

    return p.parse_args()


def _resolve_atlases(args: argparse.Namespace) -> dict[str, str]:
    """Resolve atlas paths from explicit args or atlas_dir defaults."""

    atlases: dict[str, str] = {}

    # If explicit paths are provided, use them.
    if args.atlas_cortical is not None:
        atlases["HarvardOxford-Cortical"] = args.atlas_cortical
    if args.atlas_subcortical is not None:
        atlases["HarvardOxford-Subcortical"] = args.atlas_subcortical
    if args.atlas_morel is not None:
        atlases["MorelAtlasMNI152"] = args.atlas_morel

    # Fill missing with defaults from atlas_dir.
    if args.atlas_dir is not None:
        base = args.atlas_dir
        atlases.setdefault(
            "HarvardOxford-Cortical",
            os.path.join(base, "HarvardOxford/HarvardOxford-cort-maxprob-thr50-2mm.nii.gz"),
        )
        atlases.setdefault(
            "HarvardOxford-Subcortical",
            os.path.join(base, "HarvardOxford/HarvardOxford-sub-maxprob-thr50-2mm.nii.gz"),
        )
        atlases.setdefault(
            "MorelAtlasMNI152",
            os.path.join(base, "MorelAtlasMNI152/morel_2mm.nii.gz"),
        )

    # Sanity check
    missing = [k for k, v in atlases.items() if v is None]
    if missing:
        raise ValueError(f"Missing atlas path(s) for: {missing}")

    return atlases


def main() -> None:
    args = _parse_args()

    roi_csv = args.roi_csv
    mapping_csv = args.mapping_csv
    out_nifti = args.out_nifti
    out_csv = args.out_csv

    map_cols = [c.strip() for c in str(args.map_cols).split(",") if c.strip()]
    if len(map_cols) == 0:
        raise ValueError("--map-cols must contain at least one column name.")

    atlases = _resolve_atlases(args)

    # ============================
    # 1) LOAD TABLES
    # ============================

    roi_df = pd.read_csv(roi_csv, index_col=0)
    roi_df.index.name = "ROI"

    missing_cols = [c for c in map_cols if c not in roi_df.columns]
    if missing_cols:
        raise ValueError(
            f"roi CSV is missing required map column(s): {missing_cols}. "
            f"Available columns: {list(roi_df.columns)}"
        )

    # Mapping table can be CSV or Excel
    if mapping_csv.lower().endswith((".xlsx", ".xls")):
        mapping_df = pd.read_excel(mapping_csv, dtype=str, engine="openpyxl")
    else:
        mapping_df = pd.read_csv(mapping_csv, dtype=str)

    mapping_df.fillna("", inplace=True)

    required_mapping_cols = {"Atlas", "Mapped_ROI", "Animal_ROI"}
    if not required_mapping_cols.issubset(set(mapping_df.columns)):
        raise ValueError(
            "Mapping table must contain columns: "
            f"{sorted(required_mapping_cols)}. Found: {list(mapping_df.columns)}"
        )

    # ============================
    # 2) LOAD ATLASES
    # ============================

    atlas_imgs = {name: nib.load(path) for name, path in atlases.items()}
    atlas_datas = {name: img.get_fdata() for name, img in atlas_imgs.items()}

    # Use the first atlas as reference space (should all be MNI 2mm)
    shape = next(iter(atlas_datas.values())).shape
    affine = next(iter(atlas_imgs.values())).affine

    # Output has N volumes: one per map column
    output_4d = np.zeros(shape + (len(map_cols),), dtype=np.float32)

    # ============================
    # 3) MAIN LOOP: fill atlas ROIs with mouse values
    # ============================

    for _, row in mapping_df.iterrows():
        atlas_name = row["Atlas"]

        # Fallback to 'Index' for backward compatibility if needed.
        indices_str = row.get("Index", "")
        if indices_str is None or str(indices_str).strip() == "":
            indices_str = row.get("Index", "")

        human_roi = row["Mapped_ROI"]
        animal_roi_str = row["Animal_ROI"]

        if atlas_name not in atlas_datas:
            continue

        # Parse atlas indices from CSV (comma-separated)
        parsed_indices = [
            int(i)
            for i in str(indices_str).split(",")
            if i.strip().isdigit()
        ]
        if not parsed_indices:
            continue

        # Apply atlas-specific shift (ONLY for HarvardOxford atlases)
        shift = INDEX_SHIFT_BY_ATLAS.get(atlas_name, 0)
        atlas_indices = [i + shift for i in parsed_indices]

        atlas_data = atlas_datas[atlas_name]

        # Mask for the human ROI
        roi_mask = np.isin(atlas_data, atlas_indices)
        nvox = int(roi_mask.sum())

        if nvox == 0:
            continue

        # Parse mouse ROI names
        animal_rois = [r.strip() for r in str(animal_roi_str).split(",") if r.strip()]

        # Extract values from the mouse CSV (average across multiple mouse ROIs if needed)
        vals_per_map: list[list[float]] = [[] for _ in map_cols]

        for aro in animal_rois:
            if aro not in roi_df.index:
                continue
            for mi, col in enumerate(map_cols):
                vals_per_map[mi].append(float(roi_df.loc[aro, col]))

        if all(len(v) == 0 for v in vals_per_map):
            continue

        # Fill output volumes
        filled_vals = []
        for mi, col in enumerate(map_cols):
            if len(vals_per_map[mi]) == 0:
                # If a map is missing for this ROI, leave zeros in that volume.
                filled_vals.append(np.nan)
                continue
            val = float(np.mean(vals_per_map[mi]))
            output_4d[roi_mask, mi] = val
            filled_vals.append(val)

    # ============================
    # 4) SAVE NIFTI
    # ============================

    os.makedirs(os.path.dirname(os.path.abspath(out_nifti)), exist_ok=True)
    img = nib.Nifti1Image(output_4d, affine)
    nib.save(img, out_nifti)

    # ============================
    # 5) SAVE ROI-WISE CSV SUMMARY
    # ============================

    csv_rows = []

    for _, row in mapping_df.iterrows():
        human_roi = row["Mapped_ROI"]
        animal_roi_str = row["Animal_ROI"]
        animal_rois = [r.strip() for r in str(animal_roi_str).split(",") if r.strip()]

        vals_per_map = [[] for _ in map_cols]
        for aro in animal_rois:
            if aro in roi_df.index:
                for mi, col in enumerate(map_cols):
                    vals_per_map[mi].append(float(roi_df.loc[aro, col]))

        if all(len(v) == 0 for v in vals_per_map):
            continue

        out_row = {"ROI_HUMAN": human_roi}
        for mi, col in enumerate(map_cols):
            out_row[f"{col}_VALUE"] = float(np.mean(vals_per_map[mi])) if len(vals_per_map[mi]) else np.nan

        csv_rows.append(out_row)

    roi_summary_df = pd.DataFrame(csv_rows)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    roi_summary_df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()