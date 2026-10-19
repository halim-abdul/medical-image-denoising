# Data handling

No patient data are bundled. Check each dataset's current license, access terms, attribution, and redistribution rules before use. Keep raw and derived clinical images outside version control; common NIfTI, DICOM, NumPy and output files are ignored.

The first version accepts 2-D NumPy arrays so preprocessing stays explicit. A future volume adapter should preserve affine/orientation metadata and document resampling. Record conversion software, orientation, voxel spacing, slice selection, intensity transform and masking. Avoid patient identifiers in filenames, logs and public artifacts. Dataset access rules can change; verify them with the provider. No dataset is downloaded automatically.
