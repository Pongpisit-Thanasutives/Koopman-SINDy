# Historical validation of the submitted research package

This directory preserves the validation records and checks that accompanied the submitted manuscript. The recorded 14,800 passing checks apply to that complete, frozen submission snapshot, including its manuscript, response, figures, and original source hashes.

`validate_package.py` requires the original package layout with `code_repository/`, `latex_source/`, `submission/`, and `local_checking/`. Run it in the submitted research archive. It is not a standalone code-release validator, and copying manuscript folders beside the updated GitHub code does not restore the original audited source hashes.

The GitHub release changes reporting destinations and documentation while preserving the submitted numerical implementation and saved experimental evidence. Root-level `RELEASE_MANIFEST.json` records the release files and their relationship to the submitted code; `RELEASE_VALIDATION.json` records the focused release checks. Historical execution and audit records in `results/` retain their original contents.

From the repository root, the reporting-path regression test is:

```bash
python -m unittest discover -s tests -v
```

The standalone mathematical and numerical validation scripts are `validate_complex_propagation.py`, `validate_revision_weak.py`, and `validate_revision_tv_outputs.py`. They write diagnostics; run them in a separate working copy when you want to retain all delivered output bytes. See the root README for commands and the recorded environment. No experiment reruns are needed to use the saved results.
