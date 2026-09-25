# Publish the submitted-revision code

This release preserves the submitted numerical implementation and saved evidence. It fixes weak-table output-path handling and makes reporting usable in a standalone checkout. No additional experiments are required. `RELEASE_VALIDATION.json` records the release checks; `RELEASE_MANIFEST.json` records file hashes and their relationship to the submitted code package. Historical validation records are retained as provenance.

## Update your existing checkout

Extract the release archive. Its top-level folder is `koopman_sindy/`. Start with a clean working tree and index in your existing repository, so the release commit does not absorb unrelated work. Replace the placeholder paths below with your actual paths. Copy the folder's **contents**, including `.gitignore`, into your checkout; this does not replace the repository's `.git` directory or remove other files.

```bash
cd /path/to/existing/repository
git status --short
cp -R /path/to/extracted/koopman_sindy/. .
git diff --stat
git diff
```

If your checkout already has a `.gitignore`, retain its project-specific rules when merging the supplied entries; any such local change will differ from the delivered manifest.

Review the changes, then stage only the files listed in the delivered manifest, together with the manifest itself:

```bash
python - <<'PY'
import json
import subprocess
from pathlib import Path

manifest = json.loads(Path('RELEASE_MANIFEST.json').read_text())
paths = sorted(manifest['files']) + ['RELEASE_MANIFEST.json']
subprocess.run(['git', 'add', '--', *paths], check=True)
PY

git diff --cached --stat
git diff --cached
git commit -m "Release code and results for submitted DCE revision"
git tag -a dce-revision-2026-09-25 -m "Code and results accompanying submitted DCE-2026-0088 revision"
```

The tag identifies the submitted numerical evidence plus these reporting fixes. It does not imply that the modified release passes the historical full-manuscript validator. No commit or tag has been created for you. If this tag already exists, inspect it before choosing a new name; do not overwrite it.

When you are ready, push your current branch and the new tag using your repository's normal remote (`origin` below):

```bash
git push origin HEAD
git push origin dce-revision-2026-09-25
```

Keep the delivered release unchanged if you want its manifest to remain a file-for-file reference. Run reporting or experiments in a separate working copy; those commands refresh generated outputs and provenance as described in `README.md`.
