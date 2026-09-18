# HIDE experiment result snapshot

Complete: **True**. Suites checked: **none**.

Every file under the raw result directory is included losslessly. INDEX.json maps compressed chunks to original filenames and provides SHA-256 hashes, counts and completeness checks. Chunks are byte segments; restore before reading JSONL.

Verify: `python -m hide.export_results --verify PATH_TO_THIS_FOLDER`

Restore: `python -m hide.export_results --restore PATH_TO_THIS_FOLDER --output RESTORED_FOLDER`
