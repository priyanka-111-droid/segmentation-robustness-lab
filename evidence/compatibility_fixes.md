# Execution Compatibility Fixes

The robustness protocol and cohort manifests were frozen at:

d243359b543a811b84f251565fe58cd5e5bb2e8c

Two execution-blocking compatibility fixes were made after the protocol freeze. Neither changed the cohorts, model, weights, inference settings, metrics, thresholds, or selection procedure.

## c5f700a — Fix evaluation preparation help argument

Commit:

c5f700a

Changed a typo in src/prepare_evaluation.py:

- elp= → help=

This allowed the evaluation-preparation CLI argument to be parsed correctly. No experimental configuration was changed.

## 4901a83 — Handle COCO RLE counts across pycocotools versions

Commit:

4901a8327244b6c86ed533cb26c1547010ce9f9c

The COCO mask RLE encoder returned rle["counts"] as a string in the execution environment, while the existing code assumed bytes and unconditionally called .decode("utf-8").

The serialization logic was changed to decode only when rle["counts"] is actually a bytes object:

if isinstance(rle["counts"], bytes):
    rle["counts"] = rle["counts"].decode("utf-8")

This was an execution compatibility fix only. The prediction masks, inference settings, evaluation metrics, cohorts, and failure thresholds were unchanged.

## Verification

After the second fix:

- python -m py_compile src/evaluate.py passed.
- All repository tests passed: 5 passed.
- The bounded evaluation completed successfully and produced the recorded metrics.
