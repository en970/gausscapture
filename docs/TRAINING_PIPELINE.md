# Training Pipeline

1. Import video or CapturePack.
2. Analyze quality.
3. Extract frames to `frames/images`.
4. Use metadata poses if present, otherwise run COLMAP.
5. Train locally with an external Gaussian Splatting trainer or create a Colab package.
6. Import trained `.ply`, `.splat`, `.ksplat`, or result `.zip`.
7. Build preview and exports.

Local training calls:

```text
python train.py -s <project_path> -m <run_output> --iterations <preset_iterations> --resolution <preset_resolution>
```

The wrapper is intentionally thin so a real trainer repository can be swapped in via settings.

