# Stereo calibration-sized input crop

Goal: integrate the user-approved center crop (1920x1200 to 1920x1088,
y=56) into the normal equidistant pipeline, retaining calibration intrinsics.

Implementation sequence:

1. Add regression tests in `tests/stereo_rectification/test_input_crop.py`:
   centered and unchanged inputs, invalid/missing/smaller resolutions,
   independent left/right crops, pipeline equivalence to pre-cropped video,
   and actual video frame cropping before remapping.
2. Run `python -m pytest tests/stereo_rectification/test_input_crop.py -q`
   to demonstrate failure before implementation.
3. Update `rectify_video.py`: require both calibration resolutions in the
   equidistant pipeline, inspect decoded frames from both streams, calculate
   centered integer ROIs, use calibration dimensions for rectification and
   map bounds, crop each decoded frame before remapping, and record input
   sizes/ROIs separately from the existing output crop in rectify_params.yaml.
   Keep K/D/R/t unchanged. Reject nonpositive/missing resolutions, input
   smaller than calibration, and differing left/right calibration sizes.
   For odd differences the extra removed pixel is on the bottom/right.
4. Update the module README with the center-crop assumption, failure cases,
   and `--overwrite` requirement for recomputing cached outputs.
5. Run the stereo tests and compare real sample output with the previously
   approved pre-cropped workflow, including P1/P2/Q and decoded frames.

Scope: no source calibration rewrite, rescaling, or changes to database
or stage CLI options. Existing unrelated working-tree changes are preserved.
