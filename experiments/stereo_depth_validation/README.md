# Stereo depth validation

Evidence for the depth accuracy claims in `docs/stereo_depth.md`. Reproduce
every published figure, without a camera, from this repository:

```bash
python scripts/report_stereo_validation.py            # the numbers
python scripts/report_stereo_validation.py --markdown # paper-ready tables
pytest tests/test_stereo_validation.py                # the claims still hold
```

`measurements.json` is the raw record: the hardware, the method, and every
disparity reading with the frame-to-frame spread that produced it. Nothing in
the documentation is a number typed by hand — the report recomputes all of it,
and the tests fail if a change to the depth model would invalidate a claim.

## What was measured

A side-by-side stereo webcam, against a rigid textured box at tape-measured
distances. Five distances between 0.31 m and 1.26 m fit the model; three
further distances, measured afterwards and never used in the fit, test it.

## What it shows

1. **The sensors are toed in, converging at 0.93 m.** Disparity passes through
   zero there and is negative beyond, so the usual search over positive
   disparity alone fails past that distance — and fails quietly, reporting
   distant objects at roughly the convergence distance. A box at 1.26 m was
   reported at 0.87 m. For obstacle avoidance that is the dangerous direction.
2. **Fitting the disparity offset alongside the focal length gives 1-2%.**
   Worst fit residual 22 mm; worst held-out error 17 mm.
3. **One distance is not enough to calibrate.** With two unknowns, a single
   measurement fixes a focal length that agrees with itself and is wrong
   everywhere else: the first attempt here read 0.42 m for a 0.50 m target.
4. **Contrast equalisation roughly doubles edge agreement for free**, and
   leaves distance untouched, while full 8-direction matching buys nothing for
   4.5 times the time.

## Honest scope

Central-patch measurements from one camera in one session, correcting depth
scale only. Distortion and residual misalignment are uncorrected, so this is a
best case, and the errors away from the image centre will be larger. The
held-out residuals are systematically negative, which is consistent with the
tape origin sitting about a centimetre ahead of the optical centre. Rolling
shutter, which matters more than depth error for a moving robot, was not
characterised. `measurements.json` carries the full list.
