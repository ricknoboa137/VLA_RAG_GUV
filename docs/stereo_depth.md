# Stereo depth on the 3D USB Camera

## The camera is toed in

Its two lenses converge rather than pointing in parallel, which suits the 3D
video it is sold for and complicates depth. Disparity reaches zero at **0.93 m**
and is negative beyond it, so a matcher searching only positive disparity
fails past that point — and fails silently, reporting everything distant at
roughly the convergence distance. An object at 1.26 m came back as 0.87 m
before this was found.

Always pass a negative `min_disparity` (`-48` for this camera).

## Calibration

`config/stereo_3dusb.json`, fitted from a box at five tape-measured distances
between 0.31 m and 1.26 m (`config/stereo_3dusb_samples.json`):

| | |
|---|---|
| focal | 567.5 px at 640x480 per eye |
| baseline | 60 mm |
| disparity offset | -36.5 px |
| convergence | 0.93 m |
| fit error | 22 mm worst, over 0.31-1.26 m |

Held-out checks, at distances not used in the fit: 1.00 m read 0.994 m, and
1.10 m read 1.083 m. So roughly **1-2% of the distance**.

Redo it after any knock to the camera:

```bash
python scripts/calibrate_stereo.py sample --distance 0.31
python scripts/calibrate_stereo.py sample --distance 0.60
python scripts/calibrate_stereo.py sample --distance 1.00
python scripts/calibrate_stereo.py sample --distance 1.50 --min-disparity -48
python scripts/calibrate_stereo.py fit --out config/stereo_3dusb.json
```

Use a rigid textured box, not a person: a box holds still to 0.2 px, a person
sways by several pixels. Spread the distances widely — near and far samples
are what separate the focal length from the offset, and two nearby distances
cannot, which is how a single-distance fit agrees with itself and is wrong
everywhere else.

## Useful range

At 640x480 per eye, from `error = 0.5 px * Z^2 / (focal * baseline)`:

| Distance | Disparity | Error | |
|---|---|---|---|
| 0.30 m | +77.0 px | 1 mm | nearest the search reaches (0.29 m) |
| 0.50 m | +31.6 px | 4 mm | |
| 1.00 m | -2.5 px | 15 mm | |
| 2.00 m | -19.5 px | 59 mm | |
| 3.00 m | -25.2 px | 132 mm | practical limit |
| 5.00 m | -29.7 px | 367 mm | disparity barely changes with distance |

**Work between 0.3 m and 3 m.** The far limit is set by the 60 mm baseline:
disparity asymptotes to -36.5 px at infinity, so past 3 m metres of distance
separate fractions of a pixel.

`2560x960` doubles the focal length in pixels and so halves every error,
reaching about 6 m, for 109 ms per frame instead of 16 ms.

## Getting better shapes

Measured on one frame, 640x480 per eye, `min_disparity=-48`:

| | Coverage | Time | Edge agreement |
|---|---|---|---|
| baseline (3WAY, block 9) | 51.5% | 31 ms | 18.9 |
| full 8-direction (HH) | 50.9% | 141 ms | 18.8 |
| block size 15 | 55.8% | 31 ms | 19.4 |
| uniqueness 5 | 56.5% | 31 ms | 18.0 |
| **contrast equalisation (CLAHE)** | 49.0% | 16 ms | **34.5** |

"Edge agreement" is how strongly depth discontinuities coincide with image
edges — whether shapes have the right outline.

`equalise_contrast` is on by default: it roughly doubles edge agreement for
no extra time, and leaves distances untouched (identical disparity measured
with and without). It costs about 3 points of coverage, trading ambiguous
pixels for sharper boundaries.

Full 8-direction mode is not worth 4.5x the time here. Raising coverage by
relaxing `uniqueness_ratio` makes shapes worse, not better: the extra pixels
are the uncertain ones.

Coverage stays near 50% because textureless surfaces — plain walls, smooth
floor — cannot be matched at all. That is inherent to block matching. A
projected IR pattern is how the RealSense and Kinect avoid it, and this
camera has none.

## For SLAM

Two caveats before building on this.

**Rolling shutter.** Cheap USB cameras expose row by row, so the image skews
while the robot moves and the geometry SLAM depends on is no longer a single
instant. This hurts far more than depth accuracy. Worth measuring before
committing: wave the camera and look for slanted verticals.

**No IMU and a short baseline.** 60 mm and a 3 m horizon suit indoor and
close-range outdoor work. Outdoors at speed there is little parallax to track.

Given those, feature-based stereo SLAM (ORB-SLAM3 in stereo mode, or RTAB-Map
which takes stereo or RGB-D) is the better fit than anything depending on
dense depth: it uses sparse keypoints, which the toed-in geometry handles
correctly once calibrated, and does not care about the 50% dense coverage.
Feed dense depth to occupancy mapping, not to pose estimation.
