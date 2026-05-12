RINEX 4 Observation File
    ->
Parse GPS pseudorange measurements per epoch
    ->
Extract recording time range
    ->
Load local SP3 file OR auto-download matching SP3 with gnss-lib-py
    ->
Parse SP3 satellite ECEF positions
    ->
For each epoch, keep GPS-only observations
    ->
Pseudorange plausibility filter
    ->
SNR filter
    ->
For each satellite, compute signal travel time
    ->
Compute satellite transmission time
    ->
Interpolate satellite position at transmission time using Lagrange interpolation
    ->
Apply Earth rotation correction
    ->
Elevation mask filter
    ->
Apply satellite clock / atmospheric corrections
    ->
Build pseudorange equations
    ->
Solve receiver ECEF position + receiver clock bias using iterative least squares
    ->
Minimum satellite count check
    ->
Residual outlier rejection
    ->
PDOP geometry filter
    ->
Earth-surface sanity check
    ->
Inter-epoch jump filter
    ->
Warm-start protection
    ->
Convert ECEF position to latitude, longitude, and height
    ->
Estimate velocity from consecutive positions
    ->
Export CSV file
    ->
Export KML path for Google Earth visualization
