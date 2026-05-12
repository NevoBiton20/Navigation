## Algorithm Workflow

```
RINEX 4 input
    ->
GPS pseudorange extraction
    ->
SP3 orbit download
    ->
SP3 satellite position parsing
    ->
Pseudorange plausibility filtering
    ->
SNR-based filtering
    ->
Signal transmission-time computation
    ->
Lagrange interpolation of satellite position
    ->
Earth-rotation correction
    ->
Elevation-mask filtering
    ->
Satellite clock and atmospheric corrections
    ->
Pseudorange equation construction
    ->
Iterative least-squares positioning
    ->
Minimum-satellites validation
    ->
Residual-based outlier rejection
    ->
PDOP geometry validation
    ->
Earth-surface sanity validation
    ->
Inter-epoch movement validation
    ->
ECEF to latitude/longitude/height conversion
    ->
Velocity estimation
    ->
CSV + KML export
```
