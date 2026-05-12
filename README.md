## Algorithm Workflow

```
RINEX 4 input
    Receives the raw GNSS observation file recorded by the receiver.
    ->
GPS pseudorange extraction
    Extracts GPS satellite measurements and keeps usable code pseudoranges such as C1C/C1W.
    ->
SP3 orbit download
    Downloads or loads the matching precise satellite orbit file according to the recording time.
    ->
SP3 satellite position parsing
    Reads satellite ECEF positions from the SP3 file and converts them to meters.
    ->
Pseudorange plausibility filtering
    Removes pseudorange measurements outside a physically reasonable GPS range.
    ->
SNR-based filtering
    Removes weak satellite signals that are likely to produce noisy measurements.
    ->
Signal transmission-time computation
    Estimates when each satellite transmitted its signal using pseudorange divided by light speed.
    ->
Lagrange interpolation of satellite position
    Computes the satellite position at the exact transmission time using multi-point interpolation.
    ->
Earth-rotation correction
    Corrects satellite position for Earth’s rotation during signal travel time.
    ->
Elevation-mask filtering
    Removes low-angle satellites that are more affected by atmosphere and multipath.
    ->
Satellite clock and atmospheric corrections
    Applies satellite clock correction and models signal delays through the atmosphere.
    ->
Pseudorange equation construction
    Builds one equation per satellite relating measured pseudorange to receiver position and clock bias.
    ->
Iterative least-squares positioning
    Solves receiver ECEF x, y, z and clock bias by minimizing pseudorange residuals.
    ->
Minimum-satellites validation
    Skips epochs with fewer than four valid satellites, since four unknowns must be solved.
    ->
Residual-based outlier rejection
    Removes satellites whose measured pseudorange does not fit the solved position well.
    ->
PDOP geometry validation
    Rejects epochs with poor satellite geometry that would produce unstable positions.
    ->
Earth-surface sanity validation
    Rejects solved positions that are physically far from Earth’s surface.
    ->
Inter-epoch movement validation
    Removes points that imply an unrealistic movement jump from the previous accepted point.
    ->
ECEF to latitude/longitude/height conversion
    Converts Cartesian receiver coordinates into WGS-84 latitude, longitude, and height.
    ->
Velocity estimation
    Estimates receiver velocity from differences between consecutive accepted positions.
    ->
CSV + KML export
    Writes numerical results to CSV and visualizes the reconstructed path in Google Earth.
```
