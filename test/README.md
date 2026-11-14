# Testing

You can compile and run the tests yourself by running:

```
make
```

To run a specific test configuration, run one of:

```
make c
make c_no_sse
make cpp
make cpp_no_sse
```

## Randomized property tests

The property suite covers matrix orthogonality, inverse round-trips,
quaternion interpolation endpoints, projection clip-space ranges, and
quaternion/matrix consistency:

```
make property
```

Each property runs 500,000 generated inputs by default, split evenly across
five input classes: `regular`, `near_singular`, `tiny`, `zero_vector`, and
`huge`. Every input is evaluated twice and the complete observations are
compared byte-for-byte. The `make property` target also runs the whole binary
twice and compares the transcripts, so the fixed-seed run is deterministic
across processes as well. A failing input is automatically shrunk to a
one-minimal scalar counterexample before it is printed.

The count can be changed for local iteration, but the default and CI path use
the required 500,000 cases per property:

```
make property PROPERTY_COUNT=10000
```

The suite can also be built through the C++ API:

```
make property_cpp
```
