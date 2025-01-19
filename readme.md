A simple test script to experiment with pixel art downscaling.

It attempts to preserve relevant details on downscaling.

Suggested command lines:

For tiles/large features.
- `python pix.py <input>.png <output>.png -o 1,2 -s 2` 

For characters/fine details/lots of transparency:
- `python pix.py <input>.png <output>.png -o 2 -s 2` 

Running the command without parameters will provide help, 
of a sort, and I'll be filling in this readme as I refine things further.

TODO: Work on preserving outlines (e.g. as in the Yoshi example).
