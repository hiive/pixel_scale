A simple test script to experiment with pixel art downscaling.

It attempts to preserve relevant details on downscaling.

Suggested command lines:

For tiles/large features.
- `uv run pix.py <input>.png <output>.png -o 1,2 -s 2` 

For characters/fine details/lots of transparency:
- `uv run pix.py <input>.png <output>.png -o 2 -s 2` 

Running the command without parameters will provide help, 
of a sort, and I'll be filling in this readme as I refine things further.

To run the Streamlit application, make sure you install uv!

Then navigate to the project directory in your terminal and execute the following commands:

- `uv pip install -r pyproject.toml`

- `uv run streamlit run streamlit_main.py`

TODO: Work on preserving outlines (e.g. as in the Yoshi example).
