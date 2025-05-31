A simple test script to experiment with pixel art downscaling.

It attempts to preserve relevant details on downscaling.

Suggested command lines:

For tiles/large features.
- `uv run pix.py <input>.png <output>.png -o 1,2 -s 2` 

For characters/fine details/lots of transparency:
- `uv run pix.py <input>.png <output>.png -o 2 -s 2` 

For streamlit run: 

To run the Streamlit application, make sure you Streamlit and Streamlit Image Comparison installed. 

pip install streamlit
pip install streamlit-image-comparison

Then navigate to the project directory in your terminal and execute the following command:

streamlit run streamlit_main.py

Running the command without parameters will provide help, 
of a sort, and I'll be filling in this readme as I refine things further.

TODO: Work on preserving outlines (e.g. as in the Yoshi example).
