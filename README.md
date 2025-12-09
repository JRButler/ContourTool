Geodesic contour extraction using fast marching method.
Computes distance maps from source points and extracts isolines.

Original project here: https://github.com/krummrey/contour-drawing/blob/main/contour-drawing.py
inspiration and thanks to https://www.reddit.com/user/docricky/ and https://www.reddit.com/user/krummrey/
for sharing Fast Marching Algorithm info

input:
  image file (jpg, png etc)
  
output:
  svg file with contour path segments

example run commands:
  input.jpg -o output.svg
  input.jpg --source 100,200 --source 300,400 --num 50
  input.jpg --source 100,200,300,400,500,600 --num 50
  input.jpg --gamma 1.5 --blur 2.0 --thickness 2
  input.jpg --dark-boost 2.0 --bright-cut 0.7 --num 120

this version contains fixes and updates:

- addressed matplotlib QuadContourSet issue. Using cs.get_paths() instead of .collections
- added `smooth` parameter for smoothing output lines
- addressed matplotlib MOVETO - now split_path_at_moves() segments lines to prevent unwanted segments
- swapped in optional scikit-fmm (C) library to speed up Fast Marching Algorithm execution ~20-50x faster that plain python

Before (original):

<img width="1115" height="1482" alt="contourTool_original" src="https://github.com/user-attachments/assets/7d864b04-17ee-46b0-a3f2-22e2d7754d88" />

After:

<img width="1115" height="1482" alt="contourTool_original_tweaked" src="https://github.com/user-attachments/assets/50d095c1-0a1c-4357-8902-3d99f40892eb" />
