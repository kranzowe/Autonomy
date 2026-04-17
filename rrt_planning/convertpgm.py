from PIL import Image
import numpy as np

# Load the ASCII PGM
img = Image.open(r'src\Autonomy\rrt_planning\Course11edit.pgm')

# Save as binary PGM (P5 format)
img.save('Course11edit_binary.pgm')

# Or save as PNG (more universal)
img.save('Course11edit.png')

print("Conversion complete!")