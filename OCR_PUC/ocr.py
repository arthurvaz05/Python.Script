import pytesseract
import numpy as np

convert -density 300 tb4.png -depth 8 -strip -background white -alpha off file.tiff

tesseract file.tiff output.txt