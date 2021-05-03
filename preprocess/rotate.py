from PIL import Image, ExifTags
from glob import glob
import sys
import os

input_image_dir = sys.argv[1]

tags = {v: k for k, v in ExifTags.TAGS.items()}
orientation = tags["Orientation"]

# for each image, check exif tags and rotate accordingly
for f in glob(os.path.join(input_image_dir, "*")):

    try:
        image = Image.open(f)
        exif = image._getexif()
        rotated = False

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
            rotated = True
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
            rotated = True
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
            rotated = True

        if rotated:
            image.save(f)

        image.close()

    except (AttributeError, KeyError, IndexError) as e:
        print(e)