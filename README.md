# How to prepare the dataset

## Download the dataset

- Navigate to this [site](https://public.roboflow.com/object-detection/pklot/1/download/coco).
- Select `Download zip to computer` and click Continue.
- Extract the contents from the .zip file.
- Move `valid`, `test`, and `train` folders to the location `data/PKLot/` in the repository.

## Run code

Go to the `code` folder:
- Run `preprocessing.py`.
- Run `modeling.py`.
- Run `testing-images_json.py` for testing the model predictions using individual test images with a .json file 
defining the bounding boxes of the parking spots; OR run `testing-video_mask.py` for testing using a video file
and a mask image for defining the bounding boxes.

## Notes

- I tried to do my best so that the paths to files where generic so anyone could run. If something is off, let me know.
