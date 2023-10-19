# crop-pad-fix
Initial fix for text detection using docTR v2 for documents with smaller words. This fix generates a cropped image by generating a meta bounding box for initial detections. Text Detection model is again run on the cropped image and the final bboxes are scaled back to the original page dimensions
