import dlib
import numpy as np
from imutils import face_utils
import config


img = np.load(config.image_cache_path)

fd = dlib.get_frontal_face_detector()
faces = fd(img, 0)
face = faces[0]

pd = dlib.shape_predictor(config.mod)
img_feature = pd(img, face)
img_feature = face_utils.shape_to_np(img_feature)

np.save(config.face_return_path, img_feature)
