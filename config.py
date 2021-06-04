import os

basic_path = os.path.join("D:\\", "dlib-models")
image_cache_path = os.path.join(basic_path, "image.npy")
face_return_path = os.path.join(basic_path, "return.npy")
command_line = "python " + os.path.join(basic_path, "py36_dlib.py")
mod = os.path.join(basic_path, "shape_predictor_68_face_landmarks.dat")
