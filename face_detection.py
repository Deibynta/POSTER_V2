from deepface import DeepFace
import matplotlib.pyplot as plt
from PIL import Image


def face_detection(img_path):
    face_objs = DeepFace.extract_faces(
        img_path=img_path,
        target_size=(224, 224),
        detector_backend="mtcnn",
    )

    coordinates = face_objs[0]["facial_area"]
    image = Image.open(img_path)

    cropped_image = image.crop(
        (
            coordinates["x"],
            coordinates["y"],
            coordinates["x"] + coordinates["w"],
            coordinates["y"] + coordinates["h"],
        )
    )
    cropped_image.save("test.jpg")
    return cropped_image
