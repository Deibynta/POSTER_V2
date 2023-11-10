from main import *
from deepface import DeepFace

# Checking for all types of devices available
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")
# Predicting the model
# def prediction(model, image_path):
model = pyramid_trans_expr2(img_size=224, num_classes=7)

model = torch.nn.DataParallel(model)
model = model.to(device)

model_path = "raf-db-model_best.pth"
image_arr = [
    "/Users/futuregadgetlab/Downloads/Testing/pexels-kailash-kumar-693791.jpg",  # angry
    "/Users/futuregadgetlab/Downloads/Testing/pexels-luna-lovegood-1104007.jpg",  # happy
    "/Users/futuregadgetlab/Downloads/Testing/pexels-anna-shvets-3771681.jpg",  # sad
    "/Users/futuregadgetlab/Downloads/Testing/pexels-pixabay-415229.jpg",  # sad
    "/Users/futuregadgetlab/Downloads/Testing/little-boy-crying-isolated-on-260nw-97198298.jpeg",  # sad
    "/Users/futuregadgetlab/Downloads/Testing/pexels-monstera-production-7114749.jpg",  # angry
    "/Users/futuregadgetlab/Downloads/Testing/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvay1wci1zMzAtdGVuLTAyMi1qb2I1MS1sLmpwZw.jpeg", #surprised
]


def main():
    if model_path is not None:
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path, map_location=device)
            best_acc = checkpoint["best_acc"]
            best_acc = best_acc.to()
            print(f"best_acc:{best_acc}")
            model.load_state_dict(checkpoint["state_dict"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    model_path, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(model_path))
        predict(model, image_path=image_arr)
        return


def predict(model, image_path):
    from face_detection import face_detection

    with torch.no_grad():
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(p=1, scale=(0.05, 0.05)),
            ]
        )

        for image_file in image_arr:
            face = face_detection(image_file)
            image_tensor = transform(face).unsqueeze(0)
            image_tensor = image_tensor.to(device)

            model.eval()
            img_pred = model(image_tensor)
            topk = (1,)
            with torch.no_grad():
                maxk = max(topk)
                # batch_size = target.size(0)
                _, pred = img_pred.topk(maxk, 1, True, True)
                pred = pred.t()

            img_pred = pred
            img_pred = img_pred.squeeze().cpu().numpy()
            im_pre_label = np.array(img_pred)
            y_pred = im_pre_label.flatten()
            emotions = {
                0: "Surprise",
                1: "Fear",
                2: "Disgust",
                3: "Happy",
                4: "Sad",
                5: "Angry",
                6: "Neutral",
            }
            labels = []
            for i in y_pred:
                labels.append(emotions.get(i))

            print(f"    [!] The predicted labels are {y_pred} and the label is {labels}")
    return


if __name__ == "__main__":
    main()
