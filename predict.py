import os
from ultralytics import YOLO
import cv2

IMG_DIR = os.path.join('.', 'images')

img_path = os.path.join(IMG_DIR, 'img.jpg')
img = cv2.imread(r'data/train/images/MicrosoftTeams-image-11-_png.rf.970a7046a5e7859f542e4d876d383043.jpg')
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')

threshold = 0.4

class_name_dict = {0: 'glove'}
directory = r'./images/'

model = YOLO(model_path)
result = model(r'data/train/images/MicrosoftTeams-image-11-_png.rf.970a7046a5e7859f542e4d876d383043.jpg')[0]
for result in result.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result
    if score > threshold:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(img, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
os.chdir(directory)
cv2.imwrite("predicted.jpg", img)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
