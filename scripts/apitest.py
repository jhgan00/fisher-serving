import os
import requests
import cv2


if __name__ == "__main__":

    img_dir = "/home/fisher/DATA/AI-HUB/fish_activity_captured_image_data/Validation/[원천]validation/of"
    img_fpath = "swim_of_2021-01-18-09-00_00-36-43_275.jpg"

    res = requests.post(
        'http://127.0.0.1:12000/disease',
        json={"img_fpath": img_fpath}
    )
    result = res.json()

    img = cv2.imread(os.path.join(img_dir, img_fpath))

    for bbox, kp in zip(result['bboxes'], result['keypoints']):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        x0_0, y0_0, x0_1, y0_1, x1_0, y1_0, x1_1, y1_1 = kp

        cv2.line(img, (x0_0, y0_0), (x0_1, y0_1), (0, 0, 255), 3)
        cv2.line(img, (x1_0, y1_0), (x1_1, y1_1), (0, 0, 255), 3)

        cv2.circle(img, (x0_0, y0_0), 10, (255, 0, 0), -1)
        cv2.circle(img, (x0_1, y0_1), 10, (0, 255, 0), -1)
        cv2.circle(img, (x1_0, y1_0), 10, (255, 0, 255), -1)
        cv2.circle(img, (x1_1, y1_1), 10, (0, 255, 255), -1)

    cv2.imwrite("bbox.png", img)
