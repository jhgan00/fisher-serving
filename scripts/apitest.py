import cv2
import requests
import logging
import time
import glob
import os


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    HOST = os.environ.get("HOST", "127.0.0.1")
    PORT = os.environ.get("PORT", 12000)

    if not os.path.exists("response"):
        os.mkdir("./response")

    frames = glob.glob("./resources/sample/*")[:50]

    # Capture frame-by-frame
    for i, frame in enumerate(frames):
        start = time.time()
        response = requests.post(f"http://{HOST}:{PORT}/detection", files={"file": open(frame, 'rb')})
        end = time.time()
        elapsed = end - start
        logging.info(f"took {elapsed:.4f} sec")

        if not response.ok:
            logging.error(response.text)
        data = response.json()
        print(data)

        frame = cv2.imread(frame)
        for label, (xmin, ymin, xmax, ymax) in zip(data['diseases'], data['bboxes']):
            color = (36, 255, 12) if not label else (36, 36, 255)
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 5)
            if label:
                cv2.putText(frame, "Abnormal", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)
        cv2.imwrite(f"response/result.{i}.jpg", frame)
