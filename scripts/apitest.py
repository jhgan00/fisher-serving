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

    frames = glob.glob("../resources/sample/*")[:50]
    print(frames)
    frames = [cv2.imencode(".bmp", cv2.resize(cv2.imread(frame), (640, 640)))[1].tobytes() for frame in frames]

    # Capture frame-by-frame
    for frame in frames:
        start = time.time()
        response = requests.post("http://127.0.0.1:8080/detection", files={"file": frame})
        end = time.time()
        elapsed = end - start
        logging.info(f"took {elapsed:.4f} sec")
        if not response.ok:
            break
        data = response.json()
        print(data)

        # for label, (xmin, ymin, xmax, ymax) in zip(data['label'], data['bbox']):
        #     frame = cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (36, 255, 12), 5)
        #     cv.putText(frame, label, (xmin, ymin - 10), cv.FONT_HERSHEY_SIMPLEX, 3, (36, 255, 12), 5)
        # cv.imshow("frame", frame)
        # if cv.waitKey(1) == ord('q'):
        #     break

    # When everything done, release the capture
    # capture.release()
    # cv.destroyAllWindows()
