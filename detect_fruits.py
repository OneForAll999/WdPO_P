import json
from pathlib import Path
from typing import Dict
import numpy as np
import click
import cv2
from tqdm import tqdm


def detect_fruits(img_path: str) -> Dict[str, int]:
    """Fruit detection function, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each fruit.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    h, w, d = img.shape
    if h > w:
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)

    def nothing(x):
        pass

    cv2.namedWindow("Tracking")
    cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
    cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
    cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

    while True:

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos("LH", "Tracking")
        l_s = cv2.getTrackbarPos("LS", "Tracking")
        l_v = cv2.getTrackbarPos("LV", "Tracking")

        u_h = cv2.getTrackbarPos("UH", "Tracking")
        u_s = cv2.getTrackbarPos("US", "Tracking")
        u_v = cv2.getTrackbarPos("UV", "Tracking")

        l_b = np.array([l_h, l_s, l_v])
        u_b = np.array([u_h, u_s, u_v])

        mask = cv2.inRange(hsv, l_b, u_b)

        res = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow("frame", img)
        cv2.imshow("mask", mask)
        cv2.imshow("res", res)

        key = cv2.waitKey(1)
        if key == 27:
            L_HSV = l_b
            H_HSV = u_b
            break

    cv2.destroyAllWindows()

    # #Banan
    # H_l = 19
    # S_l = 100
    # V_l =0
    # H_A = 45
    # S_A = 255
    # V_A = 255
    # banan_lower = np.array([H_l, S_l, V_l], np.uint8)
    # banan_upper = np.array([H_A, S_A, V_A], np.uint8)
    # banan_mask = cv2.inRange(img, banan_lower, banan_upper)
    # output_orange = cv2.bitwise_and(img, img, mask=banan_mask)
    # cv2.imshow('orange', output_orange)
    # cv2.waitKey(0)
    # ORange
    banan_lower = np.array(L_HSV, np.uint8)
    banan_upper = np.array(H_HSV, np.uint8)
    banan_mask = cv2.inRange(img, banan_lower, banan_upper)
    output_orange = cv2.bitwise_and(img, img, mask=banan_mask)
    cv2.imshow('orange', output_orange)
    cv2.waitKey(0)

    # TODO: Implement detection method.

    apple = 0
    banana = 0
    orange = 0

    return {'apple': apple, 'banana': banana, 'orange': orange}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False,
                                                                                  path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path),
              required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect_fruits(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
