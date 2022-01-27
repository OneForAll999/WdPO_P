import json
from pathlib import Path
from typing import Dict
import numpy as np
import click
import cv2
from tqdm import tqdm


#komentarz
#jajebie

def HSV_IMG(img):
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

        # cv2.imshow("frame", img)
        # cv2.imshow("mask", mask)
        cv2.imshow("res", res)

        key = cv2.waitKey(1)
        if key == 27:
            L_HSV = l_b
            H_HSV = u_b
            break

    cv2.destroyAllWindows()


def detect_fruits(img_path: str) -> Dict[str, int]:
    # TODO: Implement detection method.

    apple = 0
    banana = 0
    orange = 0
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

    def detect_oranges():
        # orange
        img_orange = cv2.imread(img_path)
        img_orange = cv2.cvtColor(img_orange, cv2.COLOR_BGR2HSV)
        img_orange = cv2.resize(img_orange, (0, 0), fx=0.3, fy=0.3)
        img_orange = cv2.GaussianBlur(img_orange, (181, 181), 0)

        h, w, d = img_orange.shape
        if h > w:
            img_orange = cv2.rotate(img_orange, cv2.cv2.ROTATE_90_CLOCKWISE)

        # HSV_IMG(img_orange)
        hsv = cv2.cvtColor(img_orange, cv2.COLOR_BGR2HSV)
        l_b = np.array([17, 231, 219])
        u_b = np.array([255, 255, 255])

        mask = cv2.inRange(hsv, l_b, u_b)
        kernel = np.ones((29, 29), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        res = cv2.bitwise_and(img_orange, img_orange, mask=mask)
        res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        to_show = res
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(res, 30, 255, 0)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(to_show, contours, -1, (0, 255, 0), 5)
        var = len(contours)
        orange = var
        return orange

    def detect_bananas():
        img_banana = cv2.imread(img_path)
        img_banana = cv2.resize(img_banana, (0, 0), fx=0.3, fy=0.3)
        img_banana = cv2.GaussianBlur(img_banana, (181, 181), 0)

        h, w, d = img_banana.shape
        if h > w:
            img_banana = cv2.rotate(img_banana, cv2.cv2.ROTATE_90_CLOCKWISE)
        # HSV_IMG(img_banana)#Banan
        img_banana = cv2.cvtColor(img_banana, cv2.COLOR_BGR2HSV)
        l_b = np.array([21, 104, 0])
        u_b = np.array([26, 255, 255])
        mask = cv2.inRange(img_banana, l_b, u_b)
        kernel = np.ones((31, 31), np.uint8)
        banana_mask = cv2.dilate(mask, kernel, iterations=1)
        # banana_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        output_banana = cv2.bitwise_and(img_banana, img_banana, mask=banana_mask)
        output_banana = cv2.cvtColor(output_banana, cv2.COLOR_HSV2BGR)
        res = cv2.cvtColor(output_banana, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(res,123, 255, 0)
        to_show = res
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        real_number_of_ctr = 0
        for contour in contours:
            if cv2.contourArea(contour) > 30000:
                cv2.drawContours(output_banana, contour, -1, (255, 255, 255), 3)
                real_number_of_ctr = real_number_of_ctr + 1
        return real_number_of_ctr

    def detect_apples():
        img_apples = cv2.imread(img_path)
        img_apples = cv2.resize(img_apples, (0, 0), fx=0.3, fy=0.3)
        img_apples = cv2.GaussianBlur(img_apples, (11,11), 0)
        # img_apples = cv2.blur(img_apples, (9, 9))

        h, w, d = img_apples.shape
        if h > w:
            img_apples = cv2.rotate(img_apples, cv2.cv2.ROTATE_90_CLOCKWISE)
        # HSV_IMG(img_apples)
        img_apples = cv2.cvtColor(img_apples, cv2.COLOR_BGR2HSV)
        l_b = np.array([0, 135, 0])
        u_b = np.array([205, 255, 148])
        mask = cv2.inRange(img_apples, l_b, u_b)
        kernel = np.ones((51,51), np.uint8)
        apple_mask = cv2.dilate(mask, kernel, iterations=1)
        apple_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        output_apples = cv2.bitwise_and(img_apples, img_apples, mask=apple_mask)
        output_apples = cv2.cvtColor(output_apples, cv2.COLOR_HSV2BGR)
        cv2.imshow('a',output_apples)
        cv2.waitKey(0)


    # ORange
    orange = detect_oranges()
    banana = detect_bananas()
    # detect_apples()
    print(f' \n Oranges: {orange}, Bananas: {banana} \n')
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
