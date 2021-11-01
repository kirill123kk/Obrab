import numpy as np
import cv2 as cv
import sys

def gnil (image) :

    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
    # image_blur = cv.erode(image,kernel)
    image_blur = cv.blur(image, (7,7)) 
    norm_img1 = cv.normalize(image_blur, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    norm_img1 = (255*norm_img1).astype(np.uint8)

    rgb_planes = cv.split(image_blur)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv.medianBlur(dilated_img, 21)
        diff_img = 255 - cv.absdiff(plane, bg_img)
        norm_img = cv.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    result = cv.merge(result_planes)
    result_norm = cv.merge(result_norm_planes)
    hsv_img = cv.cvtColor( image_blur , cv.COLOR_BGR2HSV)

    markers = np.zeros((image.shape[0], image.shape[1]) , dtype = "int32" )
    markers [ 100 : 140 , 100 : 140 ] = 255
    markers [ 236 : 255 , 0 : 20 ] = 1
    markers [ 0 : 20 , 0 : 20 ] = 1
    markers [ 0 : 20 , 236 : 255 ] = 1
    markers [ 236 : 255 , 236 : 255 ] = 1
    leafs_area_BGR = cv.watershed(image_blur, markers)
    healthy_part = cv.inRange(hsv_img, (36,25,25), (86,255,255))
    ill_part = leafs_area_BGR - healthy_part
    mask = np.zeros_like(image, np.uint8)
    mask [leafs_area_BGR > 1] = ( 102, 255, 102)
    mask [ill_part > 1 ] = (250, 0, 0)

    cv.imshow("hsv_img", hsv_img)
    cv.imshow("Blured", image_blur)
    return mask
def main():
    img = cv.imread('12.jpg')
    if img is None:
     sys.exit("Could not read the image.")
    cv.imshow("Origin", img)

    resimg = gnil(img)
    cv.imshow("end", resimg)
    k = cv.waitKey(0)
if __name__ == '__main__':
    main()