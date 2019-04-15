import cv2
import numpy as np


def detect_faces(im):
    hc = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = hc.detectMultiScale(im, minSize=(30, 30))
    if len(faces) == 0:
        raise Exception('no faces')
    return faces


def canny_edges(im):
    edge_im = cv2.Canny(im, 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    closing_im = cv2.morphologyEx(edge_im, cv2.MORPH_CLOSE, kernel)
    return closing_im


def detect_contours(im):
    closing_im = canny_edges(im)
    ret, thresh = cv2.threshold(closing_im, 127, 255, 0)
    height, width = thresh.shape[:2]
    thresh[0:3, 0:width-1] = 255
    thresh[height-3:height-1, 0:width-1] = 255
    thresh[0:height-1, 0:3] = 255
    thresh[0:height-1, width-3:width-1] = 255
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def calc_face_histgram_normalized(im, faces, padding=0):
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    hsv_histgrams = []
    for i, face in enumerate(faces):
        origin_x, origin_y, width, height = face
        area = (height - padding*2)*(width - padding*2)
        roi_image = im_hsv[(origin_y + padding):(origin_y + height - padding), (origin_x + padding):(origin_x + width - padding)]
        cv2.imshow('win', roi_image)
        cv2.waitKey(1000)

        h_h = cv2.calcHist([roi_image], [0], None, [180], [0, 180], None, 0)
        h_s = cv2.calcHist([roi_image], [1], None, [256], [0, 256], None, 0)
        h_v = cv2.calcHist([roi_image], [2], None, [256], [0, 256], None, 0)

        # normalization and append
        if area == 0:
            area = 1
        hsv_histgrams.append([h_h/area, h_s/area, h_v/area])

    return hsv_histgrams


def mask_from_contour(im_shape, contour):
    mask = np.zeros(im_shape, np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    return mask

def mask_from_contours(im_shape, contours):
    mask = np.zeros(im_shape, np.uint8)
    for contour in contours:
        cv2.drawContours(mask, [contour], 0, 255, -1)
    return mask

def calc_contour_histgram_normalized(im, contour):
    mask = mask_from_contour(im.shape, contour)
    area = cv2.contourArea(contour)
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    im_mask = cv2.inRange(mask, 10, 255)

    h_h = cv2.calcHist([im_hsv], [0], im_mask, [180], [0, 180], None, 0)
    h_s = cv2.calcHist([im_hsv], [1], im_mask, [256], [0, 256], None, 0)
    h_v = cv2.calcHist([im_hsv], [2], im_mask, [256], [0, 256], None, 0)

    # normalization and return
    if area == 0:
        area = 1
    return [h_h/area, h_s/area, h_v/area]


def blown_skin_mask(im, mask, diff_color):
    height, width = im.shape[:2]
    hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    for y in range(height):
        for x in range(width):
            if (mask[y, x, 0]):
                h = hsv_im[y, x, 0] + diff_color[0]
                s = hsv_im[y, x, 1] + diff_color[1]
                v = hsv_im[y, x, 2] + diff_color[2]
                if h < 0:
                    hsv_im[y, x, 0] = h + 180
                elif h > 180:
                    hsv_im[y, x, 0] = h - 180
                else:
                    hsv_im[y, x, 0] = h
                if s < 0:
                    hsv_im[y, x, 1] = 0
                elif s > 255:
                    hsv_im[y, x, 1] = 255
                else:
                    hsv_im[y, x, 1] = s
                if v < 0:
                    hsv_im[y, x, 2] = 0
                elif v > 255:
                    hsv_im[y, x, 2] = 255
                else:
                    hsv_im[y, x, 2] = v
    return cv2.cvtColor(hsv_im, cv2.COLOR_HSV2BGR)


def calc_hsv_diff(skin_hist):
    blown_color = np.array([10, 70, 255])
    return blown_color - skin_color


def max_bin_histgram(hist):
    idx, _ = np.unravel_index(hist.argmax(), hist.shape)
    return idx


if __name__ == '__main__':
    thresh = 10
    image = cv2.imread('humantest.png')
    im_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    faces = detect_faces(image)
    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    cv2.imshow('face',image)
    
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    '''
    face_hists = calc_face_histgram_normalized(image, faces, padding=60)
    contours = detect_contours(image)
    skin_contours = []

    for index, contour in enumerate(contours):
        hists = calc_contour_histgram_normalized(image, contour)
        for face_hist in face_hists:
            h_dist = cv2.compareHist(face_hist[0], hists[0], 1)
            s_dist = cv2.compareHist(face_hist[1], hists[1], 1)
            v_dist = cv2.compareHist(face_hist[2], hists[2], 1)
            if abs(h_dist + s_dist + v_dist) < thresh:
                print(h_dist + s_dist + v_dist)
                print('is face')
                skin_contours.append(contour)

    skin_color = np.array([max_bin_histgram(face_hists[0][0]), max_bin_histgram(face_hists[0][1]), max_bin_histgram(face_hists[0][2])])
    diff_color = calc_hsv_diff(skin_color)
    print(diff_color)

    mask = mask_from_contours(image.shape, skin_contours)
    im_blown = blown_skin_mask(image, mask, diff_color)

    cv2.imshow('blown', cv2.medianBlur(im_blown, 3))
    cv2.waitKey(0)'''
    
