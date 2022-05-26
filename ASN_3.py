import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

img_color = cv2.imread("/Users/admin/Downloads/ASN3/Frame0064.png", cv2.IMREAD_COLOR)
img_unchanged = cv2.imread("/Users/admin/Downloads/ASN3/Frame0064.png", cv2.IMREAD_UNCHANGED)
cv2.imshow('color image', img_color)
cv2.waitKey(0)
cv2.imshow('unchanged image', img_unchanged)
cv2.waitKey(0)
Ig = cv2.imread("/Users/admin/Downloads/ASN3/Frame0064.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow('grayscale', Ig)
cv2.waitKey(0)
filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])


def convolution(image, kernel, average=False, verbose=False):
    if len(image.shape) == 3:
        print("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))
    print("Kernel Shape : {}".format(kernel.shape))

    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Image")
        plt.show()
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
    output = np.zeros(image.shape)
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height,
    pad_width:padded_image.shape[1] - pad_width] = image
    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image")
        plt.show()
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]
    print("Output Image size : {}".format(output.shape))
    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}X{} Kernel".format(kernel_row,
                                                           kernel_col))
        plt.show()
    return output


def sobel_edge_detection(image, filter, verbose=False):
    new_image_x = convolution(image, filter, verbose)

    if verbose:
        plt.imshow(new_image_x, cmap='gray')
        plt.title("Horizontal Edge")
        plt.show()
    new_image_y = convolution(image, np.flip(filter.T, axis=0), verbose)
    if verbose:
        plt.imshow(new_image_y, cmap='gray')
        plt.title("Vertical Edge")
        plt.show()
    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    print("|Gx| " + str(np.linalg.norm(new_image_x)))
    print("|Gy| " + str(np.linalg.norm(new_image_y)))
    print("G " + str(gradient_magnitude))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    if verbose:
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title("Gradient Magnitude")
        plt.show()
    return gradient_magnitude

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="Path", default="/Users/admin/Downloads/ASN3/Frame0064.png")
    args = vars(ap.parse_args())
    image = cv2.imread(args["image"])
    sobel_edge_detection(image, filter, verbose=True)
    img1 = cv2.imread("/Users/admin/Downloads/ASN3/Frame0064.png", )
    cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    cv2.imshow("rgb", img1)
    blue_channel = img1[:, :, 0]
    green_channel = img1[:, :, 1]
    red_channel = img1[:, :, 2]
    cv2.imshow('blue', blue_channel)
    cv2.waitKey(0)
    cv2.imshow('green', green_channel)
    cv2.waitKey(0)
    cv2.imshow('red', red_channel)
    cv2.waitKey(0)
    hsvImg = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv', hsvImg)
    cv2.waitKey(0)

    cap = cv2.VideoCapture('/Users/admin/Downloads/ASN3/Vid.mp4')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    r1 = cv2.VideoWriter('SimpleColorThresholder.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, size)
    while (True):
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lower_orange = np.array([153, 51, 0])
            upper_orange = np.array([255, 239, 213])
            mask = cv2.inRange(rgb, lower_orange, upper_orange)
            res_with_color = cv2.bitwise_and(frame, frame, mask=mask)
            cv2.imshow('mask_rgb', mask)
            cv2.imshow('mask_rgb_color', res_with_color)
            f2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            r1.write(f2)
        else:
            break
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    r1.release()
    cap = cv2.VideoCapture('/Users/admin/Downloads/ASN3/Vid.mp4')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    r2 = cv2.VideoWriter('GaussianColorThresholder.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, size)
    while (True):
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lower_orange = np.array([153, 51, 0])
            upper_orange = np.array([255, 239, 213])
            mask = cv2.inRange(rgb, lower_orange, upper_orange)
            res_with_color = cv2.bitwise_and(frame, frame, mask=mask)
            height, width, _ = res_with_color.shape
            m = []
            for i in range(height):
                for j in range(width):
                    if list(res_with_color[i, j]) != [0, 0, 0]:
                        m.append(list(res_with_color[i, j]))
                        m = np.array(m)
            mean = np.mean(m)
            cov = np.cov(m)
            for i in range(height):
                for j in range(width):
                    if list(res_with_color[i, j]) != [0, 0, 0]:
                        res_with_color[i, j] = numpy.asarray(
                            [(np.multiply((1 / (2 * np.pi * cov)), np.exp(-0.5 *
                                                                          (np.asarray((list(res_with_color[i, j]) - np.asarray([mean, mean, mean]))).transpose()) * (np.asarray((list(res_with_color[i, j]) - np.asarray([mean, mean, mean]))).transpose())))) * 255,
                             (np.multiply((1 / (2 * np.pi * cov)), np.exp(-0.5 * (np.asarray((list(res_with_color[i, j]) - np.asarray([mean, mean, mean]))).transpose()) * (np.asarray((list(res_with_color[i, j]) -
                                                                                                                                                                                        np.asarray([mean, mean, mean]))).transpose())))) * 255,
                             (np.multiply((1 / (2 * np.pi * cov)), np.exp(-0.5 *
                                                                          (np.asarray((list(res_with_color[i, j]) - np.asarray([mean, mean, mean]))).transpose()) * (np.asarray((list(res_with_color[i, j]) - np.asarray([mean, mean, mean]))).transpose())))) * 255])
            f2 = cv2.cvtColor(res_with_color, cv2.COLOR_GRAY2RGB)
            r2.write(f2)
        else:
            break
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    r2.release()
    cap = cv2.VideoCapture('/Users/admin/Downloads/ASN3/Vid.mp4')
    while (True):
        ret, frame = cap.read()
        if ret:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_orange = np.array([10, 100, 20])
            upper_orange = np.array([25, 255, 255])
            mask = cv2.inRange(hsv, lower_orange, upper_orange)
            res_with_color = cv2.bitwise_and(frame, frame, mask=mask)
            cv2.imshow('mask_hsv', mask)
            cv2.imshow('mask_hsv_color', res_with_color)
        else:
            break
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
