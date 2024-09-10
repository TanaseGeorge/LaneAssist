import cv2
import numpy as np

cam = cv2.VideoCapture("Lane_Detection_Test_Video_01.mp4")

positions = [
        (100, 0),
        (480 + 1, 0),
        (2 * (300 + 1), 0),
        (3 * (370 + 1), 0),
        (0, 300 + 1),
        (400 + 1, 300 + 1),
        (2 * (400 + 1), 300 + 1),
        (3 * (400 + 1), 300 + 1),
    ]

def showFrame(name, frame, offset):
    cv2.namedWindow(name)
    cv2.moveWindow(name, offset[0], offset[1])
    cv2.imshow(name, frame)

while True:

    ret, frame = cam.read()

    if ret is False:
        break

    #ex2
    frame = cv2.resize(frame, (400, 200))
    showFrame('Resized', frame, positions[0])

    #ex3
    grayscale_frame = np.zeros((200, 400), dtype=np.uint8)
    black_frame = np.zeros((200, 400), dtype = np.uint8)
    for i in range(200):
        for j in range(400):
            b, g, r = frame[i, j]
            grayscale_value = int((int(b) + int(g) + int(r)) / 3)
            grayscale_frame[i, j] = grayscale_value

    #ex4

    height = 200
    width = 400

    upper_left = (int(width * 0.44), int(height * 0.8))
    upper_right = (int(width * 0.56), int(height * 0.8))
    lower_left = (0, height)
    lower_right = (width, height)
    trapezoid_points = np.array([upper_left, upper_right, lower_right, lower_left], dtype='int32')
    trapezoid_frame = np.zeros((height, width), dtype='uint8')
    trapezoid_frame = cv2.fillConvexPoly(trapezoid_frame, points=trapezoid_points, color=1)
    trapezoid_frame_3ch = cv2.merge((trapezoid_frame, trapezoid_frame, trapezoid_frame))
    showFrame('Trapezoid', trapezoid_frame * 255, positions[2])
    trapezoid_frame = trapezoid_frame_3ch * frame
    showFrame('Road', trapezoid_frame, positions[3])

    #ex 5
    screen_points = np.array([(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)], dtype='float32')
    magical_matrix = cv2.getPerspectiveTransform(np.float32(trapezoid_points), screen_points)
    stretched_trapezoid_frame = cv2.warpPerspective(trapezoid_frame, magical_matrix, (400, 200))
    showFrame('Top-Down', stretched_trapezoid_frame, positions[4])

    #ex 6
    frame = cv2.blur(stretched_trapezoid_frame, ksize=(7, 5))
    showFrame('Blur', stretched_trapezoid_frame, positions[5])

    #ex 7
    sobel_vertical = np.float32([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    sobel_horizontal = np.transpose(sobel_vertical)

    frame_1 = cv2.filter2D(np.float32(frame), -1, sobel_vertical)
    frame_2 = cv2.filter2D(np.float32(frame), -1, sobel_horizontal)
    sobel_frame = np.sqrt((frame_1 * frame_1) + (frame_2 * frame_2))

    frame = cv2.convertScaleAbs(sobel_frame)
    showFrame('Sobel', frame, positions[6])

    # Ex 8
    frame = cv2.threshold(frame, int(255 / 2), 255, cv2.THRESH_BINARY)[1]
    showFrame('Binarized', frame, positions[7])

    # Ex 9
    copy_frame = frame.copy()
    num_cols_to_remove = int(width * 0.05)
    copy_frame[:, :num_cols_to_remove] = 0
    copy_frame[:, -num_cols_to_remove:] = 0

    half = int(width / 2)
    first_half = copy_frame[:, :half]
    second_half = copy_frame[:, half:]

    left_y = np.where(first_half == 255)
    left_x = np.where(first_half == 255)

    right_y = np.where(second_half == 255)
    right_x = np.where(second_half == 255)
    right_x = (right_x[0] + half, right_x[1])

    #cv2.imshow('Gray', grayscale_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()