import cv2
import numpy as np
import math

for i in range(1, 9):
    img_path = f'photos/tray{i}.jpg'
    img = cv2.imread(img_path)

    blur = cv2.medianBlur(img, 3)

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 500, 650, apertureSize=5)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 90, minLineLength=50, maxLineGap=5)

    x_coords = []
    y_coords = []

    for line in lines:
        x1 = line[0][0]
        y1 = line[0][1]

        x_coords.append(x1)
        y_coords.append(y1)

    low_x = min(x_coords)
    high_x = max(x_coords)
    low_y = min(y_coords)
    high_y = max(y_coords)

    cv2.rectangle(img, (low_x, low_y), (high_x, high_y), (0, 0, 0), 5)

    rect_area = (high_x - low_x) * (high_y - low_y)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10, param1=100, param2=40, minRadius=20, maxRadius=40)

    circles = np.uint16(np.around(circles))

    print(circles)

    inside5zl = 0
    outside5zl = 0
    inside5gr = 0
    outside5gr = 0
    total_5zl_area = 0

    for circle in circles[0]:
        x, y, r = circle
        area = math.pi * (r ** 2)
        ra = 31
        if i == 1:
            ra = 33

        is_inside = low_x < x < high_x and low_y < y < high_y

        if r > ra:
            coin_type = '5zl'
            color = (100, 0, 255)
            total_5zl_area += area
        else:
            coin_type = '5gr'
            color = (255, 0, 0)

        if is_inside:
            if coin_type == '5zl':
                inside5zl += 1
            else:
                inside5gr += 1
            center_color = (0, 255, 0)
        else:
            if coin_type == '5zl':
                outside5zl += 1
            else:
                outside5gr += 1
            center_color = (0, 0, 255)

        cv2.circle(img, (x, y), r, color, 2)
        cv2.circle(img, (x, y), 3, center_color, -1)
        print(f"{coin_type} coin at ({x}, {y}) with radius {r} is {area:.2f} square pixels.")

    average_5zl_area = total_5zl_area / inside5zl + outside5zl
    average_difference = (rect_area - average_5zl_area) / rect_area * 100

    inside_t = inside5zl * 5 + inside5gr * 0.05
    outside_t = outside5zl * 5 + outside5gr * 0.05
    total = inside_t + outside_t

    print('Tootal 5zl:', inside5zl + outside5zl)
    print('Tootal 0.05zl:', inside5gr + outside5gr)
    print('Rectangle area', rect_area)
    print(f'How smaller are avarage 5zl than rectangle: {average_difference:.2f}%')
    print('5zl Inside:', inside5zl)
    print('5zl Outside:', outside5zl)
    print('0.05zl Inside:', inside5gr)
    print('0.05zl Outside:', outside5gr)
    print('Total inside:', inside_t, 'zl')
    print('Total outside:', outside_t, 'zl')
    print(f'Total: {total:.2f} zl')

    cv2.imshow(f'img {i}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()