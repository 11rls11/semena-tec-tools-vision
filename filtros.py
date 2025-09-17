import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pytesseract

imgs = {
    "placa_1" : cv.imread("Images/placa_1.jpg"),
    "placa_2" : cv.imread("Images/placa_2.jpg"),
    "placa_3" : cv.imread("Images/placa_3.jpg"),
    "placa_4" : cv.imread("Images/placa_4.jpg")
}

resultsLaplacian = {}

resultsSobelX = {}

resultsSobelY = {}

resultsMask = {}

resultsCleared = {}

lower = np.array([0, 0, 0])
upper = np.array([180, 255, 70])
kernel = np.ones((5, 5), np.uint8)

for key, imagen in imgs.items():
    src = cv.cvtColor(imagen, cv.COLOR_BGR2HSV)
    
    black_mask = cv.inRange(src, lower, upper)
    resultsMask[key] = black_mask

    src_gaussian = cv.GaussianBlur(black_mask, (5, 5), 0)    

    src_eroded = cv.erode(src_gaussian, kernel, iterations=1)

    src_dilated = cv.dilate(src_eroded, kernel, iterations=2)

    src_cleared = cv.bitwise_not(src_dilated)
    resultsCleared[key] = src_cleared

    laplacian = cv.Laplacian(src_dilated, cv.CV_64F)
    resultsLaplacian[key] = laplacian

    sobel_x = cv.Sobel(src_dilated, cv.CV_64F, 1, 0, ksize=5)
    resultsSobelX[key] = sobel_x

    sobel_y = cv.Sobel(src_dilated, cv.CV_64F, 0, 1, ksize=5)
    resultsSobelY[key] = sobel_y

set_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKMNOPQRSTUVWXYZ'
for key in imgs.keys():
    plt.figure(figsize=(14,8))
    
    # Imagen original
    plt.subplot(2,3,1)
    plt.title(f"Original {key}")
    plt.imshow(cv.cvtColor(imgs[key], cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Laplacian
    plt.subplot(2,3,2)
    plt.title(f"Laplacian {key}")
    plt.imshow(resultsLaplacian[key], cmap='gray')
    plt.axis('off')
    
    # Sobel X
    plt.subplot(2,3,3)
    plt.title(f"Sobel X {key}")
    plt.imshow(resultsSobelX[key], cmap='gray')
    plt.axis('off')
    
    # Sobel Y
    plt.subplot(2,3,4)
    plt.title(f"Sobel Y {key}")
    plt.imshow(resultsSobelY[key], cmap='gray')
    plt.axis('off')
    
    # Mascara
    plt.subplot(2,3,5)
    plt.title(f"Mascara")
    plt.imshow(resultsMask[key], cmap='gray')
    plt.axis('off')
    
    # Resultados
    plt.subplot(2,3,6)
    plt.title(f"Resultado")
    plt.imshow(resultsCleared[key], cmap='gray')
    plt.axis('off')

    plt.show()

    text = pytesseract.image_to_string(resultsCleared[key], config=set_config)
    print("Texto de la placa: ", text)