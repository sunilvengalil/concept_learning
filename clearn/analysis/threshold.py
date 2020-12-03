import cv2 as cv
# from matplotlib import pyplot as plt
img = cv.imread('/Users/sunilv/Desktop/signature.png', 0)
ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
cv.imwrite( '/Users/sunilv/Desktop/signature_bw.png',thresh1)

# for i in xrange(6):
#     plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()