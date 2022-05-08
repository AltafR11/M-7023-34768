import cv2
img = cv2.imread('test1.jpg')
plt.imshow(img)
img.shape
# Converting to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray.shape
plt.imshow(gray)
# Resizing to a 28x28 image
# Please note my image was already in correct dimension
resized = cv2.resize(gray, (28,28), interpolation = cv2.INTER_AREA)
resized.shape
# 0-1 scaling
newimg = tf.keras.utils.normalize(resized, axis = 1)
# For kernal operations
newimg = np.array(newimg).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
newimg.shape
#making predictions
predictions = model.predict(newimg)
print(np.argmax(predictions[0]))
