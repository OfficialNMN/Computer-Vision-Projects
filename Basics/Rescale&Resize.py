import cv2


def rescaleframe(frame,scale=0.5):
    # Works for Images, Videos and Live Video
    # whereas (.set) function works only for live videos
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimensions=(height,width)

    return cv2.resize(frame,dimensions,interpolation=cv2.INTER_AREA)

# Rescaling Image

img=cv2.imread('pics/Landscape.PNG')
resized_image =rescaleframe(img)
cv2.imshow('resized',resized_image)
cv2.waitKey(0)


# Rescaling Videos

capture=cv2.VideoCapture('dog.mp4')

while True:
    isTrue,frame=capture.read()
    frame_resized=rescaleframe(frame)

    cv2.imshow('Video Resized',frame_resized)
    cv2.imshow('Video',frame)

    if cv2.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv2.destroyAllWindows()


