import os
import numpy as np
import cv2
# DEPTHAI
import depthai as dai


filename = 'video.avi' # .avi .mp4
frames_per_seconds = 24.0
my_res = '720p' # 1080p

# Set resolution for the video capture
# Function adapted from https://kirr.co/0l6qmh
#def change_res(cap, width, height):
#    cap.set(3, width)
#    cap.set(4, height)

# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS['720p']
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    change_res(cap, width, height)
    return width, height


# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

fps = 10
frames_per_seconds = 10
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")

# Properties
# camRgb.initialControl.setManualFocus(100)
# camRgb.setPreviewKeepAspectRatio(False)
camRgb.setPreviewSize(1280, 720)
# camRgb.setPreviewSize(1280, 720)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
#camRgb.setIspScale(2, 3)
camRgb.setFps(fps)

# Linking
camRgb.preview.link(xoutRgb.input)

#cap = cv2.VideoCapture(0)
#dims = get_dims(cap, res=my_res)
video_type_cv2 = get_video_type(filename)
save_path = os.path.join('', filename)

out = cv2.VideoWriter(save_path, video_type_cv2, frames_per_seconds, (1280, 720))

with dai.Device(pipeline) as device:
    print('Connected cameras: ', device.getConnectedCameras())
    # Print out usb speed
    print('Usb speed: ', device.getUsbSpeed().name)
    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    while True:
        # Capture frame-by-frame
        inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived
        # Retrieve 'bgr' (opencv format) frame
        # cv2.imshow("rgb", inRgb.getCvFrame())
        #print("here2")
        frame = inRgb.getCvFrame()
        frame = cv2.rotate(frame, cv2.ROTATE_180)  # Rotates the video 180 degrees
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #print("here1")
        out.write(frame)
        # Display the resulting frame
        cv2.imshow('frame',frame)
        #print("here")
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

# When everything done, release the capture
#cap.release()
out.release()
cv2.destroyAllWindows()