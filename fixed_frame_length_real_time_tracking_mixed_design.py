from pypylon import pylon
import cv2
import time
import pygame
import csv

fly_name = ("female1")

moviename = fly_name + ".mp4"
csv_name = fly_name+ ".csv"
plotname = fly_name+ ".pdf"


# connecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.ExposureTime.SetValue(8000)
# Set the desired frame rate
frame_rate = 20

time_list = []
velocity_list = []

# Set the duration of the video directly IN SECONDS
video_duration = 80

camera.AcquisitionFrameRateEnable.SetValue(True)
camera.AcquisitionFrameRate.SetValue(frame_rate)

# initialize background subtractor and bounding box variables
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100)

bbox = None

# initialize variables for calculating velocity
prev_center = None
prev_time = time.time()
velocity = None

writer = cv2.VideoWriter(moviename, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (1440,1080))

pygame.init()
# loads in the audio file
#pygame.mixer.music.load("/home/joeh/Documents/control_experiment/stimulus/mating_call_normal.wav")
pygame.mixer.music.load("/home/joeh/Documents/control_experiment/stimulus/5s_delay_500hz_one_min_break_5s_delay_mating_call.wav")
#pygame.mixer.music.load("/home/joeh/Documents/control_experiment/stimulus/700hz_normal.wav")

# converter for converting to OpenCV BGR format
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

start_time = time.time()
elapsed_time = 0

# start the frame count
frame_count = 0

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)



pygame.mixer.music.play()

# loop over camera frames
while camera.IsGrabbing() and elapsed_time < video_duration:
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    elapsed_time = time.time() - start_time


    if grabResult.GrabSucceeded():
        # access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()

        frame = img

    # apply background subtraction
    fgmask = fgbg.apply(img)
    
    # dilate the foreground mask to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)
    
    # find contours in the foreground mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    # loop over the contours to find the largest one
    max_area = 300
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    #set minimum dims
    min_width=69
    min_height=69
    max_width = 250
    max_height = 250

    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        if w >= min_width and h >= min_height:
        
            if w > max_width:
                w = max_width
            if h > max_height:
                h = max_height
        
            bbox = (x, y, w, h)

        # calculate velocity if previous center is available
        center = (x + w//2, y + h//2)
        if prev_center is not None:
            current_time = time.time()
            dt = current_time - prev_time
            dx = center[0] - prev_center[0]
            dy = center[1] - prev_center[1]
            velocity = (dx/dt, dy/dt)
            prev_time = current_time
        prev_center = center

            # add the time and velocity values to the lists
    #if velocity is not None:
    time_list.append(elapsed_time)
    velocity_list.append(cv2.norm(velocity))

    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

 
        


 
    # draw the bounding box and velocity on the image
    if bbox is not None:
        cv2.rectangle(img, bbox, (0, 255, 0), 2)
    #if velocity is not None:
        cv2.putText(img, "Velocity: {:.2f} pixels/sec".format(cv2.norm(velocity)), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Time: " + str(round(elapsed_time, 1)), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        writer.write(img)
    # display the image
    cv2.imshow("Motion Detection", img)
    cv2.resizeWindow("Motion Detection", 40, 480)





    if cv2.waitKey(1) == ord('q'):
        break








print (elapsed_time)

grabResult.Release()

# release resources
camera.Close()
cv2.destroyAllWindows()
writer.release()


with open(csv_name, mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['Time (milliseconds)', 'Velocity (pixels/sec)'])
    for i in range(len(time_list)):
        writer.writerow([round(time_list[i]*1000, 1), round(velocity_list[i], 2)])

import matplotlib.pyplot as plt

# plot the data
plt.plot(time_list, velocity_list)

# add axis labels and title
plt.xlabel("Time (s)")
plt.ylabel("Velocity (p/s)")
plt.axvline(x = 5, color = 'r', label = 'axvline - full height')
plt.axvline(x = 75, color = 'r', label = 'axvline - full height')
plt.axvspan(5, 10, color='red', alpha=0.3)
plt.axvspan(75, 80, color='green', alpha=0.3)
plt.axvspan(10, 70, color='gray', alpha=0.3)
plt.title("Velocity vs Time")


# show the plot
plt.savefig(plotname)
plt.show()
