############ @chumo 2015

# imports
import cv2
import numpy as np

import autopy
import autopy.mouse as am

# globals
corners = []
subproc = 'Set image corners'
color_to_detect = cv2.cvtColor(np.uint8([[[0,0,255]]]),cv2.COLOR_BGR2HSV) # pure red by default
mousedown = False
mouseH = 0 # mouse position along height direction
mouseW = 0 # mouse position along width direction


brightness = 1

# functions
def click_corner(event, x, y, flags, frame):
	if event == cv2.EVENT_LBUTTONUP:
		corners.append([x,y])

def pick_color(event, x, y, flags, frame):	
	global mouseH, mouseW, color_to_detect
	mouseH = x
	mouseW = y
	if event == cv2.EVENT_LBUTTONUP:
		color_to_detect = np.max(frame[:,:,2])#cv2.cvtColor(np.uint8([[frame[y,x].tolist()]]),cv2.COLOR_BGR2HSV)
		print int(color_to_detect*1.2)

def getThresImage():
	# smooth it
	frame_blur = frame.copy()
	#frame_blur = cv2.blur(frame,(5,5))

	##################### 1
	# # convert to hsv 
	# hsv = cv2.cvtColor(frame_blur,cv2.COLOR_BGR2HSV)
	# # filter the color to detect
	# h_to_detect = color_to_detect[0][0][0]
	# thresh = cv2.inRange(hsv,np.array((h_to_detect*0.85,127,127)), np.array((h_to_detect*1.15,255,255)))

	##################### 2
	# total = frame_blur[:,:,2].astype('float32')#+frame_blur[:,:,1].astype('float32')+frame_blur[:,:,0].astype('float32')
	# maxV = np.max(total)
	# minV = np.min(total)
	# total -= minV
	# total /= (maxV-minV)
	# # thresh = cv2.inRange(total,255*3*0.92, 255*3)    
	# # thresh = cv2.inRange(total,0.9, 1.0)    
 #    # print np.min(total),np.max(total)
	# thresh = cv2.inRange(frame[:,:,1],int(color_to_detect*1.2), 255)    

	##################### 3
	RmG = frame_blur[:,:,1]
	thresh = cv2.inRange(RmG,235, 255)    

	return thresh


def track_color():
	global mousedown
	global thresh
	
	# Get threshold image
	thresh = getThresImage()
	#cv2.imshow('thresh', thresh)

    # find contours in the threshold image
	contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    # finding contour with maximum area and store it as best_cnt
	max_area = 0
	for cnt in contours:
	    area = cv2.contourArea(cnt)
	    if area > max_area:
	        max_area = area
	        best_cnt = cnt
    
	try:
		# finding centroids of best_cnt and draw a circle there
		M = cv2.moments(best_cnt)
		cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
		cv2.circle(frame,(cx,cy),8,[0, 0, 0],5)
		cv2.circle(frame,(cx,cy),8,[0, 255, 0],4)
		# mouse
		if subproc == 'Mouse control':
			scX, scY = ps.screenXY(cx,cy)
			am.move(scX, scY)
			if mousedown==False:
				#am.toggle(True,am.LEFT_BUTTON)
				mousedown = True
				print 'mousedown'
			
	except NameError: # the color was not detected
    	# mouse
		if subproc == 'Mouse control':
			if mousedown==True:
				#am.toggle(False,am.LEFT_BUTTON)
				mousedown = False
				print 'mouseup'
        #cv2.circle(frame,(50,50),5,0,-1)


def order_points(pts): # from http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
	'''
	Initialize a list of coordinates that will be ordered
	in the following order: top-left, top-right, bottom-right, bottom-left
	'''
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

class PS(object):
	
	def __init__(self,corners):
		self.corners = np.float32(corners)
		# obtain screen corners
		screen_size = autopy.screen.get_size()
		self.screen_corners = np.array([[0,0],[screen_size[0]-1,0],[screen_size[0]-1,screen_size[1]-1],[0,screen_size[1]-1]],'float32')
		self.screen_size = screen_size
	
	def screenXY(self,imX,imY):
		M = cv2.getPerspectiveTransform(order_points(self.corners), self.screen_corners)
		sc = cv2.perspectiveTransform(np.float32([[[imX,imY]]]),M) #I have to use triple [] because: http://answers.opencv.org/question/252/cv2perspectivetransform-with-python/?answer=254#post-id-254
		scX = sc[0][0][0].astype('int')
		scY = sc[0][0][1].astype('int')
		# don't go out of bounds
		scX = max(scX, 0)
		scX = min(scX, self.screen_size[0]-1)
		scY = max(scY, 0)
		scY = min(scY, self.screen_size[1]-1)

		return scX, scY

# create video capture
capture = cv2.VideoCapture(1)
#capture.set(cv2.cv.CV_CAP_PROP_FPS, 30)
capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480.0);
capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640.0);

frameW = int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
frameH = int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

print capture.get(cv2.cv.CV_CAP_PROP_FPS)

# windows
cv2.namedWindow('frame')
cv2.moveWindow('frame', 205, 0)
# cv2.namedWindow('zoom')
# cv2.namedWindow('thresh')

# acquisition loop
while(1):
	_, frame = capture.read()

	# setting flags according to key strokes
	key = cv2.waitKey(1)
	c = chr(key & 255)
	if c in ['q', 'Q', chr(27)]:
		break
	elif c in ['c', 'C']:
		subproc = 'Set image corners'
	elif c in ['p', 'P']:
		subproc = 'Pick color to track'
	elif c in ['t', 'T']:
		subproc = 'Mouse test'
	elif c in ['m', 'M']:
		subproc = 'Mouse control'
	elif c in ['S']:
		brightness += 1
	elif c in ['s']:
		brightness -= 1

	# depending on the flag, do one thing or another
	if subproc == 'Set image corners':
		for corner in corners[-4:]: # just the last four selected corners
			cv2.circle(frame,(corner[0],corner[1]),5,255,-1)

		cv2.setMouseCallback('frame', click_corner, frame)
		if len(corners) >= 4: # if we have four, instantiate PS
			ps = PS(corners)
	
	elif subproc == 'Pick color to track':
		cv2.setMouseCallback('frame', pick_color, frame)
		
		zoom = np.zeros((21,21,3),dtype='uint8')
		#zoom = frame[max(0,mouseW-10):min(frameW-1,mouseW+10), max(0,mouseH-10):min(frameH-1,mouseH+10),:]
		try:
			zoom[:,:,:] = frame[mouseW-10:mouseW+11, mouseH-10:mouseH+11, :]
			# print max(0,mouseH-50),min(upX-1,mouseH+50),max(0,mouseW-50),min(upY-1,mouseW+50)
			# if mouseH != 0:
			zoom = cv2.resize(zoom, (210,210),interpolation=cv2.INTER_AREA)
			
		except:
			print 'The zoom window exceeds the frame.'

		cv2.rectangle(zoom, (95,95), (105,105), [255-frame[mouseW,mouseH,0],255-frame[mouseW,mouseH,1],255-frame[mouseW,mouseH,2]], 1)
		cv2.imshow('zoom', zoom)

	elif subproc == 'Mouse test':
		track_color()
		cv2.imshow('thresh', thresh)

	elif subproc == 'Mouse control':
		if len(corners) >= 4: # check that the corners have been defined
			track_color()
		else:
			print 'Tell me where are the corners of the projection on the camera image.'
			subproc = 'Set image corners'

	#print brightness
	#capture.set(cv2.cv.CV_CAP_PROP_BRIGHTNESS,brightness)

	# negative conditions
	if subproc != 'Pick color to track':
		cv2.destroyWindow('zoom')
	# if subproc != 'Mouse test':
	# 	cv2.destroyWindow('thresh')

	# update frame	 
	cv2.putText(frame, subproc, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	cv2.imshow('frame', frame)
	
	#hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	#cv2.imshow('red', frame[:,:,2])
	#cv2.imshow('green', frame[:,:,1])
	#cv2.imshow('blue', frame[:,:,0])
	#cv2.imshow('red minus green', frame[:,:,2]-frame[:,:,1]-frame[:,:,0])

	# cv2.imshow('H', frame[:,:,0])
	# cv2.imshow('S', frame[:,:,1])
	# cv2.imshow('V', frame[:,:,2])

	thresh = getThresImage()
	cv2.imshow('thresh', thresh)

# Clean up everything before leaving
cv2.destroyAllWindows()
capture.release()




















