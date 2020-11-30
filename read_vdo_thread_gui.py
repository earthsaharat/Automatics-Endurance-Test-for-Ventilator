import sys, time, datetime, threading, cv2
import queue as Queue
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import numpy as np
import statistics

import csv
CSV_FILE_NAME = 'result.csv'
csv_writer = None

IMG_FORMAT  = QImage.Format_RGB888
DISP_SCALE  = 4                 # Image scale
DISP_MSEC   = 50                # Display loops

VIDEO_PATH = "eVentilator_VDO_5min.mp4"

camera_num  = 1
image_queue = Queue.Queue()
is_capturing = False

PREVIEW_HEIGHT = 260

is_pressed = False
pulse = 0
last_bpm = 0
last_pulse = None
last_area = 0

COLOR_RANGE_LOWWER = np.array([80,80,120]) 
COLOR_RANGE_UPPER  = np.array([115,255,255]) 

CHART_HEIGHT = 55
CHART_RANGE  = 500

ANALYSIS_AVG_LENGTH = 5
ANALYSIS_LOW_LENGTH = 20
ANALYSIS_LOW_THRESHOLD = 0.15
ANALYSIS_RATIO_DEFAULT = 1.8

bpm_log = []
area_log = []
area_change_log = []

array1_log = []
array2_log = []
array3_log = []
area_change_buffer = []

def morphology_kernal(size):
	return np.ones((size,size), np.uint8)

def grab_images(path, queue):
	cap = cv2.VideoCapture(path)
	while is_capturing:
		if cap.grab():
			retval, image = cap.retrieve(0)
			if image is not None and queue.qsize() < 2:
				queue.put(image)
			else:
				time.sleep(DISP_MSEC / 1000.0)
		else:
			print("Error: can't grab camera image")
			break
	cap.release()


class ImageWidget(QWidget):
	def __init__(self, parent=None):
		super(ImageWidget, self).__init__(parent)
		self.image = None

	def setImage(self, image):
		self.image = image
		self.setMinimumSize(image.size())
		self.update()

	def paintEvent(self, event):
		qp = QPainter()
		qp.begin(self)
		if self.image:
			qp.drawImage(QPoint(0, 0), self.image)
		qp.end()


class MyWindow(QMainWindow):
	def __init__(self, parent=None):
		QMainWindow.__init__(self, parent)

		self.central     = QWidget(self)
		self.layout      = QVBoxLayout()        # Window layout

		# Layout display
		self.disp		= ImageWidget(self)
		self.disp2		= ImageWidget(self)

		self.chart_A0		= ImageWidget(self)
		self.chart_A1		= ImageWidget(self)
		self.chart_B1		= ImageWidget(self)
		self.chart_B2		= ImageWidget(self)
		self.chart_B3		= ImageWidget(self)
		self.chart_A2		= ImageWidget(self)

		self.label_A0 = QLabel('Video path : '+VIDEO_PATH, self)
		self.label_A1 = QLabel('CSV path : '+CSV_FILE_NAME, self)
		self.label_A2 = QLabel('', self)
		self.label_B1 = QLabel('', self)
		self.label_B2 = QLabel('', self)
		self.label_B3 = QLabel('Developed by : Saharat Saengsawang', self)

		self.group_disp = QVBoxLayout()
		self.group_disp.addWidget(self.disp)
		self.group_disp.addWidget(self.disp2)

		self.group_result = QVBoxLayout()
		self.group_result.setAlignment(Qt.AlignTop)
		self.group_result.addWidget(self.chart_A0)
		self.group_result.addWidget(self.label_A0)
		self.group_result.addWidget(self.chart_A1)
		self.group_result.addWidget(self.label_A1)
		self.group_result.addWidget(self.chart_A2)
		self.group_result.addWidget(self.label_A2)
		self.group_result.addWidget(self.chart_B1)
		self.group_result.addWidget(self.label_B1)
		self.group_result.addWidget(self.chart_B2)
		self.group_result.addWidget(self.label_B2)
		self.group_result.addWidget(self.chart_B3)
		self.group_result.addWidget(self.label_B3)


		# self.horizontalGroupBox = QHBoxLayout()
		# self.horizontalGroupBox.addWidget(self.label_A1)
		# self.horizontalGroupBox.addWidget(self.label_B1)

		self.group_column = QHBoxLayout()
		self.group_column.addLayout(self.group_disp)
		self.group_column.addLayout(self.group_result)


		# Layout menu
		self.layout_menu = QVBoxLayout()
		self.button1 = QPushButton('start')
		self.button1.released.connect(self.on_button1_released)
		self.layout_menu.addWidget(self.button1)

		self.layout.addLayout(self.group_column)
		self.layout.addLayout(self.layout_menu)
		self.central.setLayout(self.layout)
		self.setCentralWidget(self.central)

		exitAction = QAction('&Exit', self)
		exitAction.setShortcut('Ctrl+Q')
		exitAction.triggered.connect(self.close)

		# self.start()

	def start(self):
		global is_capturing
		is_capturing = True
		self.button1.setText('stop')
		self.timer = QTimer(self) # Timer to trigger display
		self.timer.timeout.connect(lambda: self.timer_handle())
		self.timer.start(DISP_MSEC)
		self.capture_thread = threading.Thread(target=grab_images, args=(VIDEO_PATH, image_queue))
		self.capture_thread.start() # Thread to grab images

		self.label_A0.setText('Respiratory rate \t\tcalculating...')
		self.label_A1.setText('Volume change \t\tcalculating...')

	def stop(self):
		global is_capturing
		self.timer.stop()
		is_capturing = False
		self.button1.setText('start')
		self.capture_thread.join()

	def timer_handle(self):
		if not image_queue.empty():
			img_org = image_queue.get()
			img_scale = PREVIEW_HEIGHT/img_org.shape[0]
			img_size = int(img_org.shape[1]*img_scale),PREVIEW_HEIGHT
			image = cv2.resize(img_org, img_size,interpolation=cv2.INTER_CUBIC)
			self.display_image(image, self.disp, 1)
			image = self.process(image)
			image = cv2.resize(image, img_size,interpolation=cv2.INTER_CUBIC)
			self.display_image(image, self.disp2, 1)
			self.display_image(self.plot_areaLog(bpm_log, 			normalize=True), self.chart_A0, 1)
			self.display_image(self.plot_areaLog(area_change_log,	normalize=True), self.chart_A1, 1)
			self.display_image(self.plot_areaLog(area_log), 	self.chart_A2, 1)
			self.display_image(self.plot_areaLog(array1_log), 	self.chart_B1, 1)
			self.display_image(self.plot_areaLog(array2_log), 	self.chart_B2, 1)
			self.display_image(self.plot_areaLog(array3_log), 	self.chart_B3, 1)

	# Display
	def display_image(self, img, display, scale=1):
		if not (img is not None and len(img) > 0): return
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		disp_size = img.shape[1]//scale, img.shape[0]//scale
		disp_bpl = disp_size[0] * 3
		if scale > 1:
			img = cv2.resize(img, disp_size,interpolation=cv2.INTER_CUBIC)
		qimg = QImage(img.data, disp_size[0], disp_size[1], disp_bpl, IMG_FORMAT)
		display.setImage(qimg)

	def plot_areaLog(self, data_array, normalize=False):
		img = np.zeros((CHART_HEIGHT,CHART_RANGE+70,3),dtype=np.uint8)
		if len(data_array) == 0: return img
		value_min = min(data_array)
		value_max = max(data_array)
		for i in range(0,len(data_array)):
			value = data_array[i]
			x1 = i; x2 = i
			if value_max-value_min == 0: y1 = 0; y2 = 0
			else:
				y2 = CHART_HEIGHT-1-int((value-value_min)/(value_max-value_min)*CHART_HEIGHT)
				if normalize:
					x1 = int(CHART_RANGE/len(data_array)*i)
					x2 = int(CHART_RANGE/len(data_array)*(i+1))-1
				y1 = CHART_HEIGHT-1
			cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,255),-1)
		cv2.putText(img, str(round(value_max,2)), (CHART_RANGE+4,10), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255,255,255))
		cv2.putText(img, str(round(value_min,2)), (CHART_RANGE+4,CHART_HEIGHT-4), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255,255,255))
		return img

	# Processing
	def process(self, img):
		global is_pressed, pulse, last_bpm, last_pulse, area_log, bpm_log, last_area, area_change_buffer

		imgPS = img.copy()

		hsv = cv2.cvtColor(imgPS, cv2.COLOR_BGR2HSV) 
		imgPS = cv2.inRange(hsv, COLOR_RANGE_LOWWER, COLOR_RANGE_UPPER)
		
		imgPS = cv2.morphologyEx(imgPS, cv2.MORPH_ERODE, morphology_kernal(11))
		imgPS = cv2.morphologyEx(imgPS, cv2.MORPH_DILATE, morphology_kernal(15))

		imgPS_cpy = imgPS.copy()
		ff_h, ff_w = imgPS.shape[:2]
		ff_mask = np.zeros((ff_h+2, ff_w+2), np.uint8)
		cv2.floodFill(imgPS_cpy, ff_mask, (0,0), 255)
		imgPS_cpy = cv2.bitwise_not(imgPS_cpy)

		imgPS = imgPS | imgPS_cpy

		contours, hierarchy = cv2.findContours(imgPS, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

		# imgDraw = cv2.cvtColor(imgPS,cv2.COLOR_GRAY2RGB)
		imgDraw = cv2.bitwise_and(img, img, mask = imgPS) 
		cv2.drawContours(imgDraw,contours,-1,(255,0,255),2)

		# x,y,w,h = cv2.boundingRect(contours[0])
		c = max(contours, key = cv2.contourArea)
		x,y,w,h = cv2.boundingRect(c)

		## chart_A2
		area = cv2.countNonZero(imgPS)
		self.label_A2.setText('Volume \t\t\t'+str(area)+'px')
		area_log.append(area)
		if len(area_log) > CHART_RANGE: area_log.pop(0)

		## chart_B1
		array1_value = w/h
		self.label_B1.setText('Ratio (Width/Height)\t'+str(round(array1_value,2)))
		array1_log.append(array1_value)
		if len(array1_log) > CHART_RANGE: array1_log.pop(0)

		## chart_B2
		array2_value = statistics.mean(array1_log[-ANALYSIS_AVG_LENGTH:])
		self.label_B2.setText('Ratio (Average)\t\t'+str(round(array2_value,2)))
		array2_log.append(array2_value)
		if len(array2_log) > CHART_RANGE: array2_log.pop(0)

		## chart_B3
		highpass_min = min(array2_log[-ANALYSIS_LOW_LENGTH:])
		highpass_max = max(array2_log[-ANALYSIS_LOW_LENGTH:])
		highpass_cutoff = highpass_min+(highpass_max-highpass_min)*ANALYSIS_LOW_THRESHOLD
		if abs(highpass_max-highpass_min) < 0.1:
			highpass_cutoff = ANALYSIS_RATIO_DEFAULT
		array3_value = array2_value > highpass_cutoff
		self.label_B3.setText('Pressed \t\t\t'+str(array3_value)+'\t(Threshold '+str(round(highpass_cutoff,2))+')')
		array3_log.append(array3_value)
		if len(array3_log) > CHART_RANGE: array3_log.pop(0)

		## Pulse Detection
		_is_pressed = array3_value
		if _is_pressed != is_pressed:
			if _is_pressed == False:
				pulse += 1
				now = datetime.datetime.now()
				if last_pulse != None:
					_timedelta = now-last_pulse
					last_bpm = 1/(((_timedelta.seconds*1000000 + _timedelta.microseconds)/1000000)/60)
				last_pulse = now
				if last_bpm != 0:
					self.label_A0.setText(
						'Respiratory rate \t\t'+str(round(last_bpm,2))+'bpm')

				if last_bpm != 0:
					## Append BPM
					bpm_log.append(last_bpm)
					if len(bpm_log) > CHART_RANGE: bpm_log.pop(0)
					## Append Volume Change
					volume_change_min = min(area_change_buffer)
					volume_change_max = max(area_change_buffer)
					volume_change_rate = (volume_change_max-volume_change_min)/volume_change_max*100
					area_change_log.append(volume_change_rate)
					if len(area_change_log) > CHART_RANGE: area_change_log.pop(0)
					self.label_A1.setText('Volume change \t\t'+str(round(volume_change_rate,2))+'%'+
						'\t(min '+str(round(volume_change_min,2))+'px, max '+str(round(volume_change_max,2))+'px)')
					with open(CSV_FILE_NAME, 'a+', newline='') as csvfile:
						csv_writer = csv.DictWriter(csvfile, fieldnames=['timestamp', 'bpm'])
						csv_writer.writerow({'timestamp': datetime.datetime.now(), 'bpm': last_bpm})
				area_change_buffer = []
		is_pressed = _is_pressed

		if is_pressed:
			area_change_buffer.append(area)
			if len(area_change_buffer) > CHART_RANGE: area_change_buffer.pop(0)

		cv2.rectangle(imgDraw,(x,y),(x+w,y+h),(255,255,0),2)

		cv2.putText(imgDraw,'pressed'				,(0,	10+0*15),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,255,0),1)
		cv2.putText(imgDraw,str(is_pressed)			,(80,	10+0*15),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,255,0),1)
		cv2.putText(imgDraw,'count'					,(0,	10+1*15),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,255,0),1)
		cv2.putText(imgDraw,str(pulse)				,(80,	10+1*15),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,255,0),1)
		cv2.putText(imgDraw,'bpm'					,(0,	10+2*15),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,255,0),1)
		cv2.putText(imgDraw,str(round(last_bpm,2))	,(80,	10+2*15),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,255,0),1)

		return imgDraw

	def on_button1_released(self):
		if not is_capturing: self.start()
		else: self.stop()

	def closeEvent(self, event):
		print('safe termination')
		self.stop()

if __name__ == '__main__':
	with open(CSV_FILE_NAME, 'w', newline='') as csvfile:
		csv_writer = csv.DictWriter(csvfile, fieldnames=['timestamp', 'bpm'])
		csv_writer.writeheader()
	

	app = QApplication(sys.argv)
	win = MyWindow()
	win.show()
	win.setWindowTitle("VDO")
	sys.exit(app.exec_())

#EOF