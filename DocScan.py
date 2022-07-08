#!/bin/python


from Transform import FourPointTransform
from skimage.filters import threshold_local
from argparse import ArgumentParser
import numpy as np
import cv2
import imutils



class DocSanner(FourPointTransform):
	def __init__(self, pts, image):
		super().__init__(pts, image)
		
		
	def load_image(self):
		self.image = cv2.imread(self.image)
	
	
	def calc_old_to_new_height_ratio(self):
		self.ratio = self.image.shape[0] / 500.0
	
	
	def clone_original_image(self):
		self.orig_image = self.image.copy()
	
	
	def resize_original_image(self):
		self.image = imutils.resize(
			self.image, 
			height = 500
		) 
	
	
	def grayscale_original_image(self):
		self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	
	def apply_gaussian_blur_to_grayscale_image(self):
		self.gauss_image = cv2.GaussianBlur(self.gray, (5, 5), 0)
		
		
	def canny_edge_detection_on_blurred_image(self):
		self.edged_image = cv2.Canny(self.gauss_image, 75, 200)
	
	
	def find_countours_of_edge_detected_image(self):
		self.contours = cv2.findContours(
			self.edged.copy(), 
			cv2.RETR_LIST, 
			cv2.CHAIN_APPROX_SIMPLE
		)
		self.contours = imutils.grab_contours(self.contours)
		self.contours = sorted(
			self.contours,
			key = cv2.contourArea,
			reverse = True
		)
	
	def find_the_document_image(self):
		# loop over the contours
		for contour in self.contours:
			# approximate the contour
			peri = cv2.arcLength(contour, True)
			approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
			# if our approximated contour has four points, then we
			# can assume that we have found our screen
			if len(approx) == 4:
				self.screen_contour = approx
				break
	
	def get_top_down_image_view(self):
		self.image = self.orig_image
		self.pts = self.screen_contour.reshape(4, 2) * self.ratio
		self.order_points()
		self.four_point_transform()
	
	
	def apply_black_and_white_filter(self):
		self.warped = cv2.cvtColor(self.warped, cv2.COLOR_BGR2GRAY)
		threshold = threshold_local(
			self.warped, 
			11, 
			offset=10, 
			method="gaussian"
		)
		self.warped = (self.warped > threshold).astype("uint8")*255
		
	
	def scan_doc_with_four_point_transform(self):
		self.load_image()
		self.calc_old_to_new_height_ratio()
		self.clone_original_image()
		self.resize_original_image()
		self.grayscale_original_image()
		self.apply_gaussian_blur_to_grayscale_image()
		self.canny_edge_detection_on_blurred_image()
		self.find_countours_of_edge_detected_image()
		self.find_the_document_image()
		self.get_top_down_image_view()
		self.apply_black_and_white_filter()
		self.new_scanned_image = self.warped
		
		
		
if __name__=="__main__":
	argparser = ArgumentParser(
		description = """
			Scans document and performs four point transform of the image to obtain a 
			"bird's eye" 90 degree view of the document"""
	)
	
	argparser.add_argument(
		"-i", "--image",
		required=True,
		help="The path to the image to be scanned"
	)
	
	pts = ""
	image = argparser.parse_args().image
	basename = os.path.basename(image)
	filename = basename.split(".")[:-1]
	file_extension = basename.split(".")[-1]
	
	doc_scanner_obj = DocSanner(pts, image)
	doc_scanner_obj.scan_doc_with_four_point_transform()
	cv2.imwrite(
		"{}_scanned_{}".format(filename, file_extension), 
		doc_scanner_obj.new_scanned_image
	) 
	
	
	
