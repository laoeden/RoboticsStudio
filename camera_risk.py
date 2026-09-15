#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


MAX_TOLERANCE = 50


def nothing(_value: int) -> None:
	"""Trackbar callback placeholder."""


def create_trackbars(window_name: str) -> None:
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
	for name, maximum in (
		('Hue', 179),
		('Saturation', 255),
		('Value', 255),
		('Hue tolerance %', MAX_TOLERANCE),
		('Saturation tolerance %', MAX_TOLERANCE),
		('Value tolerance %', MAX_TOLERANCE),
	):
		cv2.createTrackbar(name, window_name, 0, maximum, nothing)

	cv2.setTrackbarPos('Saturation', window_name, 128)
	cv2.setTrackbarPos('Value', window_name, 128)
	cv2.setTrackbarPos('Hue tolerance %', window_name, 10)
	cv2.setTrackbarPos('Saturation tolerance %', window_name, 10)
	cv2.setTrackbarPos('Value tolerance %', window_name, 10)


def build_masked_display(image: np.ndarray, window_name: str) -> np.ndarray:
	if image.ndim == 2:
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hue = cv2.getTrackbarPos('Hue', window_name)
	saturation = cv2.getTrackbarPos('Saturation', window_name)
	value = cv2.getTrackbarPos('Value', window_name)
	hue_tolerance = cv2.getTrackbarPos('Hue tolerance %', window_name)
	saturation_tolerance = cv2.getTrackbarPos(
		'Saturation tolerance %', window_name
	)
	value_tolerance = cv2.getTrackbarPos('Value tolerance %', window_name)

	hue_delta = int(round(179 * hue_tolerance / 100))
	low_hue = hue - hue_delta
	high_hue = hue + hue_delta
	low_saturation = max(0, saturation - round(255 * saturation_tolerance / 100))
	high_saturation = min(255, saturation + round(255 * saturation_tolerance / 100))
	low_value = max(0, value - round(255 * value_tolerance / 100))
	high_value = min(255, value + round(255 * value_tolerance / 100))

	if low_hue < 0:
		mask = cv2.inRange(
			hsv_image,
			np.array([0, low_saturation, low_value]),
			np.array([high_hue, high_saturation, high_value]),
		)
		mask |= cv2.inRange(
			hsv_image,
			np.array([180 + low_hue, low_saturation, low_value]),
			np.array([179, high_saturation, high_value]),
		)
	elif high_hue > 179:
		mask = cv2.inRange(
			hsv_image,
			np.array([low_hue, low_saturation, low_value]),
			np.array([179, high_saturation, high_value]),
		)
		mask |= cv2.inRange(
			hsv_image,
			np.array([0, low_saturation, low_value]),
			np.array([high_hue - 180, high_saturation, high_value]),
		)
	else:
		mask = cv2.inRange(
			hsv_image,
			np.array([low_hue, low_saturation, low_value]),
			np.array([high_hue, high_saturation, high_value]),
		)

	masked_image = cv2.bitwise_and(image, image, mask=mask)
	return np.hstack((image, masked_image))


class HuskyCameraViewer(Node):
	"""Display the Husky RGB camera stream in an OpenCV window."""

	def __init__(self):
		super().__init__('husky_camera_viewer')
		self.declare_parameter('image_topic', '/parrot1/camera/image')
		self.window_name = 'Husky camera'
		self.subscription = self.create_subscription(
			Image,
			self.get_parameter('image_topic').value,
			self.image_callback,
			10,
		)
		self.get_logger().info(
			f'Subscribed to {self.get_parameter("image_topic").value}. '
			'Press q or Escape in the camera window to quit.'
		)
		create_trackbars(self.window_name)

	def image_callback(self, message: Image) -> None:
		try:
			image = self._image_from_message(message)
		except ValueError as error:
			self.get_logger().error(str(error), throttle_duration_sec=2.0)
			return

		cv2.imshow(self.window_name, build_masked_display(image, self.window_name))
		key = cv2.waitKey(1) & 0xFF
		if key in (ord('q'), 27):
			rclpy.shutdown()

	@staticmethod
	def _image_from_message(message: Image) -> np.ndarray:
		channels = {'mono8': 1, 'bgr8': 3, 'rgb8': 3}.get(message.encoding)
		if channels is None:
			raise ValueError(
				f'Unsupported camera encoding: {message.encoding!r}. '
				'Expected mono8, bgr8, or rgb8.'
			)

		row_bytes = message.width * channels
		if message.step < row_bytes:
			raise ValueError(
				f'Invalid image step {message.step} for width {message.width}.'
			)

		image = np.frombuffer(message.data, dtype=np.uint8).reshape(
			message.height, message.step
		)[:, :row_bytes]
		if channels == 1:
			return image.reshape(message.height, message.width)

		image = image.reshape(message.height, message.width, channels)
		if message.encoding == 'rgb8':
			return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		return image


def main() -> None:
	rclpy.init()
	viewer = HuskyCameraViewer()
	try:
		rclpy.spin(viewer)
	except KeyboardInterrupt:
		pass
	finally:
		viewer.destroy_node()
		cv2.destroyAllWindows()
		if rclpy.ok():
			rclpy.shutdown()


if __name__ == '__main__':
	main()

