#!/usr/bin/env python3

import cv2
import math
import numpy as np

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image


MAX_TOLERANCE = 50
FX = 207.85
FY = 207.85
CX = 360.0
CY = 240.0
MIN_CONTOUR_AREA = 100.0
MIN_DEPTH_M = 0.4
MAX_DEPTH_M = 10.0


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


def build_mask(image: np.ndarray, window_name: str) -> np.ndarray:
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

	return mask


def camera_to_base(point: np.ndarray) -> np.ndarray:
	"""Transform a point from the depth optical frame into parrot base frame."""
	# URDF: base -> camera_link -> camera_depth_frame -> optical frame.
	def rotation_x(angle: float) -> np.ndarray:
		return np.array([
			[1.0, 0.0, 0.0],
			[0.0, math.cos(angle), -math.sin(angle)],
			[0.0, math.sin(angle), math.cos(angle)],
		])

	def rotation_y(angle: float) -> np.ndarray:
		return np.array([
			[math.cos(angle), 0.0, math.sin(angle)],
			[0.0, 1.0, 0.0],
			[-math.sin(angle), 0.0, math.cos(angle)],
		])

	def rotation_z(angle: float) -> np.ndarray:
		return np.array([
			[math.cos(angle), -math.sin(angle), 0.0],
			[math.sin(angle), math.cos(angle), 0.0],
			[0.0, 0.0, 1.0],
		])

	# URDF fixed joints: base -> camera_link -> depth_frame -> optical_frame.
	# The camera_link rotation is +90 degrees about Y. The optical joint's
	# RPY is applied as Rz(yaw) @ Rx(roll), not in textual RPY order.
	rotation = rotation_y(math.pi / 2) @ rotation_z(-math.pi / 2) @ rotation_x(-math.pi / 2)
	translation = np.array([0.0, 0.0, -0.25]) + rotation_y(math.pi / 2) @ np.array([0.005, 0.028, 0.013])
	return rotation @ point + translation


def rotate_by_quaternion(point: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
	x, y, z, w = quaternion
	rotation = np.array([
		[1 - 2 * y * y - 2 * z * z, 2 * (x * y - z * w), 2 * (x * z + y * w)],
		[2 * (x * y + z * w), 1 - 2 * x * x - 2 * z * z, 2 * (y * z - x * w)],
		[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * x * x - 2 * y * y],
	])
	return rotation @ point


def quaternion_yaw(quaternion: np.ndarray) -> float:
	x, y, z, w = quaternion
	return math.atan2(
		2.0 * (w * z + x * y),
		1.0 - 2.0 * (y * y + z * z),
	)


def find_objects(
	mask: np.ndarray,
	depth: np.ndarray,
) -> list[tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int], np.ndarray | None]]:
	if depth.shape[:2] != mask.shape[:2]:
		return []
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = [contour for contour in contours if cv2.contourArea(contour) >= MIN_CONTOUR_AREA]
	objects = []
	for contour in contours:
		x, y, width, height = cv2.boundingRect(contour)
		center = (x + width // 2, y + height // 2)
		contour_mask = np.zeros(mask.shape, dtype=np.uint8)
		cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
		valid_depth = depth[(contour_mask > 0) & np.isfinite(depth)]
		valid_depth = valid_depth[(valid_depth >= MIN_DEPTH_M) & (valid_depth <= MAX_DEPTH_M)]
		point = None
		if valid_depth.size > 0:
			depth_m = float(np.median(valid_depth))
			point = np.array([
				(center[0] - CX) * depth_m / FX,
				(center[1] - CY) * depth_m / FY,
				depth_m,
			])
		objects.append((contour, (x, y, width, height), center, point))
	return objects


def build_display(
	image: np.ndarray,
	depth: np.ndarray | None,
	window_name: str,
	odom: Odometry | None,
) -> np.ndarray:
	if image.ndim == 2:
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	mask = build_mask(image, window_name)
	masked_image = cv2.bitwise_and(image, image, mask=mask)
	objects = find_objects(mask, depth) if depth is not None else []
	if odom is None:
		cv2.putText(
			masked_image,
			'odom: waiting for message',
			(10, 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			(0, 165, 255),
			2,
		)
	else:
		pose = odom.pose.pose
		quaternion = np.array([
			pose.orientation.x,
			pose.orientation.y,
			pose.orientation.z,
			pose.orientation.w,
		])
		odom_label = (
			f'odom [{odom.header.frame_id}]: '
			f'x={pose.position.x:.2f} y={pose.position.y:.2f} '
			f'z={pose.position.z:.2f} yaw={math.degrees(quaternion_yaw(quaternion)):.1f} deg'
		)
		cv2.putText(
			masked_image,
			odom_label,
			(10, 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.55,
			(255, 255, 0),
			2,
		)
	for index, (_, (x, y, width, height), center, camera_point) in enumerate(objects, start=1):
		cv2.rectangle(masked_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
		label = f'#{index}: depth unavailable'
		if odom is not None and camera_point is not None:
			base_point = camera_to_base(camera_point)
			world_point = rotate_by_quaternion(
				base_point,
				quaternion,
			) + np.array([pose.position.x, pose.position.y, pose.position.z])
			label = f'#{index}: ({world_point[0]:.2f}, {world_point[1]:.2f}, {world_point[2]:.2f}) m'
		text_y = max(18, y - 6)
		cv2.putText(
			masked_image,
			label,
			(x, text_y),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.45,
			(0, 255, 0),
			2,
		)
	return np.hstack((image, masked_image))


class HuskyCameraViewer(Node):
	"""Display the Husky RGB camera stream in an OpenCV window."""

	def __init__(self):
		super().__init__('husky_camera_viewer')
		self.declare_parameter('image_topic', '/parrot1/camera/image')
		self.declare_parameter('depth_topic', '/parrot1/camera/depth/image')
		self.declare_parameter('odom_topic', '/parrot1/odometry')
		self.window_name = 'Husky camera'
		self.depth_image = None
		self.odom = None
		self.subscription = self.create_subscription(
			Image,
			self.get_parameter('image_topic').value,
			self.image_callback,
			10,
		)
		self.depth_subscription = self.create_subscription(
			Image,
			self.get_parameter('depth_topic').value,
			self.depth_callback,
			10,
		)
		self.odom_subscription = self.create_subscription(
			Odometry,
			self.get_parameter('odom_topic').value,
			self.odom_callback,
			10,
		)
		self.get_logger().info(
			f'Subscribed to {self.get_parameter("image_topic").value}. '
			'Press q or Escape in the camera window to quit.'
		)
		create_trackbars(self.window_name)

	def depth_callback(self, message: Image) -> None:
		try:
			self.depth_image = self._depth_from_message(message)
		except ValueError as error:
			self.get_logger().error(str(error), throttle_duration_sec=2.0)

	def odom_callback(self, message: Odometry) -> None:
		self.odom = message

	def image_callback(self, message: Image) -> None:
		try:
			image = self._image_from_message(message)
		except ValueError as error:
			self.get_logger().error(str(error), throttle_duration_sec=2.0)
			return

		cv2.imshow(
			self.window_name,
			build_display(image, self.depth_image, self.window_name, self.odom),
		)
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

	@staticmethod
	def _depth_from_message(message: Image) -> np.ndarray:
		if message.encoding not in ('32FC1', '16UC1'):
			raise ValueError(
				f'Unsupported depth encoding: {message.encoding!r}. '
				'Expected 32FC1 or 16UC1.'
			)
		dtype = np.float32 if message.encoding == '32FC1' else np.uint16
		bytes_per_pixel = np.dtype(dtype).itemsize
		row_bytes = message.width * bytes_per_pixel
		if message.step < row_bytes:
			raise ValueError(
				f'Invalid depth image step {message.step} for width {message.width}.'
			)
		depth = np.frombuffer(message.data, dtype=dtype).reshape(
			message.height, message.step // bytes_per_pixel
		)[:, :message.width].copy()
		if message.encoding == '16UC1':
			return depth.astype(np.float32) / 1000.0
		return depth


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

