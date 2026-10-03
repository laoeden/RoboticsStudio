#!/usr/bin/env python3

import cv2
import math
import threading
import time
import numpy as np

from geometry_msgs.msg import PointStamped
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
MIN_OBJECT_AREA_M2 = 1.0
TEXTURE_KERNEL_SIZE = 7
MIN_TEXTURE_VARIANCE = 180.0
MIN_TEXTURE_FRACTION = 0.35
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
		('Texture variance', 1000),
		('Texture fraction %', 100),
	):
		cv2.createTrackbar(name, window_name, 0, maximum, nothing)

	cv2.setTrackbarPos('Saturation', window_name, 128)
	cv2.setTrackbarPos('Value', window_name, 128)
	cv2.setTrackbarPos('Hue tolerance %', window_name, 10)
	cv2.setTrackbarPos('Saturation tolerance %', window_name, 10)
	cv2.setTrackbarPos('Value tolerance %', window_name, 10)
	cv2.setTrackbarPos('Texture variance', window_name, int(MIN_TEXTURE_VARIANCE))
	cv2.setTrackbarPos(
		'Texture fraction %', window_name, int(MIN_TEXTURE_FRACTION * 100)
	)


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


def build_texture_mask(
	image: np.ndarray,
	window_name: str,
) -> np.ndarray:
	if image.ndim == 2:
		gray_image = image.astype(np.float32)
	else:
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
	mean = cv2.GaussianBlur(
		gray_image,
		(TEXTURE_KERNEL_SIZE, TEXTURE_KERNEL_SIZE),
		0,
	)
	mean_squared = cv2.GaussianBlur(
		gray_image * gray_image,
		(TEXTURE_KERNEL_SIZE, TEXTURE_KERNEL_SIZE),
		0,
	)
	variance = np.maximum(mean_squared - mean * mean, 0.0)
	minimum_variance = cv2.getTrackbarPos('Texture variance', window_name)
	return np.where(variance >= minimum_variance, 255, 0).astype(np.uint8)


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
	texture_mask: np.ndarray | None = None,
	minimum_texture_fraction: float = MIN_TEXTURE_FRACTION,
	fx: float = FX,
	fy: float = FY,
	cx: float = CX,
	cy: float = CY,
) -> list[tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int], np.ndarray | None]]:
	if depth.shape[:2] != mask.shape[:2]:
		return []
	if texture_mask is not None and texture_mask.shape != mask.shape:
		return []
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = [contour for contour in contours if cv2.contourArea(contour) >= MIN_CONTOUR_AREA]
	objects = []
	for contour in contours:
		x, y, width, height = cv2.boundingRect(contour)
		center = (x + width // 2, y + height // 2)
		contour_mask = np.zeros(mask.shape, dtype=np.uint8)
		cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
		if texture_mask is not None:
			contour_pixels = contour_mask > 0
			textured_fraction = np.count_nonzero(
				texture_mask[contour_pixels]
			) / np.count_nonzero(contour_pixels)
			if textured_fraction < minimum_texture_fraction:
				continue
		valid_depth = depth[(contour_mask > 0) & np.isfinite(depth)]
		valid_depth = valid_depth[(valid_depth >= MIN_DEPTH_M) & (valid_depth <= MAX_DEPTH_M)]
		point = None
		if valid_depth.size > 0:
			depth_m = float(np.median(valid_depth))
			projected_area_m2 = (
				cv2.contourArea(contour) * depth_m ** 2 / (fx * fy)
			)
			if projected_area_m2 < MIN_OBJECT_AREA_M2:
				continue
			point = np.array([
				(center[0] - cx) * depth_m / fx,
				(center[1] - cy) * depth_m / fy,
				depth_m,
			])
		objects.append((contour, (x, y, width, height), center, point))
	return objects


def find_world_points(
	image: np.ndarray,
	depth: np.ndarray | None,
	window_name: str,
	odom: Odometry | None,
	scale: float = 1.0,
) -> list[np.ndarray]:
	if depth is None or odom is None:
		return []
	if scale != 1.0:
		image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
		depth = cv2.resize(depth, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
	if image.ndim == 2:
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	mask = build_mask(image, window_name)
	texture_mask = build_texture_mask(image, window_name)
	minimum_texture_fraction = cv2.getTrackbarPos(
		'Texture fraction %', window_name
	) / 100.0
	quaternion = np.array([
		odom.pose.pose.orientation.x,
		odom.pose.pose.orientation.y,
		odom.pose.pose.orientation.z,
		odom.pose.pose.orientation.w,
	])
	objects = find_objects(
		mask,
		depth,
		texture_mask,
		minimum_texture_fraction,
		fx=FX * scale,
		fy=FY * scale,
		cx=CX * scale,
		cy=CY * scale,
	)
	world_points = []
	for _, _, _, camera_point in objects:
		if camera_point is None:
			continue
		base_point = camera_to_base(camera_point)
		world_points.append(
			rotate_by_quaternion(base_point, quaternion)
			+ np.array([
				odom.pose.pose.position.x,
				odom.pose.pose.position.y,
				odom.pose.pose.position.z,
			])
		)
	return world_points


def build_display(
	image: np.ndarray,
	depth: np.ndarray | None,
	window_name: str,
	odom: Odometry | None,
	scale: float = 1.0,
) -> np.ndarray:
	if scale != 1.0:
		image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
		if depth is not None:
			depth = cv2.resize(depth, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
	if image.ndim == 2:
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	mask = build_mask(image, window_name)
	texture_mask = build_texture_mask(image, window_name)
	minimum_texture_fraction = cv2.getTrackbarPos(
		'Texture fraction %', window_name
	) / 100.0
	filtered_mask = cv2.bitwise_and(mask, texture_mask)
	masked_image = cv2.bitwise_and(image, image, mask=filtered_mask)
	objects = (
		find_objects(
			mask,
			depth,
			texture_mask,
			minimum_texture_fraction,
			fx=FX * scale,
			fy=FY * scale,
			cx=CX * scale,
			cy=CY * scale,
		)
		if depth is not None
		else []
	)
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
		cv2.putText(masked_image, odom_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
	for index, (_, (x, y, width, height), _, camera_point) in enumerate(objects, start=1):
		cv2.rectangle(masked_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
		label = f'#{index}: depth unavailable'
		if odom is not None and camera_point is not None:
			base_point = camera_to_base(camera_point)
			world_point = rotate_by_quaternion(base_point, quaternion) + np.array([pose.position.x, pose.position.y, pose.position.z])
			label = f'#{index}: ({world_point[0]:.2f}, {world_point[1]:.2f}, {world_point[2]:.2f}) m'
		cv2.putText(masked_image, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
	return np.hstack((image, masked_image))


class HuskyCameraViewer(Node):
	"""Display the Husky RGB camera stream in an OpenCV window."""

	def __init__(self):
		super().__init__('husky_camera_viewer')
		self.declare_parameter('image_topic', '/parrot1/camera/image')
		self.declare_parameter('depth_topic', '/parrot1/camera/depth/image')
		self.declare_parameter('odom_topic', '/parrot1/odometry')
		self.declare_parameter('husky_goal_topic', '/husky1/goal')
		self.declare_parameter('husky_odom_topic', '/husky1/odometry')
		self.declare_parameter('husky_goal_frame', 'husky1/odom')
		self.declare_parameter('duplicate_radius', 1.0)
		self.declare_parameter('husky_goal_tolerance', 0.2)
		self.declare_parameter('display_scale', 0.5)
		self.declare_parameter('display_fps', 20.0)
		self.window_name = 'Husky camera'
		self.odom = None
		self.husky_goal_frame = self.get_parameter('husky_goal_frame').value
		self.duplicate_radius = float(self.get_parameter('duplicate_radius').value)
		self.husky_goal_tolerance = float(
			self.get_parameter('husky_goal_tolerance').value
		)
		self.display_scale = float(self.get_parameter('display_scale').value)
		self.display_fps = float(self.get_parameter('display_fps').value)
		if any(value <= 0 for value in (
			self.duplicate_radius,
			self.husky_goal_tolerance,
			self.display_scale,
			self.display_fps,
		)) or self.display_scale > 1.0:
			raise ValueError('display_scale must be in (0, 1], other display values must be positive')
		self.published_locations: list[np.ndarray] = []
		self.pending_locations: list[np.ndarray] = []
		self.active_location: np.ndarray | None = None
		self.husky_odom = None
		self.frame_lock = threading.Lock()
		self.latest_image: np.ndarray | None = None
		self.latest_depth: np.ndarray | None = None
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
		self.husky_goal_publisher = self.create_publisher(
			PointStamped,
			self.get_parameter('husky_goal_topic').value,
			10,
		)
		self.husky_odom_subscription = self.create_subscription(
			Odometry,
			self.get_parameter('husky_odom_topic').value,
			self.husky_odom_callback,
			10,
		)
		self.get_logger().info(
			f'Subscribed to {self.get_parameter("image_topic").value}. '
			'Press q or Escape in the camera window to quit.'
		)

	def depth_callback(self, message: Image) -> None:
		try:
			depth = self._depth_from_message(message)
			with self.frame_lock:
				self.latest_depth = depth
		except ValueError as error:
			self.get_logger().error(str(error), throttle_duration_sec=2.0)

	def odom_callback(self, message: Odometry) -> None:
		self.odom = message

	def husky_odom_callback(self, message: Odometry) -> None:
		self.husky_odom = message
		if self.active_location is None:
			return
		position = message.pose.pose.position
		distance = math.hypot(
			position.x - self.active_location[0],
			position.y - self.active_location[1],
		)
		if distance <= self.husky_goal_tolerance:
			self.get_logger().info(
				f'Husky reached contour goal ({distance:.2f} m away).'
			)
			self.active_location = None
			self.publish_next_husky_goal()

	def image_callback(self, message: Image) -> None:
		try:
			image = self._image_from_message(message)
		except ValueError as error:
			self.get_logger().error(str(error), throttle_duration_sec=2.0)
			return

		with self.frame_lock:
			self.latest_image = image

	def run_gui(self) -> None:
		create_trackbars(self.window_name)
		frame_period = 1.0 / self.display_fps
		next_frame = time.monotonic()
		while rclpy.ok():
			now = time.monotonic()
			if now < next_frame:
				cv2.waitKey(max(1, int((next_frame - now) * 1000)))
				continue
			next_frame = now + frame_period
			with self.frame_lock:
				image = self.latest_image
				depth = self.latest_depth
			if image is not None:
				try:
					self.publish_new_locations(image, depth, self.display_scale)
					cv2.imshow(
						self.window_name,
						build_display(
							image,
							depth,
							self.window_name,
							self.odom,
							self.display_scale,
						),
					)
				except cv2.error as error:
					self.get_logger().error(
						f'Camera display reset: {error}',
						throttle_duration_sec=2.0,
					)
					create_trackbars(self.window_name)
			key = cv2.waitKey(1) & 0xFF
			if key in (ord('q'), 27):
				rclpy.shutdown()
				break

	def publish_new_locations(
		self,
		image: np.ndarray,
		depth: np.ndarray | None,
		scale: float,
	) -> None:
		known_locations = [
			location[:2] for location in self.published_locations
		]
		known_locations.extend(location[:2] for location in self.pending_locations)
		if self.active_location is not None:
			known_locations.append(self.active_location[:2])
		for world_point in find_world_points(
			image,
			depth,
			self.window_name,
			self.odom,
			scale,
		):
			if any(
				np.linalg.norm(world_point[:2] - old_point) < self.duplicate_radius
				for old_point in known_locations
			):
				continue
			self.published_locations.append(world_point.copy())
			known_locations.append(world_point[:2].copy())
			self.get_logger().info(
				'Queued new Husky contour goal at '
				f'({world_point[0]:.2f}, {world_point[1]:.2f}).'
			)
			self.pending_locations.append(world_point.copy())
		self.publish_next_husky_goal()

	def publish_next_husky_goal(self) -> None:
		if self.active_location is not None or not self.pending_locations:
			return
		self.active_location = self.pending_locations.pop(0)
		goal = PointStamped()
		goal.header.frame_id = self.husky_goal_frame
		goal.point.x = float(self.active_location[0])
		goal.point.y = float(self.active_location[1])
		goal.point.z = 0.0
		self.husky_goal_publisher.publish(goal)
		self.get_logger().info(
			'Published Husky contour goal at '
			f'({goal.point.x:.2f}, {goal.point.y:.2f}) '
			f'in {goal.header.frame_id}; '
			f'{len(self.pending_locations)} queued.'
		)

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
	ros_thread = threading.Thread(target=rclpy.spin, args=(viewer,), daemon=True)
	ros_thread.start()
	try:
		viewer.run_gui()
	except KeyboardInterrupt:
		pass
	finally:
		viewer.destroy_node()
		cv2.destroyAllWindows()
		if rclpy.ok():
			rclpy.shutdown()


if __name__ == '__main__':
	main()

