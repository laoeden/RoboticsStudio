#!/usr/bin/env python3

"""Pick an HSV colour and display the matching mask over an image."""

import argparse
import sys

import cv2
import numpy as np


WINDOW_NAME = "HSV colour picker"
MAX_TOLERANCE = 50


def nothing(_value: int) -> None:
	"""Trackbar callback placeholder."""


def create_trackbars() -> None:
	cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
	for name, maximum in (
		("Hue", 179),
		("Saturation", 255),
		("Value", 255),
		("Hue tolerance %", MAX_TOLERANCE),
		("Saturation tolerance %", MAX_TOLERANCE),
		("Value tolerance %", MAX_TOLERANCE),
	):
		cv2.createTrackbar(name, WINDOW_NAME, 0, maximum, nothing)

	# Start with a useful middle-brightness colour and modest tolerances.
	cv2.setTrackbarPos("Saturation", WINDOW_NAME, 128)
	cv2.setTrackbarPos("Value", WINDOW_NAME, 128)
	cv2.setTrackbarPos("Hue tolerance %", WINDOW_NAME, 10)
	cv2.setTrackbarPos("Saturation tolerance %", WINDOW_NAME, 10)
	cv2.setTrackbarPos("Value tolerance %", WINDOW_NAME, 10)


def read_source(source: str) -> tuple[cv2.VideoCapture | None, np.ndarray | None]:
	"""Return either a still image or an opened camera capture."""
	try:
		camera_index = int(source)
	except ValueError:
		image = cv2.imread(source)
		if image is None:
			raise FileNotFoundError(f"Could not read image: {source}")
		return None, image

	capture = cv2.VideoCapture(camera_index)
	if not capture.isOpened():
		capture.release()
		raise RuntimeError(f"Could not open camera {camera_index}")
	return capture, None


def build_display(image: np.ndarray) -> np.ndarray:
	hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hue = cv2.getTrackbarPos("Hue", WINDOW_NAME)
	saturation = cv2.getTrackbarPos("Saturation", WINDOW_NAME)
	value = cv2.getTrackbarPos("Value", WINDOW_NAME)

	hue_tolerance = cv2.getTrackbarPos("Hue tolerance %", WINDOW_NAME)
	saturation_tolerance = cv2.getTrackbarPos("Saturation tolerance %", WINDOW_NAME)
	value_tolerance = cv2.getTrackbarPos("Value tolerance %", WINDOW_NAME)

	# Hue is circular, so compare it with two ranges when the tolerance wraps.
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

	result = cv2.bitwise_and(image, image, mask=mask)
	return np.hstack((image, result))


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"source",
		help="image path, or camera index such as 0 (default: 0)",
		nargs="?",
		default="0",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	try:
		capture, image = read_source(args.source)
	except (FileNotFoundError, RuntimeError) as error:
		print(error, file=sys.stderr)
		return 1

	create_trackbars()
	try:
		while True:
			if capture is not None:
				ok, image = capture.read()
				if not ok:
					print("Could not read a frame from the camera.", file=sys.stderr)
					return 1
			assert image is not None
			cv2.imshow(WINDOW_NAME, build_display(image))
			# Keep polling for still images so trackbar changes are processed.
			key = cv2.waitKey(30) & 0xFF
			if key in (ord("q"), 27):
				break
	finally:
		if capture is not None:
			capture.release()
		cv2.destroyAllWindows()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
