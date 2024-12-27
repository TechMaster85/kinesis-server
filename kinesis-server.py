import argparse
import sys
import time

import numpy as np

from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess import COCODrawer
from picamera2.devices.imx500.postprocess_highernet import \
    postprocess_higherhrnet

from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder

# TODO Change this based on command line arguments
dest_ip = "35.6.168.186"
dest_port = 35155

last_boxes = None
last_scores = None
last_keypoints = None
WINDOW_SIZE_H_W = (480, 640)
OSC_ADDRESS = "/kinesis/pose"

client = udp_client.SimpleUDPClient(dest_ip, dest_port, True)

keypoint_list = (
                "nose",
                "left_eye", "right_eye",
                "left_ear", "right_ear",
                "left_shoulder", "right_shoulder",
                "left_elbow", "right_elbow",
                "left_wrist", "right_wrist",
                "left_hip", "right_hip",
                "left_knee", "right_knee",
                "left_ankle", "right_ankle"
                )


def send_pose(keypoints: np.ndarray):
    if keypoints is None:
        client.send_message(OSC_ADDRESS, 0)
        return

    # create message and send
    for i in range(len(keypoint_list)):
        builder = OscMessageBuilder(address=(OSC_ADDRESS + "/" + keypoint_list[i]))
        builder.add_arg(float(keypoints[0][i][0] / WINDOW_SIZE_H_W[1])) # x
        builder.add_arg(float(keypoints[0][i][1] / WINDOW_SIZE_H_W[0])) # y
        builder.add_arg(float(keypoints[0][i][2]))                      # z
        msg = builder.build()
        client.send(msg)
        # print(msg)

def ai_output_tensor_parse(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    global last_boxes, last_scores, last_keypoints
    np_outputs = imx500.get_outputs(metadata=metadata, add_batch=True)
    if np_outputs is not None:
        keypoints, scores, boxes = postprocess_higherhrnet(outputs=np_outputs,
                                                           img_size=WINDOW_SIZE_H_W,
                                                           img_w_pad=(0, 0),
                                                           img_h_pad=(0, 0),
                                                           detection_threshold=args.detection_threshold,
                                                           network_postprocess=True)

        if scores is not None and len(scores) > 0:
            last_keypoints = np.reshape(np.stack(keypoints, axis=0), (len(scores), 17, 3))
            last_boxes = [np.array(b) for b in boxes]
            last_scores = np.array(scores)

    return last_boxes, last_scores, last_keypoints


def ai_output_tensor_draw(request: CompletedRequest, boxes, scores, keypoints, stream='main'):
    """Draw the detections for this request onto the ISP output."""
    with MappedArray(request, stream) as m:
        if boxes is not None and len(boxes) > 0:
            drawer.annotate_image(m.array, boxes, scores,
                                  np.zeros(scores.shape), keypoints, args.detection_threshold,
                                  args.detection_threshold, request.get_metadata(), picam2, stream)


def picamera2_pre_callback(request: CompletedRequest):
    """Analyse the detected objects in the output tensor and draw them on the main output image."""
    boxes, scores, keypoints = ai_output_tensor_parse(request.get_metadata())
    ai_output_tensor_draw(request, boxes, scores, keypoints)
    send_pose(keypoints)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--detection-threshold", type=float, default=0.2,
                        help="Post-process detection threshold")
    parser.add_argument("--labels", type=str,
                        help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    return parser.parse_args()


def get_drawer():
    categories = intrinsics.labels
    categories = [c for c in categories if c and c != "-"]
    return COCODrawer(categories, imx500, needs_rescale_coords=False)


if __name__ == "__main__":
    args = get_args()

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "pose estimation"
    elif intrinsics.task != "pose estimation":
        print("Network is not a pose estimation task", file=sys.stderr)
        exit()

    # Override intrinsics from args
    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    if intrinsics.inference_rate is None:
        intrinsics.inference_rate = 10
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    drawer = get_drawer()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={'FrameRate': intrinsics.inference_rate}, buffer_count=12)

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=False)
    imx500.set_auto_aspect_ratio()
    picam2.pre_callback = picamera2_pre_callback

    while True:
        time.sleep(0.5)
