import gi
import cv2
import numpy as np

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class Tracker:
    def __init__(self):
        self.trackers = {}
        self.locked_object_id = None
        self.manual_tracker = None

    def lock_object(self, frame, bbox):
        """Lock an object for tracking"""
        self.manual_tracker = cv2.TrackerCSRT_create()
        self.manual_tracker.init(frame, bbox)

    def track_object(self, frame):
        """Track the manually selected object"""
        if self.manual_tracker is not None:
            success, bbox = self.manual_tracker.update(frame)
            if success:
                return [int(v) for v in bbox]
        return None

    def calculate_distance(self, object_size, focal_length, real_object_size):
        """Estimate the distance of the object"""
        distance = (real_object_size * focal_length) / object_size
        return distance

    def adjust_bbox(self, bbox, distance, max_distance=1000):
        """Dynamically adjust the bounding box size based on distance"""
        scale_factor = max(1, distance / max_distance)
        new_bbox = (int(bbox[0] - scale_factor * 10), int(bbox[1] - scale_factor * 10),
                    int(bbox[2] + scale_factor * 20), int(bbox[3] + scale_factor * 20))
        return new_bbox

def on_message(bus, message, loop):
    message_type = message.type
    if message_type == Gst.MessageType.EOS:
        print("End-Of-Stream reached")
        loop.quit()
    elif message_type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}, {debug}")
        loop.quit()

def main():
    # Initialize GStreamer
    Gst.init(None)

    # Create the pipeline
    pipeline = Gst.parse_launch(
        "udpsrc port=5600 ! application/x-rtp, payload=96 ! "
        "rtph264depay ! avdec_h264 ! videoconvert ! video/x-raw,format=RGB ! appsink name=mysink"
    )

    # Create a GLib MainLoop to handle the bus messages
    loop = GLib.MainLoop()

    # Get the bus to handle messages from GStreamer
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)

    # Get the appsink
    appsink = pipeline.get_by_name("mysink")

    # Start playing the pipeline
    pipeline.set_state(Gst.State.PLAYING)

    # Initialize tracker
    tracker = Tracker()
    manual_tracking_active = False
    bbox_manual = None

    # Camera parameters
    focal_length = 800
    real_object_size = 5

    def on_mouse_click(event, x, y, flags, param):
        nonlocal manual_tracking_active, bbox_manual
        if event == cv2.EVENT_LBUTTONDOWN:
            bbox_manual = (x - 25, y - 25, 50, 50)
            tracker.lock_object(frame, bbox_manual)
            manual_tracking_active = True

    cv2.namedWindow("GStreamer Video Stream")
    cv2.setMouseCallback("GStreamer Video Stream", on_mouse_click)

    try:
        print("Streaming video... Press Ctrl+C to stop.")
        while True:
            sample = appsink.emit("pull-sample")
            if sample:
                buffer = sample.get_buffer()
                caps = sample.get_caps()

                # Extract frame dimensions from caps
                width = caps.get_structure(0).get_value('width')
                height = caps.get_structure(0).get_value('height')

                # Convert GStreamer sample to OpenCV frame
                arr = np.ndarray((height, width, 3), dtype=np.uint8,
                                 buffer=buffer.extract_dup(0, buffer.get_size()))
                frame = arr.copy()

                # Manual tracking
                if manual_tracking_active:
                    bbox_manual = tracker.track_object(frame)
                    
                    if bbox_manual:    
                        object_size_in_image = bbox_manual[2]
                        distance = tracker.calculate_distance(object_size_in_image, focal_length, real_object_size)
                        
                        bbox_manual = tracker.adjust_bbox(bbox_manual, distance)

                        cv2.rectangle(frame,
                                     (bbox_manual[0], bbox_manual[1]),
                                     (bbox_manual[0] + bbox_manual[2], bbox_manual[1] + bbox_manual[3]),
                                     (255, 0, 0), 2)
                        
                        cv2.putText(frame,
                                   f"Manual Tracking - Distance: {distance:.2f} m", 
                                   (bbox_manual[0], bbox_manual[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   0.5,
                                   (255, 0, 0), 
                                   2)

                cv2.imshow("GStreamer Video Stream", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("u"):
                    manual_tracking_active = False
                    tracker.manual_tracker = None

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        # Stop the pipeline and clean up
        pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
