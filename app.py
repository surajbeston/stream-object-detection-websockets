import websockets
import asyncio
import cv2
import base64
from ultralytics import YOLO
import numpy as np
import json

import os

from time import time

from websockets.exceptions import ConnectionClosedError

class StreamClosedError(Exception):
    def __init__(self):
        super().__init__("Stream is not open.")

class CannotReadStreamError(Exception):
    def __init__(self):
        super().__init__("Attempt to read the stream was unsuccessful.")

class Stream:
    def __init__(self, video_path: str) -> None:
        self._video_path = video_path

    def __enter__(self) -> None:
        self.cap = cv2.VideoCapture(self._video_path)
        return self
    
    def __exit__(self, *args) -> None:
        self.cap.release()

video_path = os.getenv("STREAM_URL")

class ObjectDetect:
    def __init__(self) -> None:
        self.model = YOLO('yolov5nu.pt')

    def read(self, stream: Stream) ->  np.ndarray:  # type: ignore
        if not stream.cap.isOpened():
            raise StreamClosedError
        success, frame =  stream.cap.read()
        if not success:
            raise CannotReadStreamError
        return frame

    def detect(self, stream: Stream) -> np.ndarray:
        frame = self.read(stream)
        results = self.model.track(frame, persist=True)
        return results
    
    def detect_and_annotate(self, stream: Stream) -> np.ndarray:
        results = self.detect(stream)
        classes = results[0].boxes.cls
        names = []
        for i in classes:
            names.append(results[0].names[int(i)])
        return names, results[0].plot()
    
    def ndarr_to_string(self, arr) -> str:
        _, encoded_image = cv2.imencode('.jpg', arr)
        image_bytes = encoded_image.tobytes()
        encoded_bytes = base64.b64encode(image_bytes).decode('utf-8')
        return encoded_bytes

async def handler(websocket):
    # async for message in websocket:
    with Stream(video_path) as stream:
        detect = ObjectDetect()
        frames = 0
        t = time()
        while True:
            frames += 1
            try:
                labels, image = detect.detect_and_annotate(stream)
                try:
                    fps = frames / (time() - t)
                    data = {"fps": fps, "image": detect.ndarr_to_string(image), "labels": labels}
                    await websocket.send(json.dumps(data))
                except ConnectionClosedError:
                    break
            except StreamClosedError or CannotReadStreamError:
                break

async def main():
    async with websockets.serve(handler, "localhost", 8000):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Closing gracefully...")




