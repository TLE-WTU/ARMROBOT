#!/usr/bin/env python3
"""
AnyGrasp IPC Inference Service.

Runs inside Conda environment 'robot_env' (Python 3.10 with MinkowskiEngine, PyTorch, CUDA).
Loads AnyGrasp model into GPU VRAM once on startup, and listens on a local Unix Domain Socket
for inference requests from ROS 2 Jazzy nodes.
"""

import argparse
import os
import pickle
import signal
import socket
import struct
import sys
import numpy as np

# Ensure anygrasp_sdk is in python path
ANYGRASP_DIR = "/home/tienle/anygrasp_sdk/grasp_detection"
if ANYGRASP_DIR not in sys.path:
    sys.path.insert(0, ANYGRASP_DIR)

# Switch working directory so gsnet finds license/licenseCfg.json
os.chdir(ANYGRASP_DIR)

from gsnet import create_detector


class AnyGraspService:
    def __init__(self, checkpoint_path: str, socket_path: str = "/tmp/anygrasp_ipc.sock",
                 max_gripper_width: float = 0.06, gripper_height: float = 0.04):
        self.checkpoint_path = checkpoint_path
        self.socket_path = socket_path
        self.max_gripper_width = max_gripper_width
        self.gripper_height = gripper_height
        self.server_sock = None
        self.running = True

        # Load AnyGrasp Model
        print(f"[AnyGrasp Service] Initializing detector with checkpoint: {self.checkpoint_path}")
        class Config:
            pass
        cfgs = Config()
        cfgs.checkpoint_path = self.checkpoint_path
        cfgs.max_gripper_width = self.max_gripper_width
        cfgs.gripper_height = self.gripper_height
        cfgs.top_n = 10
        cfgs.top_n_vis = 10
        cfgs.seed = 1
        cfgs.fp16 = False

        self.detector = create_detector(cfgs)
        if self.detector is None:
            raise RuntimeError("Failed to create AnyGrasp detector! Check license and checkpoint.")
        print("✅ [AnyGrasp Service] AnyGrasp detector successfully loaded into GPU memory!")

    def start(self):
        # Clean up stale socket file if exists
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

        self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_sock.bind(self.socket_path)
        self.server_sock.listen(5)
        print(f"🚀 [AnyGrasp Service] Listening for inference requests on {self.socket_path}")

        # Register signal handlers for clean exit
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        while self.running:
            try:
                conn, _ = self.server_sock.accept()
                self._handle_client(conn)
            except (socket.error, KeyboardInterrupt):
                break

        self.cleanup()

    def _handle_client(self, conn: socket.socket):
        try:
            # Read 4-byte length prefix
            raw_len = conn.recv(4)
            if not raw_len:
                conn.close()
                return
            msg_len = struct.unpack("!I", raw_len)[0]

            # Read full payload
            data = bytearray()
            while len(data) < msg_len:
                packet = conn.recv(min(65536, msg_len - len(data)))
                if not packet:
                    break
                data.extend(packet)

            request = pickle.loads(data)
            points = request.get("points")
            optional_params = request.get("optional_params", {
                "dense_grasp": False,
                "collision_detection": False,
                "approach_steering": [0, 0, -1],  # Top-down grasp
                "approach_thresh": 0.3,
            })

            if points is None or len(points) < 10:
                response = {"status": "ok", "grasps": []}
            else:
                # Downsample if too dense (> 3000 points) to keep GPU inference < 60ms
                if len(points) > 3000:
                    indices = np.random.choice(len(points), size=3000, replace=False)
                    points = points[indices]

                points = np.asarray(points, dtype=np.float32)

                # Run inference
                gg = self.detector.get_grasp(points, optional_params)

                grasps = []
                if gg is not None and len(gg) > 0:
                    try:
                        gg = gg.nms()
                    except Exception:
                        pass
                    gg = gg.sort_by_score()
                    # Return top 10 grasps
                    for g in gg[:10]:
                        grasps.append({
                            "translation": g.translation.tolist(),
                            "rotation": g.rotation_matrix.tolist(),
                            "score": float(g.score),
                            "width": float(g.width),
                        })

                response = {"status": "ok", "grasps": grasps}

            # Send response
            resp_data = pickle.dumps(response, protocol=4)
            conn.sendall(struct.pack("!I", len(resp_data)) + resp_data)

        except Exception as e:
            print(f"[AnyGrasp Service] Client error: {e}")
            err_resp = pickle.dumps({"status": "error", "message": str(e)}, protocol=4)
            try:
                conn.sendall(struct.pack("!I", len(err_resp)) + err_resp)
            except Exception:
                pass
        finally:
            conn.close()

    def _handle_signal(self, signum, frame):
        print("\n[AnyGrasp Service] Shutting down service gracefully...")
        self.running = False
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        if self.server_sock:
            try:
                self.server_sock.close()
            except Exception:
                pass
        if os.path.exists(self.socket_path):
            try:
                os.remove(self.socket_path)
            except Exception:
                pass
        print("[AnyGrasp Service] Service stopped and socket cleaned up.")


def main():
    parser = argparse.ArgumentParser(description="AnyGrasp IPC Inference Service")
    parser.add_argument(
        "--checkpoint_path",
        default="/home/tienle/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar",
        help="Path to checkpoint_detection.tar"
    )
    parser.add_argument(
        "--socket_path",
        default="/tmp/anygrasp_ipc.sock",
        help="Unix socket path for IPC"
    )
    parser.add_argument("--max_gripper_width", type=float, default=0.06)
    parser.add_argument("--gripper_height", type=float, default=0.04)

    args = parser.parse_args()
    service = AnyGraspService(
        checkpoint_path=args.checkpoint_path,
        socket_path=args.socket_path,
        max_gripper_width=args.max_gripper_width,
        gripper_height=args.gripper_height,
    )
    service.start()


if __name__ == "__main__":
    main()
