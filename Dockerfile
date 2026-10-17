# Multi-stage Dockerfile for ARMROBOT: 4-DOF Robotic Manipulation & Grasp Synthesis
# Base image: ROS 2 Jazzy on Ubuntu 24.04

FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies & ROS 2 control / simulation packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-pytest \
    python3-numpy \
    python3-scipy \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-xacro \
    ros-jazzy-joint-state-broadcaster \
    ros-jazzy-joint-trajectory-controller \
    ros-jazzy-controller-manager \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
WORKDIR /workspace
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy repository source code
COPY . /workspace/ARMROBOT
WORKDIR /workspace/ARMROBOT

# Build ROS 2 workspace
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && \
    colcon build --symlink-install --packages-select robot_arm --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3"

# Default entrypoint runs unit tests and standalone CPU geometric benchmark
CMD ["/bin/bash", "-c", "source /opt/ros/jazzy/setup.bash && source install/setup.bash && PYTHONPATH=./:src/robot_arm pytest -v tests/ && python3 benchmark/run_academic_benchmark.py --trials 5"]
