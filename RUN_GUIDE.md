# Hướng dẫn chạy Hệ thống Robot 5 Bậc Tự Do (5-DOF)

Hệ thống đã được nâng cấp lên 5 bậc tự do (Z-Y-Y-Y-Z) để giải quyết vấn đề hướng kẹp (Yaw). Giờ đây robot có thể xoay cổ tay để gắp các vật thể ở mọi góc độ, trong khi vẫn duy trì hướng kẹp thẳng đứng từ trên xuống.

## 1. Chuẩn bị môi trường

Mở một terminal mới (hoặc sử dụng terminator để mở nhiều tab).
Luôn nhớ phải tắt các tiến trình cũ trước khi chạy mới để tránh xung đột cổng hoặc lỗi "Trajectory goal rejected".

```bash
# Tắt mọi tiến trình ROS 2 và Gazebo cũ
killall -9 python3 ruby gz rviz2 robot_state_publisher parameter_bridge ros2 || true
```

## 2. Build lại workspace (Nếu có chỉnh sửa code)

```bash
cd ~/scratch/robot_3dof_ws
# Cấu hình môi trường ROS 2 Jazzy (bỏ qua conda nếu có)
CONDA_PREFIX="" PATH="/opt/ros/jazzy/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" PYTHONPATH="/opt/ros/jazzy/lib/python3.12/site-packages:/opt/ros/jazzy/local/lib/python3.12/dist-packages" source /opt/ros/jazzy/setup.bash

# Build package
colcon build --packages-select robot_5dof --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```

## 3. Khởi chạy Hệ thống

```bash
cd ~/scratch/robot_3dof_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Chạy toàn bộ hệ thống (Gazebo + RViz + Controllers + Tầm nhìn + Gắp vật)
ros2 launch robot_5dof grasp_demo.launch.py
```

## 4. Cách hệ thống hoạt động

Khi khởi chạy, các cửa sổ sau sẽ xuất hiện:
1. **Gazebo Harmonic**: Môi trường mô phỏng 3D với robot và các vật thể trên bàn.
2. **RViz2**: Giao diện trực quan hoá dữ liệu ROS. Hiển thị PointCloud2 (Dữ liệu 3D từ camera) và các marker (Grasp poses, Bounding boxes).

**Tiến trình Pick-and-Place:**
- Node `grasp_detection_node` liên tục lấy dữ liệu từ Depth Camera, dùng thuật toán RANSAC để loại bỏ mặt bàn, sau đó dự đoán các vị trí gắp khả thi (X, Y, Z, Yaw).
- Node `pick_place_node` nhận tín hiệu gắp tốt nhất, tính toán Động học ngược (Inverse Kinematics) cho 5 khớp.
- Trình tự thực thi: `Mở kẹp -> Di chuyển mượt đến vị trí chờ (Pre-grasp) -> Đưa tay xuống -> Kẹp vật -> Nhấc lên -> Quay về Home -> Thả vật`.
- Góc Yaw của vật thể giờ đây được khớp số 5 (Wrist Roll) bù trừ hoàn hảo.
