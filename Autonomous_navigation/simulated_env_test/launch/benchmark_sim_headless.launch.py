import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = os.getenv("BENCHMARK_WS_PATH", os.getcwd())
    world_path = os.path.join(pkg_dir, "worlds", "room_world.sdf")
    bridge_config = os.path.join(pkg_dir, "config", "bridge_config.yaml")

    # Run gz sim server only (-s)
    gz_server = ExecuteProcess(
        cmd=["gz", "sim", "-s", "-r", world_path],
        output="screen"
    )

    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["--ros-args", "-p", f"config_file:={bridge_config}"],
        output="screen"
    )

    return LaunchDescription([gz_server, ros_gz_bridge])
