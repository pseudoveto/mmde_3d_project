import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = os.getenv("BENCHMARK_WS_PATH", os.getcwd())
    world_path = os.path.join(pkg_dir, "worlds", "room_world.sdf")
    bridge_config = os.path.join(pkg_dir, "config", "bridge_config.yaml")

    headless_arg = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Run Gazebo in headless mode (server only)"
    )

    # gz sim execution
    gz_args = ["-r", world_path]
    if LaunchConfiguration("headless") == "true":
        gz_args.insert(0, "-s")

    gz_sim = ExecuteProcess(
        cmd=["gz", "sim"] + gz_args,
        output="screen"
    )

    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["--ros-args", "-p", f"config_file:={bridge_config}"],
        output="screen"
    )

    return LaunchDescription([
        headless_arg,
        gz_sim,
        ros_gz_bridge
    ])
