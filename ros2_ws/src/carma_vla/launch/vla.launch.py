"""Start the live VLA pipeline with its parameter file.

ros2 launch carma_vla vla.launch.py
ros2 launch carma_vla vla.launch.py primary:=world drive:=true
ros2 launch carma_vla vla.launch.py arm:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    params = LaunchConfiguration("params_file")
    overrides = {
        "policy_url": LaunchConfiguration("policy_url"),
        "primary": LaunchConfiguration("primary"),
        "drive": ParameterValue(LaunchConfiguration("drive"), value_type=bool),
        "arm": ParameterValue(LaunchConfiguration("arm"), value_type=bool),
    }
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("carma_vla"), "config", "vla.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "policy_url", default_value="http://host.docker.internal:8000/act"
            ),
            DeclareLaunchArgument("primary", default_value="arm"),
            DeclareLaunchArgument("drive", default_value="false"),
            DeclareLaunchArgument("arm", default_value="false"),
            Node(
                package="carma_vla",
                executable="vla_node",
                name="carma_vla",
                parameters=[params, overrides],
                output="screen",
            ),
        ]
    )
