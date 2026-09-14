"""Start the analysis and stream nodes with one parameter file.

``mqtt_camera:=true`` also starts the MQTT camera source, for testing without a
camera attached to the container (see ``scripts/webcam_mqtt.py``).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    params = LaunchConfiguration("params_file")
    overrides = {"mqtt_host": LaunchConfiguration("mqtt_host")}
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("carma_vision"), "config", "vision.yaml"]
                ),
            ),
            DeclareLaunchArgument("mqtt_host", default_value="broker"),
            DeclareLaunchArgument("mqtt_camera", default_value="false"),
            Node(
                package="carma_vision",
                executable="mqtt_camera_node",
                name="carma_mqtt_camera",
                parameters=[params, overrides],
                output="screen",
                condition=IfCondition(LaunchConfiguration("mqtt_camera")),
            ),
            Node(
                package="carma_vision",
                executable="analysis_node",
                name="carma_analysis",
                parameters=[params, overrides],
                output="screen",
            ),
            Node(
                package="carma_vision",
                executable="stream_node",
                name="carma_stream",
                parameters=[params, overrides],
                output="screen",
            ),
        ]
    )
