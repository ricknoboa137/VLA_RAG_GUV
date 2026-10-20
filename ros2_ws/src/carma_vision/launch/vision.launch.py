"""Start the analysis and stream nodes with one parameter file.

``mqtt_camera:=true`` also starts the MQTT camera source, for testing without a
camera attached to the container (see ``scripts/webcam_mqtt.py``).
``arm_camera:=true`` starts a second one for the stereo camera on the arm
(MQTT ``carma/arm_source/*`` to ``/carma/arm/stereo/*``).
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
    stream_overrides = {**overrides, "vr_topic": LaunchConfiguration("vr_topic")}
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
            DeclareLaunchArgument("arm_camera", default_value="false"),
            # MqttVidFeed feeds the Quest client from the stream node. ros2 launch
            # rejects an empty value, so "none" disables it ("off" would be read
            # as a boolean).
            DeclareLaunchArgument("vr_topic", default_value="none"),
            Node(
                package="carma_vision",
                executable="mqtt_camera_node",
                name="carma_mqtt_arm_camera",
                parameters=[params, overrides],
                output="screen",
                condition=IfCondition(LaunchConfiguration("arm_camera")),
            ),
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
                parameters=[params, stream_overrides],
                output="screen",
            ),
        ]
    )
