"""Start the speech-to-text node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("carma_voice"), "config", "voice.yaml"]
                ),
            ),
            DeclareLaunchArgument("mqtt_host", default_value="host.docker.internal"),
            # ros2 launch rejects an empty "name:=" value, so auto-detect is "auto".
            DeclareLaunchArgument("language", default_value="auto"),
            DeclareLaunchArgument("languages", default_value="en,es"),
            Node(
                package="carma_voice",
                executable="stt_node",
                name="carma_stt",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "mqtt_host": LaunchConfiguration("mqtt_host"),
                        "language": LaunchConfiguration("language"),
                        "languages": LaunchConfiguration("languages"),
                    },
                ],
                output="screen",
            ),
        ]
    )
