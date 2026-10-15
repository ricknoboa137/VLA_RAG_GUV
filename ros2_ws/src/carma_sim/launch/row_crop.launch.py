"""Bring up the row-crop world, headless, with the ROS bridge attached.

    ros2 launch carma_sim row_crop.launch.py
    ros2 launch carma_sim row_crop.launch.py stage:=CANOPY_CLOSURE
    ros2 launch carma_sim row_crop.launch.py headless:=false        # needs a display

``stage`` is the one parameter the drift study varies. It takes the names of
``carma.types.PhenologyStage`` and is validated against that list here, so a
typo fails the launch instead of quietly producing a world at the default
stage: a run recorded as CANOPY_CLOSURE that actually rendered VEGETATIVE would
be an unreproducible number, which AGENTS.md section 2.2 treats as a defect of
the same severity as a crash.

``worlds/row_crop.sdf`` is a xacro template. It is expanded here, once, into
``<log_dir>/row_crop_<stage>.sdf``, and that expanded file is what Gazebo
loads. Keeping the expansion next to the log means the exact world an episode
ran against is recoverable after the fact.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    SetEnvironmentVariable,
    Shutdown,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# The names of carma.types.PhenologyStage. Duplicated rather than imported:
# ros2_ws code may import carma, but a launch file that fails to start because
# the library is not on the path is a worse failure than a stale list, and this
# list changing is a deliberate act (see the enum's docstring).
PHENOLOGY_STAGES = (
    "BARE_SOIL",
    "EMERGENCE",
    "VEGETATIVE",
    "CANOPY_CLOSURE",
    "SENESCENCE",
    "POST_HARVEST",
)

_PACKAGE = "carma_sim"


def _expand_world(context: LaunchContext) -> list[ExecuteProcess | Node]:
    """Expand the world template at the requested stage and start Gazebo."""
    share = Path(get_package_share_directory(_PACKAGE))

    stage = LaunchConfiguration("stage").perform(context)
    if stage not in PHENOLOGY_STAGES:
        raise ValueError(
            f"stage:={stage!r} is not a phenological stage. "
            f"Expected one of {', '.join(PHENOLOGY_STAGES)}."
        )

    template = Path(LaunchConfiguration("world_template").perform(context))
    out_dir = Path(LaunchConfiguration("expanded_dir").perform(context))
    out_dir.mkdir(parents=True, exist_ok=True)
    expanded = out_dir / f"row_crop_{stage}.sdf"

    xacro = shutil.which("xacro")
    if xacro is None:
        raise RuntimeError(
            "xacro is not installed, so the world template cannot be expanded. "
            "Install ros-$ROS_DISTRO-xacro, or use the docker/ sim image, which "
            "has it."
        )

    args = [
        xacro,
        str(template),
        f"stage:={stage}",
        f"rows:={LaunchConfiguration('rows').perform(context)}",
        f"row_spacing:={LaunchConfiguration('row_spacing').perform(context)}",
        f"row_length:={LaunchConfiguration('row_length').perform(context)}",
        f"obstacles:={LaunchConfiguration('obstacles').perform(context)}",
    ]
    # A template error must stop the launch, not start Gazebo on a stale
    # expansion left over from the previous run. xacro puts the useful part of
    # a failure on stderr, so surface that rather than just the exit code.
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"expanding {template} at stage {stage} failed "
            f"(xacro exit {result.returncode}):\n{result.stderr.strip()}"
        )
    expanded.write_text(result.stdout)

    headless = LaunchConfiguration("headless").perform(context).lower() in ("true", "1")
    verbosity = LaunchConfiguration("verbosity").perform(context)

    # -r run on start, -s server only, --headless-rendering renders the cameras
    # through EGL with no display, which is what makes this work in a container
    # and in CI.
    gz_args = ["gz", "sim", "-r", "-v", verbosity]
    if headless:
        gz_args += ["-s", "--headless-rendering"]
    gz_args.append(str(expanded))

    return [
        ExecuteProcess(
            cmd=gz_args,
            output="screen",
            # Gazebo's exit takes the whole launch down with it; otherwise the
            # bridge lingers, subscribed to topics nothing publishes any more.
            on_exit=[Shutdown(reason="gz sim exited")],
        ),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="carma_gz_bridge",
            output="screen",
            parameters=[
                {
                    "config_file": str(share / "config" / "bridge.yaml"),
                    "use_sim_time": True,
                }
            ],
        ),
        # Gazebo cannot render BGR, so the stereo pair arrives as rgb8 on
        # .../image_rgb and this node swaps the channels onto the contract's
        # .../image_raw. See scripts/bgr_relay.py.
        Node(
            package="carma_sim",
            executable="bgr_relay.py",
            name="carma_bgr_relay",
            output="screen",
            parameters=[{"use_sim_time": True}],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    """Declare the world arguments and bring the simulation up."""
    share = Path(get_package_share_directory(_PACKAGE))

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "stage",
                default_value="VEGETATIVE",
                description=(
                    "Crop phenological stage. One of: " + ", ".join(PHENOLOGY_STAGES)
                ),
            ),
            DeclareLaunchArgument(
                "headless",
                default_value="true",
                description="Run the server only, rendering cameras without a display.",
            ),
            DeclareLaunchArgument("rows", default_value="8"),
            DeclareLaunchArgument(
                "row_spacing",
                default_value="1.5",
                description="Metres between row centres. Fixed across stages.",
            ),
            DeclareLaunchArgument("row_length", default_value="24.0"),
            DeclareLaunchArgument("obstacles", default_value="true"),
            DeclareLaunchArgument("verbosity", default_value="2"),
            DeclareLaunchArgument(
                "world_template",
                default_value=str(share / "worlds" / "row_crop.sdf"),
            ),
            DeclareLaunchArgument(
                "expanded_dir",
                default_value=str(Path.home() / ".ros" / "log" / "carma_sim"),
                description="Where the expanded world is written, for the record.",
            ),
            # model://carma_ugv in the world resolves against this. Prepended
            # rather than assigned, so a caller's own resource path survives.
            SetEnvironmentVariable(
                "GZ_SIM_RESOURCE_PATH",
                os.pathsep.join(
                    p for p in (str(share / "models"), os.environ.get("GZ_SIM_RESOURCE_PATH")) if p
                ),
            ),
            OpaqueFunction(function=_expand_world),
        ]
    )
