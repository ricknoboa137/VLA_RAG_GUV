# ROS 2 Jazzy + the carma library + the ROS workspace, in one reproducible image.
# Build from the repository root:  docker compose -f docker/compose.yaml build
#
# Targets:
#   base  ROS 2 nodes (bridge, vision). Small enough for the robot computer.
#   sim   base + Gazebo Harmonic via ros_gz, for the simulation study.

ARG ROS_DISTRO=jazzy

FROM ros:${ROS_DISTRO}-ros-base AS base
ARG ROS_DISTRO
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-numpy \
        python3-opencv \
        python3-yaml \
        python3-pytest \
        python3-colcon-common-extensions \
        ros-${ROS_DISTRO}-cv-bridge \
        ros-${ROS_DISTRO}-message-filters \
        ros-${ROS_DISTRO}-std-srvs \
        ros-${ROS_DISTRO}-nav-msgs \
    && rm -rf /var/lib/apt/lists/*

# Pinned, like everything in pyproject.toml. numpy is pinned to the apt version
# so faster-whisper's dependencies cannot replace the numpy cv_bridge was built
# against; the check fails the build if that ever happens.
RUN pip3 install --no-cache-dir --break-system-packages \
        numpy==1.26.4 \
        paho-mqtt==2.1.0 \
        faster-whisper==1.2.1 \
        huggingface-hub==0.36.0 \
    && python3 -c "import numpy, cv2, faster_whisper; \
assert numpy.__version__ == '1.26.4', numpy.__version__; \
print('numpy', numpy.__version__, 'opencv', cv2.__version__, 'faster-whisper', faster_whisper.__version__)"

# The carma library. --no-deps: numpy, OpenCV and PyYAML come from apt above so
# they match the versions cv_bridge was built against.
COPY pyproject.toml README.md /opt/carma/
COPY src /opt/carma/src
RUN pip3 install --no-cache-dir --break-system-packages --no-deps /opt/carma
COPY configs /opt/carma/configs

WORKDIR /ros2_ws
COPY ros2_ws/src ./src
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
    && colcon build --event-handlers console_direct-

# Transport profile for ~1 MB raw images; see the comment inside the file.
COPY docker/fastdds_shm.xml /etc/carma/fastdds_shm.xml
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/etc/carma/fastdds_shm.xml

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]


FROM base AS sim
ARG ROS_DISTRO
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-${ROS_DISTRO}-ros-gz \
    && rm -rf /var/lib/apt/lists/*
