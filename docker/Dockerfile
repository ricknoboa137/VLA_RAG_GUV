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


# ---------------------------------------------------------------------------
# The Python library on its own, for colleagues who want to run experiments
# rather than the robot. No ROS: a tenth of the size, and nothing to install
# on the host but Docker.
#
#   docker compose -f docker/compose.yaml build lib
#   docker compose -f docker/compose.yaml run --rm lib carma run --config configs/experiment/smoke.yaml
#
# The base image is pinned by digest, not by tag. "python:3.11-slim-bookworm"
# is rebuilt upstream, so the same Dockerfile would otherwise produce a
# different image next month and a run would not be reproducible from its
# commit alone. Update deliberately, with the digest the pull reports.
FROM python@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89 AS lib
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --no-deps with a fully pinned lock: the resolver cannot pull in anything the
# lock does not name, so two builds of this commit install the same packages.
COPY docker/requirements-lib.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --no-deps -r /tmp/requirements.lock \
    && rm /tmp/requirements.lock

COPY pyproject.toml README.md /opt/carma/
COPY src /opt/carma/src
RUN pip install --no-cache-dir --no-deps /opt/carma

COPY configs /opt/carma/configs
COPY scripts /opt/carma/scripts
# The camera calibration the scripts read, and the measurements behind the
# published depth figures, so the evidence travels with the library.
COPY config /opt/carma/config
COPY experiments /opt/carma/experiments
WORKDIR /opt/carma

# Fail the build rather than ship an image whose library does not import.
RUN carma components > /dev/null && python -c "import carma, carma.cli.main; print('carma', carma.__version__)"

CMD ["carma", "--help"]


# The same library plus the gate tools, so the checks a colleague runs locally
# run identically in a container: make docker-check.
FROM lib AS dev
# make, so the container runs the same Makefile a colleague runs locally
# rather than a second copy of the gate commands that could drift from it.
RUN apt-get update && apt-get install -y --no-install-recommends make     && rm -rf /var/lib/apt/lists/*
COPY docker/requirements-dev.lock /tmp/requirements-dev.lock
RUN pip install --no-cache-dir --no-deps -r /tmp/requirements-dev.lock \
    && rm /tmp/requirements-dev.lock
COPY tests /opt/carma/tests
COPY Makefile /opt/carma/
CMD ["make", "check"]
