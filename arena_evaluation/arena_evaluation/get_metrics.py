#!/usr/bin/env python3

import argparse
import os
import arena_evaluation.scripts.metrics as Metrics
from ament_index_python.packages import get_package_share_directory


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        "-d",
        required=True,
        help=(
            "Directory where the data is stored. "
            "Can be either an absolute/relative path, or a directory name under "
            "<share>/arena_evaluation/data/<dir>."
        ),
    )
    parser.add_argument(
        "--world-name",
        "-w",
        default=None,
        help="World name used to load <world>/metrics/default.yaml and <world>/world.yaml (required for --subject/--zone).",
    )
    parser.add_argument(
        "--pedsim",
        action="store_true",
        default=False,
        help="Enable Pedsim metrics for the base metrics.csv output",
    )
    parser.add_argument(
        "--subject",
        action="store_true",
        default=False,
        help="Compute subject-aware metrics and write metrics_subject.csv",
    )
    parser.add_argument(
        "--zone",
        action="store_true",
        default=False,
        help="Compute zone-aware metrics and write metrics_zone.csv",
    )
    arguments = parser.parse_args()

    # Resolve directory:
    # - if arguments.dir exists as a path, use it
    # - otherwise treat it as a directory name under arena_evaluation share/data
    if os.path.exists(arguments.dir):
        dir_arg = os.path.abspath(arguments.dir)
    else:
        dir_arg = os.path.join(
            get_package_share_directory("arena_evaluation"),
            "data",
            arguments.dir,
        )

    if not os.path.isdir(dir_arg):
        raise FileNotFoundError(f"Data directory not found: '{dir_arg}'")

    if arguments.pedsim:
        metrics = Metrics.PedsimMetrics(dir=dir_arg)
    else:
        metrics = Metrics.Metrics(dir=dir_arg)

    # Save the calculated base metrics to a CSV file
    metrics.data.to_csv(os.path.join(dir_arg, "metrics.csv"))

    if arguments.subject or arguments.zone:
        if not arguments.world_name:
            raise ValueError("--world-name/-w is required when using --subject and/or --zone")

    if arguments.subject:
        subject_metrics = Metrics.SubjectAwareMetrics(
            dir=dir_arg,
            world_name=arguments.world_name,
        )
        subject_metrics.data.to_csv(os.path.join(dir_arg, "metrics_subject.csv"))

    if arguments.zone:
        zone_metrics = Metrics.ZoneAwareMetrics(
            dir=dir_arg,
            world_name=arguments.world_name,
        )
        zone_metrics.data.to_csv(os.path.join(dir_arg, "metrics_zone.csv"))

if __name__ == "__main__":
    main()