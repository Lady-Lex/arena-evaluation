"""
This file is used to calculate from the simulation data, various metrics, such as
- did a collision occur
- how long did the robot take form start to goal
the metrics / evaluation data will be saved to be preproccesed in the next step
"""
import enum
import typing
from typing import List
import numpy as np
import pandas as pd
import os
from pandas.core.api import DataFrame as DataFrame
import yaml
import rospkg
import json

# Import shapely library for geometric calculations
try:
    from shapely.geometry import Point, Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    print("Warning: shapely library not available, falling back to manual polygon detection")
    SHAPELY_AVAILABLE = False

from arena_evaluation.utils import Utils

class Action(str, enum.Enum):
    STOP = "STOP"
    ROTATE = "ROTATE"
    MOVE = "MOVE"


class DoneReason(str, enum.Enum):
    TIMEOUT = "TIMEOUT"
    GOAL_REACHED = "GOAL_REACHED"
    COLLISION = "COLLISION"


class Metric(typing.TypedDict):

    time: typing.List[int]
    time_diff: int
    episode: int
    goal: typing.List
    start: typing.List

    path: typing.List
    path_length_values: typing.List
    path_length: float
    angle_over_length: float
    curvature: typing.List
    normalized_curvature: typing.List 
    roughness: typing.List

#    cmd_vel: typing.List
    velocity: typing.List
    acceleration: typing.List
    jerk: typing.List
    
    collision_amount: int
    collisions: typing.List
    
#    action_type: typing.List[Action]
    result: DoneReason

class PedsimMetric(Metric, typing.TypedDict):

    num_pedestrians: int

    avg_velocity_in_personal_space: float
    total_time_in_personal_space: int
    time_in_personal_space: typing.List[int]

    total_time_looking_at_pedestrians: int
    time_looking_at_pedestrians: typing.List[int]

    total_time_looked_at_by_pedestrians: int
    time_looked_at_by_pedestrians: typing.List[int]


class SubjectMetric(typing.TypedDict):
    
    subject_id: str
    subject_type: str  # e.g., "doctor", "patient", "bed", "wheelchair"
    subject_position: typing.List[typing.List[float]]  # positions over time
    robot_subject_distances: typing.List[float]  # distances to subject over time
    subject_scores: typing.List[float]  # computed scores over time
    total_subject_score: float  # aggregated score for this subject
    scoring_function_type: str  # e.g., "u_curve", "distance_penalty", "proximity_reward"


class SubjectAwareMetric(Metric, typing.TypedDict):
    overall_subject_score: float  # aggregated score across all subjects
    average_subject_score: float  # average score per subject (overall / actual_subject_count)
    subject_violation_count: int    # Total subject violation count


class ZoneViolation(typing.TypedDict):
    
    timestamp: int                 # Violation timestamp
    position: typing.List[float]   # Violation position [x, y]
    duration: int                  # Violation duration
    severity: float               # Violation severity (based on zone type)
    violation_type: str           # Violation type


class ZoneAwareMetric(Metric, typing.TypedDict):
    overall_zone_score: float     # Overall zone score
    zone_violation_count: int    # Total zone violation count
    zone_violation_time: int      # Total zone violation time
    
    
class Config:
    TIMEOUT_TRESHOLD = 180e9
    MAX_COLLISIONS = 3
    MIN_EPISODE_LENGTH = 5
    
    PERSONAL_SPACE_RADIUS = 1 # personal space is estimated at around 1'-4'
    ROBOT_GAZE_ANGLE = np.radians(5) # size of (one half of) direct robot gaze cone
    PEDESTRIAN_GAZE_ANGLE = np.radians(5) # size of (one half of) direct ped gaze cone

class Math:

    @classmethod
    def round_values(cls, values, digits=3):
        return [round(v, digits) for v in values]

    @classmethod
    def grouping(cls, base: np.ndarray, size: int) -> np.ndarray:
        return np.moveaxis( 
            np.array([
                np.roll(base, i, 0)
                for i
                in range(size)
            ]),
            [1],
            [0]
        )[:-size]

    @classmethod
    def triangles(cls, position: np.ndarray) -> np.ndarray:
        return cls.grouping(position, 3)
    
    @classmethod
    def triangle_area(cls, vertices: np.ndarray) -> np.ndarray:
        return np.linalg.norm(
            np.cross(
                vertices[:,1] - vertices[:,0],
                vertices[:,2] - vertices[:,0],
                axis=1
            ),
            axis=1
        ) / 2
    
    @classmethod
    def path_length(cls, position: np.ndarray) -> np.ndarray:
        pairs = cls.grouping(position, 2)
        return np.linalg.norm(pairs[:,0,:] - pairs[:,1,:], axis=1)

    @classmethod
    def curvature(cls, position: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:

        triangles = cls.triangles(position)

        d01 = np.linalg.norm(triangles[:,0,:] - triangles[:,1,:], axis=1)
        d12 = np.linalg.norm(triangles[:,1,:] - triangles[:,2,:], axis=1)
        d20 = np.linalg.norm(triangles[:,2,:] - triangles[:,0,:], axis=1)

        triangle_area = cls.triangle_area(triangles)
        divisor = np.prod([d01, d12, d20], axis=0)
        divisor[divisor==0] = np.nan

        curvature = 4 * triangle_area / divisor
        curvature[np.isnan(divisor)] = 0

        normalized = np.multiply(
            curvature,
            d01 + d12
        )

        return curvature, normalized

    @classmethod
    def roughness(cls, position: np.ndarray) -> np.ndarray:
        
        triangles = cls.triangles(position)

        triangle_area = cls.triangle_area(triangles)
        length = np.linalg.norm(triangles[:,:,0] - triangles[:,:,2], axis=1)
        length[length==0] = np.nan

        roughness = 2 * triangle_area / np.square(length)
        roughness[np.isnan(length)] = 0

        return roughness

    @classmethod
    def acceleration(cls, speed: np.ndarray) -> np.ndarray:
        return np.diff(speed)

    @classmethod
    def jerk(cls, speed: np.ndarray) -> np.ndarray:
        return np.diff(np.diff(speed))

    @classmethod
    def turn(cls, yaw: np.ndarray) -> np.ndarray:
        pairs = cls.grouping(yaw, 2)
        return cls.angle_difference(pairs[:,0], pairs[:,1])
    
    @classmethod
    def angle_difference(cls, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
            return np.pi - np.abs(np.abs(x1 - x2) - np.pi)

    @classmethod
    def point_in_polygon(cls, point, polygon):
        """
        Use ray casting method to determine if a point is inside a polygon (fallback method)
        Use this method when the shapely library is not available
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

    # Social scoring mathematical functions
    
    @classmethod
    def u_curve_score(cls, distance: np.ndarray, optimal_distance: float, curve_width: float) -> np.ndarray:
        """
        Calculate U-curve scoring function for subjects like doctors.
        Score is highest at optimal_distance and decreases as distance moves away.
        
        Args:
            distance: distances to the subject
            optimal_distance: optimal distance for highest score
            curve_width: width parameter controlling the curve steepness
        
        Returns:
            scores based on U-curve function
        """
        deviation = np.abs(distance - optimal_distance)
        return 1.0 / (1.0 + (deviation / curve_width) ** 2)
    
    @classmethod
    def distance_penalty_score(cls, distance: np.ndarray, safe_distance: float, penalty_weight: float = 1.0) -> np.ndarray:
        """
        Calculate distance penalty scoring for subjects like patients.
        Higher penalty (lower score) when too close to the subject.
        
        Args:
            distance: distances to the subject  
            safe_distance: minimum safe distance
            penalty_weight: weight factor for penalty strength
        
        Returns:
            penalty scores (lower when too close)
        """
        violation = np.maximum(0, safe_distance - distance)
        return np.exp(-penalty_weight * violation / safe_distance)


class Metrics:

    dir: str
    _episode_data: typing.Dict[int, Metric]

    def _load_data(self) -> typing.List[pd.DataFrame]:
        episode = pd.read_csv(os.path.join(self.dir, "episode.csv"), converters={
            "data": lambda val: 0 if len(val) <= 0 else int(val) 
        })

        laserscan = pd.read_csv(os.path.join(self.dir, "scan.csv"), converters={
            "data": Utils.string_to_float_list
        }).rename(columns={"data": "laserscan"})

        odom = pd.read_csv(os.path.join(self.dir, "odom.csv"), converters={
            "data": lambda col: json.loads(col.replace("'", "\""))
        }).rename(columns={"data": "odom"})

#        cmd_vel = pd.read_csv(os.path.join(self.dir, "cmd_vel.csv"), converters={
#            "data": Utils.string_to_float_list
#        }).rename(columns={"data": "cmd_vel"})

        start_goal = pd.read_csv(os.path.join(self.dir, "start_goal.csv"), converters={
            "start": Utils.string_to_float_list,
            "goal": Utils.string_to_float_list
        })

        return [
            episode,
            laserscan,
            odom,
#            cmd_vel,
            start_goal
        ]

    def __init__(self, dir: str):

        self.dir = dir
        self.robot_params = self._get_robot_params()

        data = pd.concat(self._load_data(), axis=1, join="inner")
        data = data.loc[:,~data.columns.duplicated()].copy()

        i = 0

        episode_data = self._episode_data = {}

        while True:
            current_episode = data[data["episode"] == i]

            if len(current_episode) < Config.MIN_EPISODE_LENGTH:
                break
            
            episode_data[i] = self._analyze_episode(current_episode, i)
            i = i + 1

    @property
    def data(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self._episode_data).transpose().set_index("episode")

    
    def _analyze_episode(self, episode: pd.DataFrame, index) -> Metric:

        episode["time"] /= 10**10
        
        positions = np.array([frame["position"] for frame in episode["odom"]])
        velocities = np.array([frame["position"] for frame in episode["odom"]])

        curvature, normalized_curvature = Math.curvature(positions)
        roughness = Math.roughness(positions)

        vel_absolute = np.linalg.norm(velocities, axis=1)
        acceleration = Math.acceleration(vel_absolute)
        jerk = Math.jerk(vel_absolute)

        collisions, collision_amount = self._get_collisions(
            episode["laserscan"],
            self.robot_params["robot_radius"]
        )

        path_length = Math.path_length(positions)
        turn = Math.turn(positions[:,2])

        time = list(episode["time"])[-1] - list(episode["time"])[0]

        start_position = self._get_mean_position(episode, "start")
        goal_position = self._get_mean_position(episode, "goal")

        # print("PATH LENGTH", path_length, path_length_per_step)

        return Metric(
            curvature = Math.round_values(curvature),
            normalized_curvature = Math.round_values(normalized_curvature),
            roughness = Math.round_values(roughness),
            path_length_values = Math.round_values(path_length),
            path_length = path_length.sum(),
            acceleration = Math.round_values(acceleration),
            jerk = Math.round_values(jerk),
            velocity = Math.round_values(vel_absolute),
            collision_amount = collision_amount,
            collisions = list(collisions),
            path = [list(p) for p in positions],
            angle_over_length = np.abs(turn.sum() / path_length.sum()),
#            action_type = list(self._get_action_type(episode["cmd_vel"])),
            time_diff = time, ## Ros time in ns
            time = list(map(int, episode["time"].tolist())),
            episode = index,
            result = self._get_success(time, collision_amount),
#            cmd_vel = list(map(list, episode["cmd_vel"].to_list())),
            goal = goal_position,
            start = start_position
        )
    
    def _get_robot_params(self):
        with open(os.path.join(self.dir, "params.yaml")) as file:
            content = yaml.safe_load(file)

            model = content["model"]

        robot_model_params_file = os.path.join(
            rospkg.RosPack().get_path("arena_simulation_setup"),
            "entities",
            "robots", 
            model, 
            "model_params.yaml"
        )

        with open(robot_model_params_file, "r") as file:
            return yaml.safe_load(file)

    def _get_mean_position(self, episode, key):
        positions = episode[key].to_list()
        counter = {}

        for p in positions:
            hash = ":".join([str(pos) for pos in p])

            counter[hash] = counter.get(hash, 0) + 1

        sorted_positions = dict(sorted(counter.items(), key=lambda x: x))

        return [float(r) for r in list(sorted_positions.keys())[0].split(":")]

    def _get_position_for_collision(self, collisions, positions):
        for i, collision in enumerate(collisions):
            collisions[i][2] = positions[collision[0]]

        return collisions

    def _get_success(self, time, collisions):
        if time >= Config.TIMEOUT_TRESHOLD:
            return DoneReason.TIMEOUT

        if collisions >= Config.MAX_COLLISIONS:
            return DoneReason.COLLISION

        return DoneReason.GOAL_REACHED
    
    def _get_collisions(self, laser_scans, lower_bound):
        """
        Calculates the collisions. Therefore, 
        the laser scans is examinated and all values below a 
        specific range are marked as collision.

        Argument:
            - Array laser scans representing the scans over
            time
            - the lower bound for which a collisions are counted

        Returns tupel of:
            - Array of tuples with indexs and time in which
            a collision happened
        """
        collisions = []
        collisions_marker = []

        for i, scan in enumerate(laser_scans):

            is_collision = len(scan[scan <= lower_bound]) > 0

            collisions_marker.append(is_collision)
            
            if is_collision:
                collisions.append(i)

        collision_amount = 0

        for i, coll in enumerate(collisions_marker[1:]):
            prev_coll = collisions_marker[i]

            if coll - prev_coll > 0:
                collision_amount += 1

        return collisions, collision_amount

    def _get_action_type(self, actions):
        action_type = []

        for action in actions:
            if sum(action) == 0:
                action_type.append(Action.STOP.value)
            elif action[0] == 0 and action[1] == 0:
                action_type.append(Action.ROTATE.value)
            else:
                action_type.append(Action.MOVE.value)

        return action_type

    
        
class PedsimMetrics(Metrics):

    def _load_data(self) -> List[DataFrame]:
        pedsim_data = pd.read_csv(
            os.path.join(self.dir, "pedsim_agents_data.csv"),
            converters = {"data": Utils.parse_pedsim}
        ).rename(columns={"data": "peds"})
        
        return super()._load_data() + [pedsim_data]
    
    def __init__(self, dir: str, **kwargs):
        super().__init__(dir=dir, **kwargs)

    def _analyze_episode(self, episode: pd.DataFrame, index):

        max_peds = max(*(len(peds) for peds in episode['peds']), 0)
        episode = episode.iloc[[len(peds) == max_peds for peds in episode['peds']]].copy()

        super_analysis = super()._analyze_episode(episode, index)

        robot_position = np.array([odom["position"][:2] for odom in episode["odom"]])
        peds_position = np.array([[ped.position for ped in peds] for peds in episode["peds"]])

        # list of (timestamp, ped) indices, duplicate timestamps allowed
        personal_space_frames = np.linalg.norm(peds_position - robot_position[:,None], axis=-1) <= Config.PERSONAL_SPACE_RADIUS
        # list of timestamp indices, no duplicates
        is_personal_space = personal_space_frames.max(axis=1)

        # time in personal space
        time = np.diff(np.array(episode["time"]), prepend=0)
        total_time_in_personal_space = time[is_personal_space].sum()
        time_in_personal_space = [time[frames].sum(axis=0).astype(np.integer) for frames in personal_space_frames.T]

        # v_avg in personal space
        velocity = np.array(super_analysis["velocity"])
        velocity = velocity[is_personal_space]
        avg_velocity_in_personal_space = velocity.mean() if velocity.size else 0


        # gazes
        robot_direction = np.array([odom["position"][2] for odom in episode["odom"]])
        peds_direction = np.array([[ped.theta for ped in peds] for peds in episode["peds"]])
        angle_robot_peds = np.squeeze(np.angle(np.array(peds_position - robot_position[:,np.newaxis]).view(np.complex128)))

        # time looking at pedestrians
        robot_gaze = Math.angle_difference(robot_direction[:,np.newaxis], angle_robot_peds)
        looking_at_frames = np.abs(robot_gaze) <= Config.ROBOT_GAZE_ANGLE
        total_time_looking_at_pedestrians = time[looking_at_frames.max(axis=1)].sum()
        time_looking_at_pedestrians = [time[frames].sum(axis=0).astype(np.integer) for frames in looking_at_frames.T]
        
        # time being looked at by pedestrians
        ped_gaze = Math.angle_difference(peds_direction, np.pi - angle_robot_peds)
        looked_at_frames = np.abs(ped_gaze) <= Config.PEDESTRIAN_GAZE_ANGLE
        total_time_looked_at_by_pedestrians = time[looked_at_frames.max(axis=1)].sum()
        time_looked_at_by_pedestrians = [time[frames].sum(axis=0).astype(np.integer) for frames in looked_at_frames.T]

        return PedsimMetric(
            **super_analysis,
            avg_velocity_in_personal_space = avg_velocity_in_personal_space,
            total_time_in_personal_space = total_time_in_personal_space,
            time_in_personal_space = time_in_personal_space,
            total_time_looking_at_pedestrians = total_time_looking_at_pedestrians,
            time_looking_at_pedestrians = time_looking_at_pedestrians,
            total_time_looked_at_by_pedestrians = total_time_looked_at_by_pedestrians,
            time_looked_at_by_pedestrians = time_looked_at_by_pedestrians,
            num_pedestrians = peds_position.shape[1]
        )

class SubjectAwareMetrics(Metrics):
    def _load_data(self) -> List[DataFrame]:
        # Try to load subjects data file (if exists)
        subjects_data = pd.read_csv(
            os.path.join(self.dir, "pedsim_agents_data.csv"),
            converters = {"data": Utils.parse_pedsim}
        ).rename(columns={"data": "subjects"})
        
        return super()._load_data() + [subjects_data]
    
    def __init__(self, dir: str, **kwargs):
        # Get world and map information for configuration file path
        self.world_name = kwargs.get('world_name', 'arena_hospital_small')
        
        # Remove world_name from kwargs before passing to parent class
        parent_kwargs = {k: v for k, v in kwargs.items() if k != 'world_name'}
        
        # Load subjects configuration file
        self.subjects_config = self._load_subjects_config()
        
        super().__init__(dir=dir, **parent_kwargs)
    
    def _load_subjects_config(self):
        try:
            config_path = os.path.join(
                rospkg.RosPack().get_path("arena_simulation_setup"),
                "worlds",
                self.world_name,
                "metrics",
                "default.yaml"
            )
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            return config.get('subjects', {})
        except Exception as e:
            print(f"Warning: Could not load subjects metricsconfig: {e}")
            return {}
    
    def _analyze_episode(self, episode: pd.DataFrame, index) -> SubjectAwareMetric:
        # Get basic analysis results
        super_analysis = super()._analyze_episode(episode, index)
        
        # Initialize subjects related data
        overall_subject_score = 0.0
        subject_violation_count = 0
        actual_subject_count = 0  # Count of subjects that exist in both config and pedsim data
        total_actual_weight = 0.0  # Total weight of subjects that actually exist in this episode
        
        # Check if subjects data exists
        if 'subjects' in episode.columns:
            # Get robot positions for all timesteps
            robot_positions = np.array([odom["position"][:2] for odom in episode["odom"]])
            
            # Get subjects data for all timesteps (time-synchronized)
            subjects_data_timeline = episode['subjects'].tolist()
            
            # Analyze each subject
            for subject_id, subject_info in self.subjects_config.get('by_id', {}).items():
                subject_metric = self._analyze_subject_timeline(
                    subject_id, 
                    subject_info, 
                    subjects_data_timeline, 
                    robot_positions, 
                    episode
                )
                
                # print(f"Subject metric: {subject_metric}")
                if subject_metric:
                    # Get weight for this subject (default is 1.0)
                    weight = subject_info.get('average_weight', 1.0)
                    
                    # print("subject_metric['total_subject_score']: ", subject_metric['total_subject_score'])
                    overall_subject_score += subject_metric['total_subject_score'] * weight
                    total_actual_weight += weight
                    actual_subject_count += 1  # Count subjects that were successfully analyzed
                    
                    # Calculate violation count (based on distance threshold)
                    violation_count = self._count_subject_violations(
                        subject_metric['robot_subject_distances'],
                        subject_info
                    )
                    subject_violation_count += violation_count
        
        # If subjects exist, calculate weighted average score
        if self.subjects_config.get('scoring', {}).get('normalize_by_episode_time', False):
            episode_time = super_analysis['time_diff']
            if episode_time > 0:
                overall_subject_score /= episode_time
        
        # Calculate weighted average subject score
        average_subject_score = 0.0
        if actual_subject_count > 0 and total_actual_weight > 0:
            # Calculate dynamic weighted average: total_weighted_score / total_actual_weight * 100
            # This redistributes weights among actually present subjects
            average_subject_score = (overall_subject_score / total_actual_weight) * 100
        
        return SubjectAwareMetric(
            **super_analysis,
            overall_subject_score=overall_subject_score,
            average_subject_score=average_subject_score,
            subject_violation_count=subject_violation_count
        )
    
    def _analyze_subject_timeline(self, subject_id, subject_config, subjects_data_timeline, robot_positions, episode):
        """Analyze metrics for a single subject with proper time synchronization"""
        try:
            robot_subject_distances = []
            subject_positions_timeline = []
            subject_type = 'unknown'
            
            # Process each timestep
            for timestep_idx, subjects_data in enumerate(subjects_data_timeline):
                if not subjects_data or len(subjects_data) == 0:
                    # No subjects data at this timestep, skip
                    continue
                
                # Find corresponding subject data at this timestep
                subject_data = None
                for subj in subjects_data:
                    if hasattr(subj, 'id') and str(subj.id) == subject_id:
                        subject_data = subj
                        break
                
                if not subject_data:
                    # Subject not present at this timestep, skip
                    continue
                
                # Get subject position at this timestep
                subject_position = None
                if hasattr(subject_data, 'position'):
                    subject_position = subject_data.position
                elif hasattr(subject_data, 'positions'):
                    subject_position = subject_data.positions[0] if subject_data.positions else None
                
                if subject_position and len(subject_position) >= 2 and timestep_idx < len(robot_positions):
                    # Calculate distance between robot and subject at this timestep
                    robot_pos = robot_positions[timestep_idx]
                    distance = np.linalg.norm(robot_pos - np.array(subject_position[:2]))
                    robot_subject_distances.append(distance)
                    subject_positions_timeline.append(subject_position)
                    
                    # if subject_id == "Doctor_1":
                    #     print("robot_pos: ", robot_pos)
                    #     print("subject_position: ", subject_position)
                    #     print("distance: ", distance)
                    
                    # Store subject type (should be consistent across timesteps)
                    if subject_type == 'unknown':
                        subject_type = getattr(subject_data, 'type', 'unknown')
            
            if not robot_subject_distances:
                # print(f"No valid distance data found for subject {subject_id}")
                return None
            
            # Calculate subject score
            scoring_type = subject_config.get('scoring', 'distance_penalty')
            scoring_params = subject_config.get('params', {})
            
            subject_scores = self._calculate_subject_scores(
                scoring_type, 
                scoring_params, 
                robot_subject_distances
            )
            
            total_subject_score = np.mean(subject_scores) if subject_scores else 0.0
            
            return SubjectMetric(
                subject_id=subject_id,
                subject_type=subject_type,
                subject_position=subject_positions_timeline,
                robot_subject_distances=robot_subject_distances,
                subject_scores=subject_scores,
                total_subject_score=total_subject_score,
                scoring_function_type=scoring_type
            )
            
        except Exception as e:
            print(f"Error analyzing subject {subject_id}: {e}")
            return None
    
    def _analyze_subject(self, subject_id, subject_config, subjects_data, robot_positions, episode):
        """Legacy method - kept for backward compatibility"""
        try:
            # Find corresponding subject data
            subject_data = None
            for subj in subjects_data:
                if hasattr(subj, 'id') and str(subj.id) == subject_id:
                    subject_data = subj
                    break
            
            if not subject_data:
                return None
            
            # Get subject position (if exists)
            # print("robot_positions: ", robot_positions)
            # print("subject_data: ", subject_data)
            subject_positions = []
            if hasattr(subject_data, 'position'):
                subject_positions = [subject_data.position]
            elif hasattr(subject_data, 'positions'):
                subject_positions = subject_data.positions
            
            # Calculate distance from robot to subject
            robot_subject_distances = []
            if subject_positions and len(robot_positions) > 0:
                for subj_pos in subject_positions:
                    if len(subj_pos) >= 2:
                        distances = np.linalg.norm(robot_positions - np.array(subj_pos[:2]), axis=1)
                        robot_subject_distances.extend(distances.tolist())
            
            # if subject_id == "Doctor_1":
            #     print("robot_positions: ", robot_positions)
            #     print("subject_positions: ", subject_positions)
            #     print("robot_subject_distances: ", robot_subject_distances)
            
            # Calculate subject score
            scoring_type = subject_config.get('scoring', 'distance_penalty')
            scoring_params = subject_config.get('params', {})
            
            subject_scores = self._calculate_subject_scores(
                scoring_type, 
                scoring_params, 
                robot_subject_distances
            )
            
            total_subject_score = np.mean(subject_scores) if subject_scores else 0.0
            
            return SubjectMetric(
                subject_id=subject_id,
                subject_type=getattr(subject_data, 'type', 'unknown'),
                subject_position=subject_positions,
                robot_subject_distances=robot_subject_distances,
                subject_scores=subject_scores,
                total_subject_score=total_subject_score,
                scoring_function_type=scoring_type
            )
            
        except Exception as e:
            print(f"Error analyzing subject {subject_id}: {e}")
            return None
    
    def _calculate_subject_scores(self, scoring_type, params, distances):
        """Calculate subject scores based on different scoring functions"""
        if not distances:
            return []
        
        distances = np.array(distances)
        
        if scoring_type == 'u_curve':
            optimal_distance = params.get('optimal_distance', 1.0)
            curve_width = params.get('curve_width', 0.5)
            
            # Use the proper U-curve scoring function from Math class
            scores = Math.u_curve_score(distances, optimal_distance, curve_width)
            
        elif scoring_type == 'distance_penalty':
            safe_distance = params.get('safe_distance', 1.0)
            penalty_weight = params.get('penalty_weight', 1.0)
            
            # Use the proper distance penalty scoring function from Math class
            scores = Math.distance_penalty_score(distances, safe_distance, penalty_weight)

        
        return scores.tolist()
    
    def _count_subject_violations(self, distances, subject_config):
        """Calculate subject violation count (based on continuous violation state changes)"""
        if not distances:
            return 0
        
        distances = np.array(distances)
        scoring_type = subject_config.get('scoring', 'distance_penalty')
        params = subject_config.get('params', {})
        tolerance = params.get('tolerance', 0.5)
        
        # Calculate violation state for each time step
        if scoring_type == 'u_curve':
            # U-curve: both too far and too close distances count as violations
            optimal_distance = params.get('optimal_distance', 1.0)
            violations = np.logical_or(
                distances < (optimal_distance - tolerance),
                distances > (optimal_distance + tolerance)
            )
        elif scoring_type == 'distance_penalty':
            # Distance penalty: too close distance counts as violation
            safe_distance = params.get('safe_distance', 1.0)
            violations = distances < (safe_distance + tolerance)
        else:
            violations = np.zeros_like(distances, dtype=bool)
        
        # Calculate continuous violation state change count
        violation_count = 0
        was_in_violation = False
        
        for is_violation in violations:
            if is_violation and not was_in_violation:
                # New violation starts
                violation_count += 1
                was_in_violation = True
            elif not is_violation:
                # Violation ends
                was_in_violation = False
        
        return violation_count

class ZoneAwareMetrics(Metrics):
    def _load_data(self) -> List[DataFrame]:
        # Load basic data
        return super()._load_data()
    
    def __init__(self, dir: str, **kwargs):
        # Get world and map information for configuration file path
        self.world_name = kwargs.get('world_name', 'small_warehouse')
        
        # Remove world_name from kwargs before passing to parent class
        parent_kwargs = {k: v for k, v in kwargs.items() if k != 'world_name'}
        
        # Load zones configuration file
        self.zones_config = self._load_zones_config()
        # print(f"Loaded zones config for world '{self.world_name}': {self.zones_config}")
        super().__init__(dir=dir, **parent_kwargs)

    def _load_zones_config(self):
        """Load zones configuration file and specific zone definitions"""
        zones_config = {}
        
        try:
            # Load metrics configuration file
            metrics_config_path = os.path.join(
                rospkg.RosPack().get_path("arena_simulation_setup"),
                "worlds",
                self.world_name,
                "metrics",
                "default.yaml"
            )
            
            with open(metrics_config_path, 'r') as f:
                metrics_config = yaml.safe_load(f)
            
            zones_config = metrics_config.get('zones', {})
            
            # Load specific zone definition file
            zones_file_path = os.path.join(
                rospkg.RosPack().get_path("arena_simulation_setup"),
                "worlds",
                self.world_name,
                "map",
                "zones.yaml"
            )
            
            with open(zones_file_path, 'r') as f:
                zones_data = yaml.safe_load(f)
            
            # Merge configuration: add specific polygon data to configuration
            # Only process zones that are explicitly defined in default.yaml
            for zone_info in zones_data:
                zone_label = zone_info.get('label', 'Unknown Zone')
                if zone_label in zones_config:
                    # Add polygon data to configuration
                    zones_config[zone_label]['polygons'] = zone_info.get('polygon', [])
                    zones_config[zone_label]['category'] = zone_info.get('category', [])
                # Remove the else clause - we only want zones explicitly defined in default.yaml
            
            # print(f"Loaded zones config for world '{self.world_name}': {list(zones_config.keys())}")
            
        except Exception as e:
            print(f"Warning: Could not load zones metrics config: {e}")
            zones_config = {}
        
        return zones_config
    
    def _analyze_episode(self, episode: pd.DataFrame, index) -> ZoneAwareMetric:
        # Get basic analysis results
        super_analysis = super()._analyze_episode(episode, index)
        
        # Get robot position data
        robot_positions = np.array([odom["position"][:2] for odom in episode["odom"]])
        
        # Analyze zone violations
        zone_violations = []
        
        # Check each zone type
        for zone_label, zone_config in self.zones_config.items():
            zone_violations.extend(
                self._check_zone_violations(zone_label, zone_config, robot_positions, episode)
            )
        
        # print("Zone Violations: ")
        # print(zone_violations)
        
        # Calculate total violation count and total violation time
        zone_violation_count = len(zone_violations)
        zone_violation_time = sum(violation['duration'] for violation in zone_violations)
        
        # Calculate overall zone score (based on violation severity)
        overall_zone_score = 100.0
        
        if zone_violations:
            # Use negative value of violation severity as score (more severe violations result in lower scores)
            total_severity = sum(violation['severity'] for violation in zone_violations)
            # print(total_severity, len(zone_violations))

            # Convert severity to 0-100 score range
            overall_zone_score = 100.0 + total_severity / len(zone_violations)
            # print(overall_zone_score)
        
        return ZoneAwareMetric(
            **super_analysis,
            overall_zone_score=overall_zone_score,
            zone_violation_count=zone_violation_count,
            zone_violation_time=zone_violation_time
        )
    
    def _check_zone_violations(self, zone_type, zone_config, robot_positions, episode):
        """Check violations for specific zone type"""
        violations = []
        
        # Get zone polygon coordinates
        polygons = zone_config.get('polygons', [])
        
        if not polygons:
            return violations
        
        # Track violation state changes
        was_in_violation = False
        violation_start_time = None
        violation_start_index = None
        
        # Check robot position at each time step
        for i, robot_pos in enumerate(robot_positions):
            # Check if robot is inside any polygon
            is_in_zone = False
            for polygon_points in polygons:
                if len(polygon_points) >= 3:  # Ensure polygon has at least 3 points
                    polygon = np.array(polygon_points)
                    if self._is_point_in_polygon(robot_pos, polygon):
                        is_in_zone = True
                        break
            
            # Detect violation state changes
            if is_in_zone and not was_in_violation:
                # New violation starts
                violation_start_time = int(episode["time"].iloc[i]) if i < len(episode["time"]) else 0
                violation_start_index = i
                was_in_violation = True
                
            elif not is_in_zone and was_in_violation:
                # Violation ends, record violation
                if violation_start_index is not None:
                    # Calculate violation duration
                    duration = i - violation_start_index
                    
                    # Calculate violation severity (using start position and duration)
                    start_robot_pos = robot_positions[violation_start_index]
                    severity = self._calculate_violation_severity(
                        zone_type, zone_config, start_robot_pos, violation_start_index, duration
                    )
                    # print("Severity: ", severity)
                    
                    violation = ZoneViolation(
                        timestamp=violation_start_time,
                        position=start_robot_pos.tolist(),
                        duration=duration,
                        severity=severity,
                        violation_type=zone_config.get('penalty_type', 'zone_violation')
                    )
                    violations.append(violation)
                
                was_in_violation = False
                violation_start_time = None
                violation_start_index = None
        
        # Handle case where violation continues at episode end
        if was_in_violation and violation_start_index is not None:
            duration = len(robot_positions) - violation_start_index
            start_robot_pos = robot_positions[violation_start_index]
            severity = self._calculate_violation_severity(
                zone_type, zone_config, start_robot_pos, violation_start_index, duration
            )
            
            violation = ZoneViolation(
                timestamp=violation_start_time,
                position=start_robot_pos.tolist(),
                duration=duration,
                severity=severity,
                violation_type=zone_config.get('penalty_type', 'zone_violation')
            )
            violations.append(violation)
        
        return violations
    
    def _is_point_in_polygon(self, point, polygon):
        """Use ray casting method to determine if a point is inside a polygon"""
        if not SHAPELY_AVAILABLE:
            return Math.point_in_polygon(point, polygon)

        polygon_shapely = Polygon(polygon)
        point_shapely = Point(point)
        return polygon_shapely.contains(point_shapely)
    
    def _calculate_violation_severity(self, zone_type, zone_config, robot_pos, time_index, duration):
        params = zone_config.get('params', {})
        base_penalty = params.get('violation_penalty', 0.0)
        penalty_weight = params.get('penalty_weight', 1.0)
        
        # Base penalty
        # print("Base penalty: ", base_penalty)
        # print("Penalty weight: ", penalty_weight)
        severity = base_penalty * penalty_weight
        
        # If continuous violation accumulation penalty is supported
        if params.get('continuous_violation', False):
            # Time-based penalty: severity increases with duration
            # You can adjust the time multiplier as needed
            time_multiplier = params.get('time_multiplier', 0.1)  # Penalty per time step
            time_penalty = duration * time_multiplier
            severity = severity + (base_penalty * time_penalty)
        
        # Apply maximum penalty limit (prevent penalty from being too severe)
        max_penalty = params.get('max_penalty', float('-inf'))
        if max_penalty > float('-inf'):
            # For negative values, we want severity to be >= max_penalty (less negative)
            severity = max(severity, max_penalty)
        
        return severity
    
    def _group_violations_by_zone(self, violations):
        """Group violations by zone_id"""
        zone_metrics = {}
        
        for violation in violations:
            # Extract zone information from violation_type
            zone_id = violation['violation_type']
            
            if zone_id not in zone_metrics:
                zone_metrics[zone_id] = {
                    'zone_id': zone_id,
                    'total_violations': 0,
                    'total_severity': 0.0,
                    'violation_count': 0,
                    'max_severity': float('-inf'),
                    'min_severity': float('inf')
                }
            
            zone_metrics[zone_id]['total_violations'] += 1
            zone_metrics[zone_id]['total_severity'] += violation['severity']
            zone_metrics[zone_id]['max_severity'] = max(zone_metrics[zone_id]['max_severity'], violation['severity'])
            zone_metrics[zone_id]['min_severity'] = min(zone_metrics[zone_id]['min_severity'], violation['severity'])
        
        # Calculate average values
        for zone_id in zone_metrics:
            if zone_metrics[zone_id]['total_violations'] > 0:
                zone_metrics[zone_id]['avg_severity'] = (
                    zone_metrics[zone_id]['total_severity'] / zone_metrics[zone_id]['total_violations']
                )
            else:
                zone_metrics[zone_id]['avg_severity'] = 0.0
        
        return list(zone_metrics.values())
