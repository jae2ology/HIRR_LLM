import time
import numpy as np
from stretch_mujoco import StretchMujocoSimulator, Actuators, StretchSensors


class StretchProactiveExecutor:

    def __init__(self, headless=False):
        self.sim = StretchMujocoSimulator()
        self.headless = headless

        # pre-defined spatial locations (x, y, z, lift_m, arm_m) for objects
        self.known_object_poses = {
            "bathroom_cabinet": {"lift": 0.9, "arm": 0.1, "base_v": 0.0},
            "toothbrush_holder": {"lift": 0.7, "arm": 0.3, "base_v": 0.1},
            "tooth_paste": {"lift": 0.65, "arm": 0.4, "base_v": 0.0},
            "notebook": {"lift": 0.5, "arm": 0.35, "base_v": 0.2},
            "cup": {"lift": 0.6, "arm": 0.2, "base_v": 0.0}
        }

    def start(self):
        self.sim.start(headless=self.headless)
        print("Initializing robot pose...")
        self.sim.home()
        time.sleep(1.0)

    def stop(self):
        self.sim.stop()

    def execute(self, predicted_object: str) -> dict:
        print(f"\nExecuting assistance for: '{predicted_object}'")
        start_time = time.time()

        if predicted_object not in self.known_object_poses:
            print(f"Warning: Target '{predicted_object}' unknown. Executing default search pose.")
            target_config = {"lift": 0.6, "arm": 0.1, "base_v": 0.0}
        else:
            target_config = self.known_object_poses[predicted_object]

        # 1. Point camera / Head towards object
        self.sim.move_to('head_pan', -0.4)

        # 2. Adjust Base Position if required
        if target_config["base_v"] > 0:
            self.sim.set_base_velocity(v_linear=target_config["base_v"], v_angular=0.0)
            time.sleep(1.0)
            self.sim.set_base_velocity(0.0, 0.0)

        # 3. Extend Arm and Lift to target object pose
        self.sim.move_to('lift', target_config["lift"])
        self.sim.move_to('arm', target_config["arm"])

        # Wait until physical trajectories settle
        self.sim.wait_while_is_moving('lift', timeout=3.0)
        self.sim.wait_while_is_moving('arm', timeout=3.0)

        # 4. Actuate End-Effector / Gripper to simulate object handoff/pickup
        self.sim.move_to('stretch_gripper', 50)  # Open/close gripper
        time.sleep(0.5)

        exec_duration = time.time() - start_time

        # 5. Read back status for evaluation
        status = self.sim.pull_status()
        actual_lift = status.get_joint_position(Actuators.lift)
        actual_arm = status.get_joint_position(Actuators.arm)

        execution_summary = {
            "target": predicted_object,
            "actual_lift": actual_lift,
            "actual_arm": actual_arm,
            "execution_time_s": exec_duration,
            "success": True
        }

        return execution_summary