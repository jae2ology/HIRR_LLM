import time
from openai import OpenAI
from typing import Optional, List, Dict
from pydantic import BaseModel


# pydantic classes for user actions and known plan
class UserAction(BaseModel):
    activity: str # like, "cooking", "eating", "reading", etc
    objects_in_activity: List[str] # like ["bowl", "fork"]
    completed: bool = False # to determine if the action can be assisted with or not

class Prediction(BaseModel):
    known_sequence: List[str] # like ["fetch_bowl", "fetch_fork", "fetch_cup"]
    curr_index: int = 0 # current step in sequence


# logic
class HIRRBase:
    """this class will handle known routines and predictions"""
    def __init__(self, predefined_plans: Dict[str, List[str]]):
        self.plans = predefined_plans
        # e.g, predefined_plans = "eating" : ["fetch_bowl", "fetch_spoon", "fetch_cereal", "fetch_milk"]

    def can_assist(self, action: UserAction) -> bool:
        # check if the human intent/activity is recognized and the task is still going
        return action.activity in self.plans & action.completed is False

    def evaluate_next_step(self, action: UserAction, curr_step_index: int) -> Optional[str]:
        sequence = self.plans.get(action.activity, []) # get the activity, and all the steps needed for it

        if curr_step_index < len(sequence):
            return sequence[curr_step_index + 1] # next step is predictable

        return None # next step is not predictable, must go to LLM system
class LLMFallback:
    """this class will handle unpredictable intent/actions from a human user"""
    def __init__(self, api_key: str):
        self.client = OpenAI()

    def infer_goal_create_task(self, action: UserAction, predicted_objects) -> str:
        prompt = """
        You are the brain of an assistance robot. 
        The human user is currently doing this activity: {activity}.
        The user has picked up these objects in order: {objects} 
        The n-gram sequence model has predicted these potential objects: {predicted}
        
        Choosing only from the predicted object list, what object is needed next?
        Do not explain your answer, simply give the response.
        """

        response = self.client.responses.create(
            model="gpt-5.6",
            input= [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt
                        },
                        {
                            "type": "input_text",
                            "text": action.activity
                        },
                        {
                            "type": "input_text",
                            "text": action.objects_in_activity
                        },
                        {
                            "type": "input_text",
                            "text": predicted_objects
                        },
                    ],
                }
            ]
        )

        return response.output_text



# simulated robot
class RobotExecution:
    """simulates fetching"""
    def fetch_object(self, object_name: str) -> bool:
        print(f"Fetching target object -> '{object_name}'")
        return True

    def sit_and_observe(self):
        print("Waiting")

    def clarify(self):
        print("I am currently unsure of what action you are performing. What would you like me to do?")