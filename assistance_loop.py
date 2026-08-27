import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, List, Dict
from pydantic import BaseModel
from markov_model import SequenceModel

load_dotenv()
open_ai_key = os.getenv("OPEN_AI_API_KEY")

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

    def can_assist_activity(self, action: UserAction) -> bool:
        # check if the human intent/activity is recognized
        return action.activity in self.plans

    def can_assist_objects(self, action: UserAction) -> bool:
        # check if the human intent/activity is recognized
        return action.completed is False # TODO: check what actually determines whether an activity has been completed or not

    def evaluate_next_step(self, action: UserAction) -> Optional[str]:
        sequence = self.plans.get(action.activity, action.objects_in_activity) # get the activity, and all the steps it has currently

        # call the n gram model and train on predefined plans (dataset)
        model = SequenceModel(threshold=0.5)
        model.train(self.plans)

        # assume that the recognized activity IS in the dataset
        current_activity = action.activity
        current_objects = action.objects_in_activity

        pred_obj, confidence, top_candidates, trigger_llm = model.predict_next(current_activity, current_objects)

        # if trigger LLM is false, then the model has high confidence for the next object being pred_obj
        if not trigger_llm:
            print(f"Next object needed in sequence: {pred_obj} with confidence: {confidence}")

        else:
            print(f"Model has low confidence. Top candidates: {top_candidates}. Triggering LLM response")
            llm = LLMFallback(api_key=open_ai_key)
            response = llm.infer_goal_create_task(action=action, predicted_objects=top_candidates)
            print(f"Model will pick up {response}. Is this correct?")

        #TODO: change return type to string object when testing

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