import os

import cohere
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, List, Dict
from pydantic import BaseModel
from models.markov_model import SequenceModel
import json

load_dotenv()
open_ai_key = os.getenv("OPEN_AI_API_KEY")

# pydantic classes for user actions and known plan
class UserAction(BaseModel):
    activity: str # like, "cooking", "eating", "reading", etc
    objects_in_activity: List[str] # like ["bowl", "fork"]
    completed: bool = False # to determine if the action can be assisted with or not


# logic
class HIRRBase:
    """this class will handle known routines and predictions"""
    def __init__(self, path_to_train:str):
        self.path_to_train = path_to_train

        with open(path_to_train, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.plans = data
        # e.g, predefined_plans = "eating" : ["fetch_bowl", "fetch_spoon", "fetch_cereal", "fetch_milk"]

    def can_assist_activity(self, action: UserAction) -> bool:
        # check if the human intent/activity is recognized
        return action.activity in self.plans

    def can_assist_objects(self, action: UserAction) -> bool:
        # check if the human intent/activity is recognized
        return action.completed is False

    def evaluate_next_step(self, action: UserAction) -> Optional[str]:
        # call the n gram model and train on predefined plans (dataset)
        model = SequenceModel(threshold=0.5)
        model.train(self.path_to_train)

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
            response = str(response).lower().strip()
            print(f"Model will pick up {response}.")
            pred_obj = response

        return pred_obj


class LLMFallback:
    """handles unpredictable intent/actions from a user using OpenAI SDK"""
    def __init__(self, api_key: str):
        self.client = cohere.ClientV2(api_key=api_key)

    def infer_goal_create_task(self, action: UserAction, predicted_objects: list) -> str:
        # Extract candidate names if predicted_objects contains (object, probability) tuples
        candidate_names = [
            obj[0] if isinstance(obj, (tuple, list)) else obj
            for obj in predicted_objects
        ]

        objects_str = ", ".join(action.objects_in_activity) if action.objects_in_activity else "None"
        candidates_str = ", ".join(candidate_names) if candidate_names else "None"

        prompt = f"""You are the brain of an assistance robot.
        The human user is currently doing this activity: {action.activity}.
        The user has picked up these objects in order: {objects_str}
        The n-gram sequence model has predicted these potential candidate objects: {candidates_str}

        Choosing ONLY from the predicted object list, what object is needed next?
        Do not explain your answer. Do not give an explanation. Do not walk through your answer. Only give the answer.
        """

        response = self.client.chat(
            model="command-a-plus-05-2026",
            messages=[
                {"role": "user",
                 "content": prompt}
            ],
        )

        for content_item in response.message.content:
            if content_item.type == "text":
                final_res = content_item

        if final_res and hasattr(final_res, 'text'):
            return final_res.text.strip().lower()

        return ""
