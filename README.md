# SMART-PRO - Real-Time robotic assistance using object prediction

Proactive robot assistance refers to a robot’s ability to assist a user
without requiring explicit instruction. In household environments,
this means anticipating a user’s needs during household tasks rather
than waiting to be asked. However, existing approaches rely on
longitudinal routine tracking, which requires observing daily habits
over days or weeks before providing assistance. In this work, we
present SMART-PRO (Sequence Modeling via HumAn Intention
Recognition and AdapTive LLM Reasoning for PRedicting Object
usage), a novel framework designed for immediate, real-time object
prediction in open-ended household tasks. Rather than waiting
to observe long-term temporal routines, SMART-PRO operates
on an event-driven system. The framework combines real-time
visual perception (YOLOv8) with activity prediction using a Naïve
Bayes model and an activity-conditioned 𝑛-gram sequence model
(Markov model) to predict the next object required in a task. When
prediction confidence drops below a threshold of 0.5, an adaptive
Large Language Model (LLM) fallback module is queried for semantic reasoning.

## How to start?
Run the simulation by going to main_test.py. This will open MujoCo and the Stretch 3 Mobile Manipulator in the virtual environment.

## How can I train SMART-PRO on other datasets?
The datasets must be in JSON TUPLES of activity : objects in sequence. Put your data in the data directory, separate by train and test. Go to the evaluation directory and change the train_path and test_path to your new dataset, then run the file. You will find the new F1 score and confusion matrices using your dataset.

## Can I use this without the simulated environment?
Yes! The SMART-PRO base framework is located in the simulation directory under the file main_loop.py. If you want to implement this framework into other applications, be sure to take all the models in the models directory, too!
