# Real-Time Proactive Robot Assistance via HIRR Sequence Models and Adaptive LLM Reasoning

## Jae Jackson & Hayden Wimmer

### 🎯 Abstract

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
Large Language Model (LLM) fallback module is queried for semantic reasoning. SMART-PRO performance without the LLM
queries raises the F1-score metric to 0.82, and 0.77 with queries,
over a score of 0.73 from previous work. SMART-PRO is further validated through tasks within the NVIDIA Isaac Sim environment integrated with ROS 2. 
