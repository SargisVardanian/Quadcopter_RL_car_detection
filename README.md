Autonomous Drone Control System

This repository contains the implementation of an autonomous drone control system using reinforcement learning algorithms. The project is designed to develop a reliable and efficient mechanism for tracking a moving vehicle with a drone, leveraging state-of-the-art machine learning techniques.

Files and Descriptions

Python Files

	1.	Result.py
	•	The main entry point for the reinforcement learning training process. This script sets up the environment, initializes the learning agent, and orchestrates the training loop. It also handles logging and saving of the trained models.
	2.	reward.py
	•	This script defines the reward function for the reinforcement learning algorithm. The reward function is crucial for guiding the learning process, as it provides feedback to the agent based on its actions and the resulting states.
	3.	Actor_Critic.py
	•	Implementation of the Actor-Critic algorithm for reinforcement learning. This script contains the definitions of the actor and critic networks, as well as the training procedures for optimizing these networks based on the observed rewards and states.
	4.	C#Script_tester.py
	•	A Python script designed to test the integration between Unity (where the drone simulation runs) and the Python-based reinforcement learning algorithms. This script ensures that the data exchange and command execution between the two environments are functioning correctly.

C# Files

	1.	Mover.cs
	•	A script for controlling basic movements of the drone within the Unity environment. This script handles the low-level commands for translating and rotating the drone based on input parameters.
	2.	DroneControlC.cs
	•	The main control script for the drone in Unity. This script integrates with the reinforcement learning model to receive commands and update the drone’s state accordingly. It handles sensor data, communicates with Python scripts, and ensures the drone follows the optimal path.
	3.	ResetPosition.cs
	•	A utility script to reset the drone’s position in the simulation environment. This script is used to ensure the drone starts from a predefined position after each episode or in the event of a collision or other failure conditions.

Getting Started

To get started with this project, follow these steps:

	1.	Clone the repository to your local machine.
	2.	Set up your Python environment with the required dependencies listed in requirements.txt.
	3.	Open the Unity project and ensure all C# scripts are correctly attached to the appropriate GameObjects.
	4.	Run main.py to start the training process.
	5.	Use the Unity Editor to visualize the drone’s behavior and debug any issues.
