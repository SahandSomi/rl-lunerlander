# rl-lunerlander
# Introduction 
Reinforcement learning (RL) planning in construction involves training agents to make decisions about construction tasks, scheduling, resource allocation, and risk management to improve project efficiency and reduce costs.

# Getting Started
1.	Clone repo and then run this command in your terminal to install all dependencies
`conda env create --name RL_Luner --file=environment.yml` 
2.	Open terminal and run `streamlit run app.py` in project root.
3.	Streamlit will create a port and open that network URL to see the app. If are connecting to a remote server, you need to forward the port to open it locally.

# Prerequisites
  - Python 3.6 or higher
  - pytorch------------> Deep learning general library.
  - torchvision--------> Deep learning for computer vision.
  - torchaudio---------> Deep learning for audio and signal.

# Data

# In Details
Folder Structure and files in detail
```
├──  data
│
├──  models 
|
|
├── data converter                          - converting .sim file to excel 
│
|
| 
├──  notebooks   
│    
│
├── src                                     - source code for ML model.
│ 
│
├── app.py                                  - app for demoing RL model usage with streamlit.
│ 
│
├── environment.yml                         - essential libraries to create env which use conda channels for installation.
│   
│ 
├──  requirements.txt                       - third-party libraries to install with pip.
```

## Plans

## Challenges