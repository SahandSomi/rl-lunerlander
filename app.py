import streamlit as st
import torch
import numpy as np
import gymnasium as gym # Using gymnasium as per modern OpenAI Gym standards
import os
from collections import deque
import time

# Adjust path to import from src
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Agent.dqn import DQN
from src.Model.main import QNetwork
from src.Memory.ReplayBuffer import Experience as ReplayBuffer # Renamed class for clarity
from src.Agent.config import DQN_soft_update as DQNConfig

# --- Checkpoint Directory ---
CHECKPOINT_DIR = "agent_checkpoint"
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# --- Session State Initialization ---
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'environment' not in st.session_state:
    st.session_state.environment = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'replay_buffer' not in st.session_state:
    st.session_state.replay_buffer = None
if 'q_network_local' not in st.session_state:
    st.session_state.q_network_local = None
if 'q_network_target' not in st.session_state:
    st.session_state.q_network_target = None
if 'selected_env' not in st.session_state:
    st.session_state.selected_env = "LunarLander-v2"
if 'training_started' not in st.session_state:
    st.session_state.training_started = False
if 'episode_rewards' not in st.session_state:
    st.session_state.episode_rewards = []
if 'current_episode_reward' not in st.session_state:
    st.session_state.current_episode_reward = 0.0
if 'epsilon_values' not in st.session_state:
    st.session_state.epsilon_values = []
if 'current_epsilon' not in st.session_state:
    st.session_state.current_epsilon = 1.0
if 'total_steps' not in st.session_state:
    st.session_state.total_steps = 0
if 'current_episode_steps' not in st.session_state:
    st.session_state.current_episode_steps = 0
if 'max_episodes' not in st.session_state:
    st.session_state.max_episodes = 2000
if 'max_steps_per_episode' not in st.session_state:
    st.session_state.max_steps_per_episode = 1000
if 'epsilon_start' not in st.session_state:
    st.session_state.epsilon_start = 1.0
if 'epsilon_end' not in st.session_state:
    st.session_state.epsilon_end = 0.01
if 'epsilon_decay' not in st.session_state:
    st.session_state.epsilon_decay = 0.995
if 'current_game_state' not in st.session_state: # Added as per item 5
    st.session_state.current_game_state = None
if 'training_in_progress' not in st.session_state: # Added for disabling controls
    st.session_state.training_in_progress = False
if 'current_episode_number' not in st.session_state:
    st.session_state.current_episode_number = 0
if 'num_episodes_to_run' not in st.session_state:
    st.session_state.num_episodes_to_run = 1

# --- Environment Options & Device ---
AVAILABLE_ENVS = {
    "LunarLander-v2": {"state_size": 8, "action_size": 4},
    "CartPole-v1": {"state_size": 4, "action_size": 2}
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helper Function ---
def initialize_or_reset_components(env_name):
    with st.spinner(f"Initializing environment {env_name} and agent..."):
        state_size = AVAILABLE_ENVS[env_name]["state_size"]
        action_size = AVAILABLE_ENVS[env_name]["action_size"]

        st.session_state.config = DQNConfig()
    st.session_state.replay_buffer = ReplayBuffer(action_size, st.session_state.config.BUFFER_SIZE, st.session_state.config.BATCH_SIZE, DEVICE, st.session_state.config.SEED)
    st.session_state.q_network_local = QNetwork(state_size, action_size, st.session_state.config.SEED).to(DEVICE)
    st.session_state.q_network_target = QNetwork(state_size, action_size, st.session_state.config.SEED).to(DEVICE)
    st.session_state.agent = DQN(st.session_state.q_network_local, state_size, action_size, st.session_state.replay_buffer, DEVICE, st.session_state.config)

    st.session_state.q_network_target.load_state_dict(st.session_state.q_network_local.state_dict())

    if st.session_state.environment is not None:
        st.session_state.environment.close()

    st.session_state.environment = gym.make(env_name, render_mode='rgb_array')

    # Reset training stats
    st.session_state.episode_rewards = []
    st.session_state.current_episode_reward = 0.0
    st.session_state.epsilon_values = []
    st.session_state.current_epsilon = st.session_state.epsilon_start
    st.session_state.total_steps = 0
    st.session_state.current_episode_steps = 0
    st.session_state.current_episode_number = 0 # Reset here

    st.session_state.training_started = True
    st.session_state.training_in_progress = False # Reset this flag
        initial_state, _ = st.session_state.environment.reset(seed=st.session_state.config.SEED)
        st.session_state.current_game_state = initial_state
    return initial_state

def initialize_or_reset_components_callback():
    initialize_or_reset_components(st.session_state.selected_env)
    st.session_state.training_in_progress = False # Explicitly set training_in_progress to False
    st.toast(f"Environment {st.session_state.selected_env} reset and agent initialized.", icon="🔄")
    update_ui_elements(None) # Update UI after reset
    st.rerun() # Rerun to reflect changes immediately, like button states

# --- Training Step Function ---
def run_training_step():
    if st.session_state.current_game_state is None or st.session_state.agent is None or st.session_state.environment is None:
        # Not initialized properly or called out of sequence
        return None, 0.0, True, {} # frame, reward, done, info

    # Act
    action = st.session_state.agent.act(st.session_state.current_game_state, st.session_state.current_epsilon)

    # Step Environment
    next_state, reward, terminated, truncated, info = st.session_state.environment.step(action)
    done = terminated or truncated

    # Agent Step (Learn)
    # Ensure agent is not None before calling step, though covered by the initial check
    if st.session_state.agent:
        st.session_state.agent.step(st.session_state.current_game_state, action, reward, next_state, done)

    # Update Session State
    st.session_state.current_game_state = next_state
    st.session_state.current_episode_reward += reward
    st.session_state.total_steps += 1
    st.session_state.current_episode_steps += 1

    # Render
    frame = st.session_state.environment.render()

    # Handle Episode End
    if done:
        st.session_state.episode_rewards.append(st.session_state.current_episode_reward)
        st.session_state.epsilon_values.append(st.session_state.current_epsilon) # Log epsilon at end of episode

        # Decay epsilon
        new_epsilon = max(st.session_state.epsilon_end, st.session_state.current_epsilon * st.session_state.epsilon_decay)
        st.session_state.current_epsilon = new_epsilon

        # Reset env and current episode stats
        # Consider using a different seed for subsequent episode resets if desired, for now using original config seed
        st.session_state.current_game_state, _ = st.session_state.environment.reset(seed=st.session_state.config.SEED if st.session_state.config else None)
        st.session_state.current_episode_reward = 0.0
        st.session_state.current_episode_steps = 0
        st.session_state.current_episode_number += 1 # Increment here

        # Check if max_episodes is reached
        if st.session_state.current_episode_number > st.session_state.max_episodes: # Adjusted condition
            st.session_state.training_in_progress = False # Stop training
            st.info(f"Training completed after {st.session_state.max_episodes} episodes.")
            # Potentially call a function to save model or final logs here

    elif st.session_state.current_episode_steps >= st.session_state.max_steps_per_episode:
        # Handle episode truncation due to max_steps_per_episode
        st.session_state.episode_rewards.append(st.session_state.current_episode_reward)
        st.session_state.epsilon_values.append(st.session_state.current_epsilon)

        # Decay epsilon (optional: or only decay on "natural" termination)
        new_epsilon = max(st.session_state.epsilon_end, st.session_state.current_epsilon * st.session_state.epsilon_decay)
        st.session_state.current_epsilon = new_epsilon

        st.session_state.current_game_state, _ = st.session_state.environment.reset(seed=st.session_state.config.SEED if st.session_state.config else None)
        st.session_state.current_episode_reward = 0.0
        st.session_state.current_episode_steps = 0
        st.session_state.current_episode_number += 1 # Increment here as well for truncated episodes
        done = True # Treat as done for the purpose of resetting episode stats

        if st.session_state.current_episode_number > st.session_state.max_episodes: # Adjusted condition
             st.session_state.training_in_progress = False # Stop training
             st.info(f"Training completed after {st.session_state.max_episodes} episodes.")


    return frame, reward, done, info

# --- Training Control Callbacks ---
def callback_train_one_episode():
    st.session_state.training_in_progress = False # Stop any continuous run
    if not st.session_state.get('training_started', False) or st.session_state.agent is None:
        initialize_or_reset_components(st.session_state.selected_env) # This will show its own toast
        # Initial frame already rendered by initialize_or_reset and handled by update_ui_elements if called after

    if st.session_state.current_episode_number < st.session_state.max_episodes:
        st.toast("Training one episode...", icon="⏳")
        # Loop for one complete episode
        while True:
            frame, reward, done, info = run_training_step()
            update_ui_elements(frame) # Update UI for each step in the episode
            if done:
                if st.session_state.current_episode_number >= st.session_state.max_episodes: # Check if max was reached *during* this episode
                    st.toast(f"Max episodes ({st.session_state.max_episodes}) reached.")
                else: # Only show "finished one episode" if max episodes not also hit
                    st.toast("Finished training one episode.", icon="✅")
                break
    else:
        st.toast(f"Max episodes ({st.session_state.max_episodes}) already reached. Please reset.", icon="🚫")

    update_ui_elements(None) # Ensure UI is up-to-date before rerun
    st.rerun()

def callback_toggle_continuous_training():
    if not st.session_state.get('training_started', False) or st.session_state.agent is None:
        initialize_or_reset_components(st.session_state.selected_env) # This will show its own toast
        st.session_state.training_in_progress = True # Start immediately after init
        st.toast("Continuous training started!", icon="▶️")
    else:
        st.session_state.training_in_progress = not st.session_state.training_in_progress
        if st.session_state.training_in_progress:
            st.toast("Continuous training resumed!", icon="▶️")
        else:
            st.toast("Continuous training paused.", icon="⏸️")

    update_ui_elements(None) # Ensure UI is up-to-date before rerun
    st.rerun()

# --- Checkpointing Functions ---
def save_checkpoint():
    if st.session_state.agent is None:
        st.error("Agent not initialized. Cannot save checkpoint.")
        return

    env_name = st.session_state.selected_env.replace('-', '_')
    episode = st.session_state.current_episode_number
    filename = f"{env_name}_episode_{episode}_checkpoint.pth"
    filepath = os.path.join(CHECKPOINT_DIR, filename)

    checkpoint_data = {
        'qnetwork_local_state_dict': st.session_state.agent.qnetwork_local.state_dict(),
        'qnetwork_target_state_dict': st.session_state.agent.qnetwork_target.state_dict(),
        'optimizer_state_dict': st.session_state.agent.optimizer.state_dict(),
        'selected_env': st.session_state.selected_env,
        'current_episode_number': st.session_state.current_episode_number,
        'total_steps': st.session_state.total_steps,
        'current_epsilon': st.session_state.current_epsilon,
        'episode_rewards': st.session_state.episode_rewards,
        'epsilon_values': st.session_state.epsilon_values,
        # Save buffer if small enough / desired - can be large!
        'buffer_memory': list(st.session_state.replay_buffer.memory) if st.session_state.replay_buffer else []
    }
    try:
        with st.spinner("Saving checkpoint..."):
            torch.save(checkpoint_data, filepath)
        st.success(f"Checkpoint saved to {filepath}")
    except Exception as e:
        st.error(f"Error saving checkpoint: {e}")

def load_checkpoint(filepath):
    if not os.path.exists(filepath):
        st.error(f"Checkpoint file not found: {filepath}")
        return

    with st.spinner(f"Loading checkpoint {os.path.basename(filepath)}..."):
        try:
            checkpoint_data = torch.load(filepath, map_location=DEVICE)
        except Exception as e:
            st.error(f"Error loading checkpoint file: {e}")
            return

        saved_env_name = checkpoint_data.get('selected_env', st.session_state.selected_env)

        # If environment has changed or agent not initialized, re-initialize
        # Note: initialize_or_reset_components itself has a spinner
        if saved_env_name != st.session_state.selected_env or st.session_state.agent is None:
            st.session_state.selected_env = saved_env_name
            initialize_or_reset_components(saved_env_name)
            st.info(f"Agent & environment for {saved_env_name} initialized before loading checkpoint data.")

        # Now that agent and its components are surely initialized:
        try:
            st.session_state.agent.qnetwork_local.load_state_dict(checkpoint_data['qnetwork_local_state_dict'])
            st.session_state.agent.qnetwork_target.load_state_dict(checkpoint_data['qnetwork_target_state_dict'])
        st.session_state.agent.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])

        st.session_state.current_episode_number = checkpoint_data.get('current_episode_number', 0)
        st.session_state.total_steps = checkpoint_data.get('total_steps', 0)
        st.session_state.current_epsilon = checkpoint_data.get('current_epsilon', st.session_state.epsilon_start)
        st.session_state.episode_rewards = checkpoint_data.get('episode_rewards', [])
        st.session_state.epsilon_values = checkpoint_data.get('epsilon_values', [])

        if 'buffer_memory' in checkpoint_data and hasattr(st.session_state.replay_buffer, 'memory'):
            buffer_size_on_load = st.session_state.config.BUFFER_SIZE if st.session_state.config else 10000 # Fallback buffer size
            st.session_state.replay_buffer.memory = deque(checkpoint_data['buffer_memory'], maxlen=buffer_size_on_load)

        st.session_state.training_started = True
        st.session_state.training_in_progress = False # Ensure not auto-starting

        # Reset current_game_state from the loaded environment
        if st.session_state.environment:
            current_seed = st.session_state.config.SEED if st.session_state.config else None
            st.session_state.current_game_state, _ = st.session_state.environment.reset(seed=current_seed)
            # Render the first frame of the loaded state for UI update
            frame_to_show = st.session_state.environment.render()
            update_ui_elements(frame_to_show)
        else:
            update_ui_elements(None)

        st.success(f"Checkpoint loaded from {filepath}")
        st.rerun()

    except Exception as e:
        st.error(f"Error applying checkpoint data: {e}. Agent components might be partially loaded or mismatched.")


# --- UI Update Function ---
def update_ui_elements(frame):
    # Game View
    if frame is not None:
        game_view_placeholder.image(frame, caption=f"Episode: {st.session_state.current_episode_number} / {st.session_state.max_episodes}, Step: {st.session_state.current_episode_steps}")
    else:
        if st.session_state.training_started and st.session_state.environment is not None and st.session_state.current_game_state is not None:
             img = st.session_state.environment.render()
             game_view_placeholder.image(img, caption="Initial Game State / Current State")
        else:
             game_view_placeholder.text("Game view will update once training starts/resumes.")

    # Metrics Charts
    if st.session_state.episode_rewards:
        reward_chart_placeholder.line_chart(st.session_state.episode_rewards, use_container_width=True)
    else:
        reward_chart_placeholder.empty()

    if st.session_state.epsilon_values:
        epsilon_chart_placeholder.line_chart(st.session_state.epsilon_values, use_container_width=True)
    else:
        epsilon_chart_placeholder.empty()

    # Textual Details
    details_placeholder.empty()
    with details_placeholder.container():
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Episode", f"{st.session_state.current_episode_number} / {st.session_state.max_episodes}")
        col2.metric("Total Steps", st.session_state.total_steps)
        col3.metric("Current Epsilon", f"{st.session_state.current_epsilon:.4f}")
        col4.metric("Current Ep. Reward", f"{st.session_state.current_episode_reward:.2f}")

        if len(st.session_state.episode_rewards) > 0:
            avg_reward_window = min(100, len(st.session_state.episode_rewards))
            avg_reward = np.mean(st.session_state.episode_rewards[-avg_reward_window:])
            # Use st.columns to place this metric below the others or handle layout if it becomes too wide
            st.metric("Avg Reward (last " + str(avg_reward_window) + ")", f"{avg_reward:.2f}")
        else:
            # Placeholder for avg reward if no episodes completed yet
            st.metric("Avg Reward (last 100)", "N/A")


# --- Page Configuration ---
st.set_page_config(
    page_title="DQN Agent Training Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 DQN Agent Training Dashboard")

st.sidebar.header("Controls & Settings")

# Environment selection is disabled if training is in progress
env_select_disabled = st.session_state.get('training_in_progress', False)
st.sidebar.selectbox("Select Environment", options=list(AVAILABLE_ENVS.keys()), key="selected_env", disabled=env_select_disabled)

# Start/Reset button is disabled if training is in progress
start_reset_disabled = st.session_state.get('training_in_progress', False)
st.sidebar.button("Start / Reset Training", on_click=initialize_or_reset_components_callback, disabled=start_reset_disabled)

# Train One Episode button is disabled if training is in progress
train_one_disabled = st.session_state.get('training_in_progress', False)
st.sidebar.button("Train One Episode", on_click=callback_train_one_episode, disabled=train_one_disabled)

# Toggle Continuous Training button
toggle_button_text = "Pause Continuous Training" if st.session_state.get('training_in_progress', False) else "Resume Continuous Training"
st.sidebar.button(toggle_button_text, on_click=callback_toggle_continuous_training)

# --- Configuration Details Expander ---
with st.sidebar.expander("⚙️ Current Configuration", expanded=False):
    if st.session_state.get('training_started', False) and st.session_state.selected_env in AVAILABLE_ENVS:
        st.subheader("Environment:")
        st.markdown(f"**Name:** `{st.session_state.selected_env}`")
        st.markdown(f"**State Size:** `{AVAILABLE_ENVS[st.session_state.selected_env]['state_size']}`")
        st.markdown(f"**Action Size:** `{AVAILABLE_ENVS[st.session_state.selected_env]['action_size']}`")
        st.markdown(f"**Max Steps/Episode:** `{st.session_state.max_steps_per_episode}`")
    else:
        st.caption("No environment initialized yet.")

    if st.session_state.config:
        st.subheader("Agent Hyperparameters (DQNConfig):")
        st.markdown(f"**Buffer Size:** `{st.session_state.config.BUFFER_SIZE}`")
        st.markdown(f"**Batch Size:** `{st.session_state.config.BATCH_SIZE}`")
        st.markdown(f"**Gamma:** `{st.session_state.config.GAMMA}`")
        st.markdown(f"**Tau:** `{st.session_state.config.TAU}`")
        st.markdown(f"**LR:** `{st.session_state.config.LR}`")
        st.markdown(f"**Update Every:** `{st.session_state.config.UPDATE_EVERY}`")
    else:
        st.caption("Agent config not loaded yet.")

    st.subheader("Epsilon Parameters:")
    st.markdown(f"**Start Epsilon:** `{st.session_state.epsilon_start}`")
    st.markdown(f"**End Epsilon:** `{st.session_state.epsilon_end}`")
    st.markdown(f"**Epsilon Decay:** `{st.session_state.epsilon_decay}`")

st.sidebar.write("---") # Separator

# Checkpointing UI
st.sidebar.header("🏁 Checkpointing")
save_disabled = st.session_state.get('training_in_progress', False) or not st.session_state.get('training_started', False)
st.sidebar.button("Save Agent Checkpoint", on_click=save_checkpoint, disabled=save_disabled)

st.sidebar.write("---")

# File uploader for loading checkpoint
uploaded_file = st.sidebar.file_uploader("Upload & Load Checkpoint (.pth)", type=["pth"],
                                         key="checkpoint_uploader",
                                         disabled=st.session_state.get('training_in_progress', False))
if uploaded_file is not None:
    # Ensure not to reload if already processing or training is ongoing
    if not st.session_state.get('training_in_progress', False):
        temp_load_path = os.path.join(CHECKPOINT_DIR, uploaded_file.name)
        try:
            with open(temp_load_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            load_checkpoint(temp_load_path)
            # Try to reset uploader state by rerunning; direct reset is tricky
            # st.session_state.checkpoint_uploader = None # This might not work as expected due to Streamlit's execution
            st.rerun()
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
        finally:
            # Clean up temp file if it exists
            if os.path.exists(temp_load_path):
                 os.remove(temp_load_path)


st.sidebar.write("Or select an existing checkpoint to load:")
try:
    existing_checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")], reverse=True)
except FileNotFoundError:
    existing_checkpoints = []
    st.sidebar.caption(f"Checkpoint directory {CHECKPOINT_DIR} not found or empty.")


selected_checkpoint_file = st.sidebar.selectbox("Load from existing", options=[""] + existing_checkpoints,
                                                key="selected_checkpoint",
                                                disabled=st.session_state.get('training_in_progress', False))

load_selected_disabled = (not st.session_state.selected_checkpoint) or st.session_state.get('training_in_progress', False)
if st.sidebar.button("Load Selected Checkpoint", disabled=load_selected_disabled):
    filepath_to_load = os.path.join(CHECKPOINT_DIR, st.session_state.selected_checkpoint)
    load_checkpoint(filepath_to_load)

# Placeholder for future elements
# st.sidebar.write("Environment selection and agent controls will appear here.") # Removed as widgets are added above

st.header("Training Progress")
# Placeholder for charts and game view
col1, col2 = st.columns(2)
with col1:
    st.subheader("Game View")
    game_view_placeholder = st.empty()
    if st.session_state.training_started and st.session_state.environment is not None and st.session_state.current_game_state is not None:
        # The environment.reset in initialize_or_reset_components already provides the initial state
        # and renders the first frame. We just need to display it if available.
        img = st.session_state.environment.render()
        game_view_placeholder.image(img, caption="Initial Game State / Current State")
    else:
        game_view_placeholder.text("Game window will appear here once training starts.")

with col2:
    st.subheader("Metrics")
    # st.text("Charts for rewards, epsilon, etc., will be shown here.") # Text removed, placeholders will be filled by update_ui
    reward_chart_placeholder = st.empty()
    epsilon_chart_placeholder = st.empty()

st.header("Agent & Environment Details")
# Placeholder for agent and environment details
details_placeholder = st.empty()
# details_placeholder.text("Agent and environment parameters will be displayed here.") # Text removed

if __name__ == '__main__':
    # Initial UI setup or update based on current state
    # This will render the initial view or the view from a previous step if script reran
    if not st.session_state.get('training_in_progress', False):
        # If not actively training, ensure the latest frame (if any) or initial state is shown
        # This handles the case after "Train One Episode" or when app is first loaded.
        current_frame = None
        if st.session_state.get('training_started', False) and st.session_state.environment is not None:
            # Potentially render current state if env exists, useful after reset or one episode
            if st.session_state.current_game_state is not None:
                 current_frame = st.session_state.environment.render()
        update_ui_elements(current_frame)


    # Main logic for continuous training when st.session_state.training_in_progress is True
    if st.session_state.get('training_started', False) and st.session_state.get('training_in_progress', False):
        if st.session_state.current_episode_number < st.session_state.max_episodes:
            frame, reward, done, info = run_training_step()
            update_ui_elements(frame) # Update UI with the new frame and metrics

            # If the episode ended and max episodes reached by this step, training_in_progress will be set to False
            # in run_training_step. The rerun below will then correctly update button states etc.
            # If max episodes not reached, or episode not done, continue training.
            if st.session_state.training_in_progress: # Check if it's still true after run_training_step
                 st.rerun() # Key for continuous loop
            else: # Training was stopped by run_training_step (e.g. max episodes)
                 # update_ui_elements(frame) # Already called
                 st.rerun() # Rerun to update button states etc.

        else: # Max episodes already met before this iteration
            st.session_state.training_in_progress = False
            st.toast(f"Max episodes ({st.session_state.max_episodes}) reached. Pausing training.")
            update_ui_elements(None) # Update UI to reflect paused state
            st.rerun() # Rerun once more to update UI state (e.g. button text)
    # else:
        # If not training_in_progress, the script finishes, and Streamlit waits for next interaction.
        # update_ui_elements(None) was called at the start of this block if not training.
        # No explicit pass needed here.
