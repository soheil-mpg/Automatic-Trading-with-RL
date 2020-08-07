
# Import the libraries
import numpy as np
import pandas as pd
from glob import glob
import matplotlib as mpl
import mplfinance as mpf
from typing import Iterable
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from IPython.display import clear_output
from IPython.core.display import display, HTML
import gym, gym.wrappers, time, requests, json, enum, warnings, pathlib

# Import the deep learning libraries
import torch
from torch import nn
import torch.utils.data
import torch.optim as optim
from torch.functional import F
from torchsummary import summary
import ptan
import ptan.ignite as ptan_ignite
from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger
from typing import Iterable

# Import the local libraries
from models import cnn_lstm
from environment import forex_trading_environment
from reinforcement_learning import rainbow
from data_helper import *

# ===================== #
#   Rainbow Algorithm   #
# ===================== #
class RainbowAlgorithm():
    
    # Set the device automatically
    device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ===================== #
    #      Constructor      #
    # ===================== #
    def __init__(self):
        
        # Hyperparameters
        self.GAMMA = 0.99
        self.BATCH_SIZE = 64
        self.BARS_COUNT = 10
        self.REWARD_STEPS = 2
        self.LEARNING_RATE = 1e-4
        self.STATES_TO_EVALUATE = 1000
        self.REPLAY_SIZE, self.REPLAY_INITIAL = 100000, 10000
        self.EPS_START, self.EPS_FINAL, self.EPS_STEPS = 1.0, 0.1, 1000000
        
        # Metrics
        self.METRICS = ('episode_reward', 'episode_steps', 'order_profits', 'order_steps',)

        # Path for training dataset
        self.TRAIN_PATH = "./datasets/train/"

        # Path for validation dataset
        self.VAL_PATH = "./datasets/test/"
        
        
    # ===================== #
    #      State-Values     #
    # ===================== #

    # No gradient calculation
    @torch.no_grad()

    # Function for calculating the state-values
    def calculate_state_values(self, states, network, device = device_type):

        # Initialize an empty list for mean of values
        mean_vals = []

        # Split states into bach_size splits
        all_batches = np.array_split(ary = states, indices_or_sections = self.BATCH_SIZE)

        # Loop over each batch
        for i_batch in all_batches:

            # Convert the bactch into a tensor + Convert to device
            states_v = torch.tensor(i_batch).to(device)

            # Feedforward states to get the action values
            action_values_v = network(states_v)

            # Get the best action values (by choosing the maximum values)
            best_action_values_v = action_values_v.max(1)[0]

            # Append best action values into a list
            mean_vals.append(best_action_values_v.mean().item())

        # Get the mean of values
        output = np.mean(mean_vals)

        return output


    # ===================== #
    #   Loss Calculation    #
    # ===================== #
    def calculate_loss(self, batch, network, target_network, gamma, device = device_type):

        # Unpack the given batch
        states, actions, rewards, dones, next_states = self.unpack_batch(batch)

        # Convert values to tensor + Convert to device
        states_v, next_states_v, actions_v, rewards_v, done_mask = torch.tensor(states).to(device), torch.tensor(next_states).to(device), torch.tensor(actions).to(device), torch.tensor(rewards).to(device), torch.BoolTensor(dones).to(device)

        # Get state-action values by feedforwarding the states
        state_action_values = network(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

        # Get next state-action values by feedforwarding the next states (SOURCE NETWORK)
        next_state_actions = network(next_states_v).max(1)[1]

        # Get next state-action values by feedforwarding the next states (TARGET NETWORK)
        next_state_values = target_network(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)

        # Mask next state values
        next_state_values[done_mask] = 0.0

        # Bellman equation
        expected_state_action_values = next_state_values.detach() * gamma + rewards_v

        # MSE loss
        loss = nn.MSELoss()(state_action_values, expected_state_action_values)

        return loss


    # ===================== #
    #    Batch Generator    #
    # ===================== #
    def batch_generator(self, replay_buffer: ptan.experience.ExperienceReplayBuffer, initial: int, batch_size: int):

        # Populate buffer with initial
        replay_buffer.populate(initial)

        # Infinite loop
        while True:

            # Populate buffer
            replay_buffer.populate(1)

            # Sample from buffer and yield it
            yield replay_buffer.sample(batch_size)


    # ===================== #
    #     Setup Ignite      #
    # ===================== #
    def setup_ignite(self, engine: Engine, experience_source, run_name: str, extra_metrics: Iterable[str] = ()):

        # Ignore warning for missing metrics
        warnings.simplefilter("ignore", category = UserWarning)

        # End of episode handler + Attach engine to handler
        ptan_ignite.EndOfEpisodeHandler(experience_source, subsample_end_of_episode = 100).attach(engine)

        # Episode FPS handler + Attach engine to handler
        ptan_ignite.EpisodeFPSHandler().attach(engine)

        # Function for reporting
        @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
        def episode_completed(trainer: Engine):

            # Time passed
            passed = trainer.state.metrics.get('time_passed', 0)

            # Report
            print("          ================================================")
            print("          \t          DL Report - EPISODE {}".format(trainer.state.episode))
            print("          ================================================")
            print("                      Reward:    {}".format(trainer.state.episode_reward))
            print("                      Steps:    {}".format(trainer.state.episode_steps))
            print("                      Speed:    {}  f/s".format(trainer.state.metrics.get('avg_fps', 0)))
            print("                      Elapsed:    {}".format(timedelta(seconds = int(passed))))
            print("          ================================================")

        # Get the current time
        now = datetime.now().isoformat(timespec = 'minutes')

        # Tensorboard logger
        tb = tb_logger.TensorboardLogger(log_dir = "runs/{}-{}".format(now, run_name))

        # Running average handler + Attach engine to handler
        RunningAverage(output_transform = lambda v: v['loss']).attach(engine, "avg_loss")

        # Add reward, step, average reward
        tb.attach(engine,
                  log_handler = tb_logger.OutputHandler(tag = "episodes", metric_names = ['reward', 'steps', 'avg_reward']),
                  event_name = ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)

        # Period events handler + Attach engine to handler
        ptan_ignite.PeriodicEvents().attach(engine)

        # Add average loss, average
        tb.attach(engine,
                  log_handler = tb_logger.OutputHandler(tag = "train", metric_names = ['avg_loss', 'avg_fps'].extend(extra_metrics), output_transform = lambda a: a),
                  event_name = ptan_ignite.PeriodEvents.ITERS_1000_COMPLETED)

        return tb


    # ===================== #
    #     Unpack Batches    #
    # ===================== #
    def unpack_batch(self, batch):

        # Initialize empty lists for S, A, R, done, S'
        states, actions, rewards, dones, last_states = [], [], [], [], []

        # Loop over each batch
        for exp in batch:

            # Add each "state", "action", "reward", "done", "last_state" into the list
            states.append(np.array(exp.state, copy = False))
            actions.append(exp.action)
            rewards.append(exp.reward)
            dones.append(exp.last_state is None)
            if exp.last_state is None:   last_states.append(exp.state)       # the result will be masked anyway
            else:                        last_states.append(np.array(exp.last_state, copy = False))

        # Convert lists into arrays
        states, actions, rewards, dones, last_states = np.array(states, copy = False), np.array(actions), np.array(rewards, dtype = np.float32), np.array(dones, dtype = np.uint8), np.array(last_states, copy = False)

        return states, actions, rewards, dones, last_states


    # ===================== #
    #       Validation      #
    # ===================== #
    def test_model(self, environment, network, episodes = 100, device = device_type, epsilon = 0.02, comission = 0.0):

        # Initialize stats
        stats = {metric: [] for metric in self.METRICS}

        # Iterate through episode range
        for _ in range(episodes):

            # Reset the environment and get the initial observation
            observations = environment.reset()

            # Initializations
            total_reward, position, position_steps, episode_steps = 0.0, None, None, 0

            # Infinite loop
            while True:

                # Convert observations into tensor + Convert it to device
                observations_v = torch.tensor([observations]).to(device)

                # Feedforward the observations
                output_v = network(observations_v)

                # Get the action (considering gamma)
                if np.random.random() < epsilon: action_idx, action_shares_num = environment.action_space.sample()
                else:                            action_idx, action_shares_num = output_v.max(dim = 1)[1].item()

                # Name of the action
                action = Actions(action_idx)

                # Get the last close price
                last_close_price = self.per_share_price

                # If action is "buy" AND position is none
                if (action == Actions.Buy) and (position is None):

                    # Assign close price to position
                    position = last_close_price

                    # Set the position step to zero
                    position_steps = 0

                # If action is "sell" AND position is defined
                elif (action == Actions.Close) and (position is not None):

                    # Get the profit
                    profit = env.NETWORTH - env.initial_networth

                    # Add profit and position step to stats
                    stats['order_profits'].append(profit)
                    stats['order_steps'].append(position_steps)

                    # Set position and position steps to none
                    position, position_steps = None, None

                # Take action and get the observation, reward, done, extra_info
                observations, reward, done, _ = environment.step((action_idx, action_shares_num))

                # Add reward to total rewards
                total_reward += reward

                # Increment the episode steps
                episode_steps += 1

                # If position steps is defined
                if (position_steps is not None):

                    # Increment the position steps
                    position_steps += 1

                # If terminal state
                if done:

                    if (position is not None):

                        # Calculate the profit
                        profit = env.NETWORTH - env.initial_networth

                        # Add order profits and order steps to stats
                        stats['order_profits'].append(profit)
                        stats['order_steps'].append(position_steps)

                    # Break the loop
                    break

            # Add episode reward and episode steps to stas
            stats['episode_reward'].append(total_reward)
            stats['episode_steps'].append(episode_steps)

        # Get the final resutl
        result = {key: np.mean(vals) for key, vals in stats.items()}

        return result


    # ===================== #
    #    Run the Training   #
    # ===================== #
    def train_model(self):

        # Saving path
        saves_path = "./saved models/"

        # Load the training dataset + Add more features
        dataset_training = pd.read_csv("./datasets/train/dataset.csv").reset_index()[["ticker", "time", "sell", "buy"]].sort_values(by = ["ticker", "time"])
        dataset_training.columns = ["Ticker", "Timestep", "Sell", "Buy"]
        dataset_training = add_features(dataset_training)
        
        # Load the test dataset + Add more features
        dataset_test = pd.read_csv("./datasets/train/dataset.csv").reset_index()[["ticker", "time", "sell", "buy"]].sort_values(by = ["ticker", "time"])
        dataset_test.columns = ["Ticker", "Timestep", "Sell", "Buy"]
        dataset_test = add_features(dataset_test)
        
        # Load the validation dataset + Add more features
        dataset_validation = pd.read_csv("./datasets/train/dataset.csv").reset_index()[["ticker", "time", "sell", "buy"]].sort_values(by = ["ticker", "time"])
        dataset_validation.columns = ["Ticker", "Timestep", "Sell", "Buy"]
        dataset_validation = add_features(dataset_validation)
        
        # Environments
        env_train, env_test, env_validation = forex_trading_environment.TradingEnvironment(dataset_training), forex_trading_environment.TradingEnvironment(dataset_test), forex_trading_environment.TradingEnvironment(dataset_validation)

        # Wrap environment with time limit
        #env_train = gym.wrappers.TimeLimit(env_train, max_episode_steps = 1000)

        # Source network
        network = cnn_lstm.Network(input_size = env_train.observation_space.shape[0], actions_n = env_train.action_space.n).to(self.device_type)

        # Target network
        target_network = ptan.agent.TargetNet(network)

        # Action selector (epsilon-greedy)
        action_selector = ptan.actions.EpsilonGreedyActionSelector(self.EPS_START)

        # Epsilon tracker
        epsilon_tracker = ptan.actions.EpsilonTracker(action_selector, self.EPS_START, self.EPS_FINAL, self.EPS_STEPS)

        # DQN agent
        agent = ptan.agent.DQNAgent(network, action_selector, device = self.device_type)

        # Experience source (first-last)
        experience_source = ptan.experience.ExperienceSourceFirstLast(env_train, agent, self.GAMMA, steps_count = self.REWARD_STEPS)

        # Experience replay buffer
        replay_buffer = ptan.experience.ExperienceReplayBuffer(experience_source, self.REPLAY_SIZE)

        # Adam optimizer
        optimizer = optim.Adam(network.parameters(), lr = self.LEARNING_RATE)

        # Function for processing each batch
        def process_batch(engine, batch):

            # Zero out the optimizer's weight
            optimizer.zero_grad()

            # Calculate the loss
            loss_v = self.calculate_loss(batch, network, target_network.target_model, gamma = self.GAMMA ** self.REWARD_STEPS, device = self.device_type)

            # Backpropagation
            loss_v.backward()

            # Optimize
            optimizer.step()

            # Track the epsilon
            epsilon_tracker.frame(engine.state.iteration)

            # If eval_states inside engine.state is None (if there was no value then set it to None)
            if getattr(engine.state, "eval_states", None) is None:

                # Sample from replay buffer
                eval_states = replay_buffer.sample(self.STATES_TO_EVALUATE)

                # Loop over transitions + Get each state + Convert each of them into an array
                eval_states = [np.array(transition.state, copy = False) for transition in eval_states]

                # Convert the whole states into an array
                engine.state.eval_states = np.array(eval_states, copy = False)

            # Get the loss and epsilon
            output = {"loss": loss_v.item(), "epsilon": action_selector.epsilon,}

            return output

        # Instantiate the engine
        engine = Engine(process_batch)

        # Setup the ignite
        tb = self.setup_ignite(engine, experience_source, "network", extra_metrics = ('values_mean',))

        # Set the priod event handler on for the next function
        @engine.on(ptan.ignite.PeriodEvents.ITERS_1000_COMPLETED)

        # Function for synching
        def sync_eval(engine: Engine):

            # Synch the source network's weight with target network
            target_network.sync()

            # Get the mean of values
            mean_val = self.calculate_state_values(engine.state.eval_states, network, device = self.device_type)

            # Update the metrics
            engine.state.metrics["values_mean"] = mean_val

            # If best_mean_val inside engine.state is None (if there was no value then set it to None)
            if getattr(engine.state, "best_mean_val", None) is None:

                # Assign mean_val to best_mean_val
                engine.state.best_mean_val = mean_val

            # If mean_val is larger than best_mean_val
            if engine.state.best_mean_val < mean_val:

                # Report
                print("%d: Best mean value updated %.3f -> %.3f" % (engine.state.iteration,
                                                                    engine.state.best_mean_val,
                                                                    mean_val))

                # Get the path
                path = "./saved models/mean_value-%.3f.data" % mean_val

                # Save the weights
                torch.save(network.state_dict(), path)

                # Update the best_mean_val in engine
                engine.state.best_mean_val = mean_val

        # Set the period event handler on for the next function
        @engine.on(ptan.ignite.PeriodEvents.ITERS_10000_COMPLETED)

        # Function for testing
        def validate(engine: Engine):

            # Test the model on test set
            result = self.test_model(env_tst, network, device = self.device_type)

            print("%d: tst: %s" % (engine.state.iteration, result))

            # Loop over keys and values
            for i_key, i_value in result.items():

                # Update the metrics
                engine.state.metrics[i_key + "_tst"] = i_val

            # Test the model on validation set
            result = self.test_model(env_val, network, device = self.device_type)

            print("%d: val: %s" % (engine.state.iteration, result))

            # Loop over keys and values
            for i_key, i_value in res.items():

                # Update the metrics
                engine.state.metrics[i_key + "_val"] = i_value

            # Get validation reward
            val_reward = result['episode_reward']

            # If best_val_reward inside engine.state is None (if there was no value then set it to None)
            if getattr(engine.state, "best_val_reward", None) is None:

                # Assign val_reward to best_val_reward
                engine.state.best_val_reward = val_reward

            # If val_reward is greater than best_val_reward
            if engine.state.best_val_reward < val_reward:

                # Report
                print("Best validation reward updated: %.3f -> %.3f, model saved" % (engine.state.best_val_reward,
                                                                                     val_reward))

                # Assign val_reward to best_val_reward
                engine.state.best_val_reward = val_reward

                # Saving path
                path = "./saved models/val_reward-%.3f.data" % val_reward

                # Save the weights
                torch.save(network.state_dict(), path)

        # Instantiate the period event (if 1000 iteration got completed)
        event = ptan.ignite.PeriodEvents.ITERS_10000_COMPLETED

        # Get all the test metrics
        tst_metrics = [m + "_tst" for m in self.METRICS]

        # Output handler
        tst_handler = tb_logger.OutputHandler(tag = "test", metric_names = tst_metrics)

        #
        tb.attach(engine, log_handler = tst_handler, event_name = event)

        # Get all the validation metrics
        val_metrics = [m + "_val" for m in self.METRICS]

        # Validation handler
        val_handler = tb_logger.OutputHandler(tag = "validation", metric_names = val_metrics)

        #
        tb.attach(engine, log_handler = val_handler, event_name = event)

        # Run the engine
        engine.run(self.batch_generator(replay_buffer, self.REPLAY_INITIAL, self.BATCH_SIZE))
