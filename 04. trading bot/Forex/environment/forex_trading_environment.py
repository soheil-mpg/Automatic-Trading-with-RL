
# Import the libraries
import enum
import gym
import time
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT 
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Import the local libraries
from data_helper import *

# ============== #
#  Action Space  #
# ============== #
class Actions(enum.Enum):
    
    # Set digits to each actions
    BUY, SELL, HOLD = 0, 1, 2


# ===================== #
#  Trading Environment  #
# ===================== #
class TradingEnvironment(gym.Env):
    
    # Meta data
    metadata = {'render.modes': ['human']}
    
    # ============= #
    #  Constructor  #
    # ============= #
    def __init__(self, dataset = 0, trading_type = "virtual"):

        # Initialization
        self.trading_type = trading_type   # Trading type ("virtual" or "real time")
        self.margin_indicator = 200        # Margin indicator of 200:1
        self.window_size = 500             # Window size (for data preparation section)
        self.dataset_step = 0
        self.episode = 1
        self.NETWORTH_LIST_VIS = []
        self.COMMISSION_COST = 0
        self.COMMISSION_RATE = 0

        # If trading in real time
        if (trading_type == "real time"):

            # Specify the ticker name that you want to trade in real time
            self.ticker_name = "USD/CAD"

            # Initialize the temp dataset for online trading
            self.dataset_temp_online = pd.DataFrame(columns = ["Ticker", "Sell", "Buy"])

            # Open forex.com + Sign in + Go the the fx-major tabs
            self.driver = open_forex_website()

        # If trading virtually
        elif (trading_type == "virtual"):

            # Make sure a dataset has been given
            assert (type(dataset) != int), "Uh oh! No dataset has been given."

            # Initialize the dataset
            self.df = dataset

        # Reset the state and set the initial info and profiles
        self.reset()

        # Action space + Observation space
        self.action_space = gym.spaces.Discrete(len(Actions))
        self.observation_space = gym.spaces.Box(low = -np.inf, high = np.inf, shape = (34,), dtype = np.float32)

    # ========================== #
    #  Get the Next Observation  #
    # ========================== #
    def _generate_observation(self):

        # If trading in real time
        if (self.trading_type == "real time"):

            # If there is not enough dataset then add more
            if (self.dataset_temp_online.shape[0] <= self.window_size):
                
                # Fill dataset_temp_online with enough data
                while (self.dataset_temp_online.shape[0] <= self.window_size):

                    # Get the prices in real time (for all tickers)
                    current_dataset = gather_data_online(driver = self.driver)

                    # Add current ticker data (includes ticker, sell, buy column) into dataset_temp_online
                    self.dataset_temp_online = self.dataset_temp_online.append(current_dataset[current_dataset["Ticker"] == self.ticker_name], ignore_index = True)

            # If there is enough dataset
            else:

                # Get the prices in real time
                current_dataset = gather_data_online(driver = self.driver)

                # Add current ticker data (includes ticker, sell, buy column) into dataset_temp_online
                self.dataset_temp_online = self.dataset_temp_online.append(current_dataset[current_dataset["Ticker"] == self.ticker_name], ignore_index = True)

                # Delete first row + Reset index
                self.dataset_temp_online = self.dataset_temp_online.iloc[1:, :].reset_index().iloc[:, 1:]

            # Add more features to the dataset
            df = add_features(self.dataset_temp_online)

            # Get the per share price
            self.per_share_price_buy, self.per_share_price_sell = current_dataset["Buy"], current_dataset["Sell"]

        # If trading virtually
        elif (self.trading_type == "virtual"):

            # Reset the dataset step if reaching to the end of dataset
            if (self.dataset_step >= self.df.shape[0]):   self.dataset_step = 0

            # Get asset information
            current_dataset = np.array(self.df.drop(["Timestep"], axis = 1).loc[self.dataset_step])

            # Get the per share price
            self.per_share_price_buy, self.per_share_price_sell = self.df["Buy"].loc[self.dataset_step], self.df["Sell"].loc[self.dataset_step]   

        # Get the investor profile
        investor_info = np.array([self.NETWORTH, self.SHARES_WORTH, self.BALANCE_ACCOUNT])

        # Append all observations together
        observations = np.append(current_dataset, investor_info)
        
        # Increment the dataset step
        self.dataset_step += 1

        return observations           
        
    # ========================== #
    #   BUY SHARES OF A STOCK    #
    # ========================== #
    def buy_shares(self, price_per_unit):

        # Buying power
        buying_power = self.BALANCE_ACCOUNT * self.margin_indicator

        # Set the quantity as 90% of total buying power
        quantity = np.floor((buying_power * 0.9) / price_per_unit)

        # Calculate the price on margin using leverage
        price_using_leverage = (quantity * price_per_unit) / self.margin_indicator

        # If there is enough money 
        if (price_using_leverage <= self.BALANCE_ACCOUNT):

            # If buying in real time
            if (self.trading_type == "real time"):

                # Buy from forex.com using selenium
                buy_sell_forex(driver = self.driver, action = "buy", ticker_name = self.ticker_name, quantity = quantity)

            # If buying virtually
            elif (self.trading_type == "virtual"):

                # Pay the 
                self.BALANCE_ACCOUNT -= price_using_leverage

                # Update number of holding shares
                self.num_holding_shares += quantity

        # Update the investor's profile
        self.update_investor_profile()
            
    
    # =========================== #
    #   SELL SHARES OF A STOCK    #
    # =========================== #
    def sell_shares(self, price_per_unit, quantity):

        # Calculate the price on margin using leverage
        price_using_leverage = (quantity * price_per_unit) / self.margin_indicator

        # If selling in real time
        if (self.trading_type == "real time"):

            # Sell from forex.com using selenium
            buy_sell_forex(driver = self.driver, action = "sell", ticker_name = self.ticker_name, quantity = quantity)

        # If selling virtually
        elif (self.trading_type == "virtual"):

            # If there is enough asset to sell
            if (quantity <= self.num_holding_shares):

                # Get the money
                self.BALANCE_ACCOUNT += price_using_leverage

                # Remove shares from the holding shares
                self.num_holding_shares -= quantity

        # Update the investor's profile
        self.update_investor_profile()
            
    
    # ================== #
    #     TAKE ACTION    #
    # ================== #
    def take_action(self, action):

        # Action to take
        action_to_take = action

        # If BUYING
        if (action_to_take == 0):

            # Buy 
            self.buy_shares(price_per_unit = self.per_share_price_buy)

        # If SELLING 
        elif (action_to_take == 1):

            # Sell
            self.sell_shares(price_per_unit = self.per_share_price_sell, quantity = self.num_holding_shares)

        # If HOLDING 
        elif (action_to_take == 2):

            # Add reward
            self.reward += 10 ** 5
        
    # =========================== #
    #   Update Investor Profile   #
    # =========================== #
    def update_investor_profile(self):

        # If trading in real time
        if (self.trading_type == "real time"):

            # Get the prices in real time
            current_dataset = gather_data_online(driver = self.driver)

            # Get the dataset for a specific ticker
            current_dataset = current_dataset[current_dataset["Ticker"] == self.ticker_name]   # the dataset have three columns: ticker, sell, buy

            # Get the per share price
            self.per_share_price_buy = current_dataset["Buy"]
            self.per_share_price_sell = current_dataset["Sell"]
            
            # Update the "Shares Worth"
            self.SHARES_WORTH = (self.num_holding_shares * self.per_share_price_sell) / self.margin_indicator

            # Update the "Networth"
            self.NETWORTH = self.driver.find_elements_by_class_name("balance-bar__item-number")[1].text   # Net Equity in forex.com

        # If trading virtually
        elif (self.trading_type == "virtual"):

            # Update the per share price
            self.per_share_price_buy = self.df["Buy"].loc[self.dataset_step]
            self.per_share_price_sell = self.df["Sell"].loc[self.dataset_step]

            # Update the "Shares Worth"
            self.SHARES_WORTH = (self.num_holding_shares * self.per_share_price_sell) / self.margin_indicator

            # Update the "Networth"
            self.NETWORTH = self.BALANCE_ACCOUNT + self.SHARES_WORTH
        
        
    # ============= #
    #     Step      #
    # ============= #
    def step(self, action, rendering = True):

        # Take action
        self.take_action(action)

        # Calculate profit if trading in real time
        if (self.trading_type == "real time"):   self.profit = driver.find_elements_by_class_name("balance-bar__item-number")[3].text     # unrealised P&L in forex.com
        
        # If trading virtually
        elif (self.trading_type == "virtual"):   self.profit = self.NETWORTH - self.initial_networth

        # Add profit as reward
        self.reward += ((self.profit * 100) ** 3)

        # Initialize done
        done = False

        # Game over - Set done flag on when reaching to 1000 steps of dataset
        if (self.dataset_step > 0) and (self.dataset_step % 1000 == 0):   done = True

        # Get the next observations
        observations = self._generate_observation()

        # Render
        if (rendering == True):  self.render(action)

        # Increment the episode number if in terminal state
        if done:  self.episode += 1

        return observations, self.reward, done, {} 
    

    # ============= #
    #     Reset     #
    # ============= #
    def reset(self):

        # If selling in real time
        if (self.trading_type == "real time"):

            # Investor's information
            self.SHARES_WORTH = 0
            self.BALANCE_ACCOUNT =  driver.find_elements_by_class_name("balance-bar__item-number")[2].text   # Cash in forex.com
            self.NETWORTH = self.driver.find_elements_by_class_name("balance-bar__item-number")[1].text   # Net Equity in forex.com

            # Get the prices in real time
            current_dataset = gather_data_online(driver = self.driver)

            # Get the dataset for a specific ticker
            current_dataset = current_dataset[current_dataset["Ticker"] == self.ticker_name]   # the dataset have three columns: ticker, sell, buy

            # Get the per share price
            self.per_share_price_buy, self.per_share_price_sell = current_dataset["Buy"], current_dataset["Sell"]

        # If selling virtually
        elif (self.trading_type == "virtual"):

            # Investor's information
            self.SHARES_WORTH = 0
            self.BALANCE_ACCOUNT = 100
            self.NETWORTH = self.BALANCE_ACCOUNT + self.SHARES_WORTH

            # Update the per share price
            self.per_share_price_buy, self.per_share_price_sell = self.df["Buy"].loc[self.dataset_step], self.df["Sell"].loc[self.dataset_step]   

        # Initialization
        self.initial_networth = self.NETWORTH
        self.num_holding_shares = 0
        self.reward = 0

        # Make NETWORTH_LIST_VIS more sufficient
        if len(self.NETWORTH_LIST_VIS) >= 10000:  self.NETWORTH_LIST_VIS = self.NETWORTH_LIST_VIS[1000:]

        # Update investor's information
        self.update_investor_profile()

        return self._generate_observation()
        
        
    # ============= #
    #    Render     #
    # ============= #
    def render(self, action, mode = 'human', close = False):
        """
        Rendering function for reporting the necessary information.
        """
        # Clear out the kernel
        clear_output(wait = True)

        # Get the action type in strings
        if (action == 0):   action_type = "🟢 BUY"
        elif (action == 1): action_type = "🔴 SELL"
        elif (action == 2): action_type = "🟡 HOLD"

        # Networth 
        if (self.NETWORTH >= self.initial_networth): current_networth = "{:.9f} 🔥".format(self.NETWORTH)
        else: current_networth = "{:.9f} 👎".format(self.NETWORTH)

        # Update investor's information
        self.update_investor_profile()

        # Report
        print("\t\t\t      ===================================================")
        print("\t\t\t      \t               EPISODE {}".format(self.episode))
        print("\t\t\t      \t              TIMESTEP {}".format(self.dataset_step % 1000))
        print("\t\t\t      ===================================================")
        print("\t\t\t           Action:                       {}".format(action_type))
        print("\t\t\t           Profit:                     $ {:.9f}".format(self.profit))
        print("\t\t\t           Reward:                       {:.9f}".format(np.floor(self.reward)))
        print("\t\t\t        .............................................")
        print("\t\t\t           Account Balance:            $ {:.9f}".format(self.BALANCE_ACCOUNT))
        print("\t\t\t           Initial Networth:           $ {:.9f}".format(self.initial_networth))
        print("\t\t\t           Current Networth:           $ {}".format(current_networth))
        print("\t\t\t        .............................................")
        print("\t\t\t           Per Share Price (Buy):      $ {:.9f}".format(self.per_share_price_buy))
        print("\t\t\t           Per Share Price (Sell):     $ {:.9f}".format(self.per_share_price_sell))
        print("\t\t\t           Number of Holding Shares:     {}".format(self.num_holding_shares))
        print("\t\t\t           Shares Dollar Values:       $ {:.9f}".format(self.num_holding_shares * self.per_share_price_sell))
        print("\t\t\t      ===================================================")

        # Save current networth
        self.NETWORTH_LIST_VIS.append(self.NETWORTH)

        # Visualization - Networth
        plt.figure(figsize = (16, 4))
        plt.plot(self.NETWORTH_LIST_VIS, "red")
        plt.vlines(x = np.array(range(0, len(self.NETWORTH_LIST_VIS), 1000)), ymin = min(self.NETWORTH_LIST_VIS), ymax = max(self.NETWORTH_LIST_VIS), colors = "gray", linestyles = 'dashed',)
        plt.xlabel("Timestep", fontsize = 15)
        plt.ylabel("Networth", fontsize = 15)
        plt.show()        
