import time
import RPi.GPIO as gpio
import numpy as np
import random

class BotEnvironment:

	def __init__(self, sensors, motors, rewards_weights):

		gpio.setmode(gpio.BCM)
		self.sensors = sensors
		self.motors = motors
		self.__init_pins()
		self.duration = 0.08
		self.rewards_weights = np.array(rewards_weights)
		self.stack = []
		self.actions = [self.left,
						self.right,
						self.forward]

		self.rev_actions = [self.anti_left,
							self.anti_right,
							self.backward]



	def __init_pins(self):
		'''
		Initializes the input and output pins
		'''
		for i in self.sensors:
			gpio.setup(i, gpio.IN)

		for i in self.motors:
			gpio.setup(i, gpio.OUT)

		self.stop()


	def stop(self):
		'''
		Stops the bot. Output is low across all motors
		'''
		m1, m2, m3, m4 = self.motors
		gpio.output(m1, 0)
		gpio.output(m2, 0)
		gpio.output(m3, 0)
		gpio.output(m4, 0)


	def right(self, duration=None):
		'''
		Turns left for the specified duration and then stops
		'''
		if duration is None:
			duration = self.duration

		m1, m2, m3, m4 = self.motors
		gpio.output(m1, 0)
		gpio.output(m2, 1)
		gpio.output(m3, 0)
		gpio.output(m4, 0)
		time.sleep(duration)
		self.stop()

	def sharp_right(self, duration=None):
		'''
		Turns 90 degree right for the specified duration and then stops
		'''
		if duration is None:
			duration = self.duration

		m1, m2, m3, m4 = self.motors
		gpio.output(m1, 0)
		gpio.output(m2, 1)
		gpio.output(m3, 0)
		gpio.output(m4, 0)
		time.sleep(1)
		self.stop()

	def sharp_left(self, duration=None):
		'''
		Turns 90 degree left for the specified duration and then stops
		'''
		if duration is None:
			duration = self.duration

		m1, m2, m3, m4 = self.motors
		gpio.output(m1, 0)
		gpio.output(m2, 0)
		gpio.output(m3, 0)
		gpio.output(m4, 1)
		time.sleep(1)
		self.stop()


	def left(self, duration=None):
		'''
		Turns right for the specified duration and then stops
		'''
		if duration is None:
			duration = self.duration

		m1, m2, m3, m4 = self.motors
		gpio.output(m1, 0)
		gpio.output(m2, 0)
		gpio.output(m3, 0)
		gpio.output(m4, 1)
		time.sleep(duration)
		self.stop()


	def anti_left(self, duration=None):
		'''
		Reverses the left action.
		'''
		if duration is None:
			duration = self.duration

		m1, m2, m3, m4 = self.motors
		gpio.output(m1, 0)
		gpio.output(m2, 0)
		gpio.output(m3, 1)
		gpio.output(m4, 0)
		time.sleep(duration)
		self.stop()


	def anti_right(self, duration=None):
		'''
		Reverses the right action
		'''
		if duration is None:
			duration = self.duration

		m1, m2, m3, m4 = self.motors
		gpio.output(m1, 1)
		gpio.output(m2, 0)
		gpio.output(m3, 0)
		gpio.output(m4, 0)
		time.sleep(duration)
		self.stop()

	def sharp_anti_left(self, duration=None):
		'''
		Reverses the left action.
		'''
		if duration is None:
			duration = self.duration

		m1, m2, m3, m4 = self.motors
		gpio.output(m1, 0)
		gpio.output(m2, 0)
		gpio.output(m3, 1)
		gpio.output(m4, 0)
		time.sleep(1)
		self.stop()


	def sharp_anti_right(self, duration=None):
		'''
		Reverses the right action
		'''
		if duration is None:
			duration = self.duration

		m1, m2, m3, m4 = self.motors
		gpio.output(m1, 1)
		gpio.output(m2, 0)
		gpio.output(m3, 0)
		gpio.output(m4, 0)
		time.sleep(1)
		self.stop()


	def forward(self, duration=None):
		'''
		Moves forward for the specified duration and then stops
		'''
		if duration is None:
			duration = self.duration

		m1, m2, m3, m4 = self.motors
		gpio.output(m1, 0)
		gpio.output(m2, 1)
		gpio.output(m3, 0)
		gpio.output(m4, 1)
		time.sleep(duration)
		self.stop()


	def backward(self, duration=None):
		'''
		Moves backward for the specifeid duration and then stops
		'''
		if duration is None:
			duration = self.duration

		m1, m2, m3, m4 = self.motors
		gpio.output(m1, 1)
		gpio.output(m2, 0)
		gpio.output(m3, 1)
		gpio.output(m4, 0)
		time.sleep(duration)
		self.stop()

	
	def take_action(self, index):
		'''
		Takes a specifc actions present at given index
		'''
		action = self.actions[index]
		action()
		self.stack.append(index)
		state, _ = self.get_state()
		reward = self.get_reward()
		done = all(_==0)

		return state, reward, done


	def reset(self):
		'''
		Resets the bot environment by reversing all the stack actions
		'''
		actions = self.stack[::-1]
		for i in actions:
			self.rev_actions[i]()
			time.sleep(0.2)

		self.stack = []

		return self.get_state()[0]


	def get_state(self):
		'''
		Returns the state of the 5 sensors
		'''
		state = []
		for i in self.sensors:
			state.append((gpio.input(i) + 1)%2)

		_bin = ''.join([str(i) for i in state])
		index = int(_bin, 2)

		return index, np.array(state)


	def get_reward(self):
		'''
		Returns the rewards for a specific action
		'''
		_, state = self.get_state()
		reward = np.dot(self.rewards_weights.T, state)

		return reward

if __name__ == "__main__":
	gpio.setmode(gpio.BCM)

	sensors = [5, 6, 13, 19, 26]
	motors = [4, 17, 27, 22]
	rewards_weights = [-3, 1, 4, 1, -3]
	
	env = BotEnvironment(sensors, motors, rewards_weights)

	##############
	## TRAINING ##
	##############

	action_size = 3
	state_size = 2 ** 5
	qtable = np.zeros((state_size, action_size))

	total_episodes  = 10		# Total episodes
	learning_rate = 0.8		   # Learning rate
	max_steps = 50				# Max steps per episode
	gamma = 0.95				  # Discounting rate

	# Exploration parameters
	epsilon = 1.0				 # Exploration rate
	max_epsilon = 1.0			 # Exploration probability at start
	min_epsilon = 0.01			# Minimum exploration probability 
	decay_rate = 0.01


	rewards = []

	# 2 For life or until learning is stopped
	for episode in range(total_episodes):
		state = env.reset()
		step = 0
		done = False
		total_rewards = 0
		
		for step in range(max_steps):
			exp_exp_tradeoff = random.uniform(0, 1)
			if exp_exp_tradeoff > epsilon:
				action = np.argmax(qtable[state,:])
			else:
				action = random.randint(0, action_size-1)

			print('[{}] Taking action {} at {}'.format(episode, action, state))
			new_state, reward, done = env.take_action(action)

			qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
			
			total_rewards += reward
			
			state = new_state
			
			if done == True: 
				break
			
		episode += 1
		# Reduce epsilon (because we need less and less exploration)
		epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
		rewards.append(total_rewards)

	print ("Score over time: " +  str(sum(rewards)/total_episodes))

	#############
	## Testing ##
	#############
	# qtable = np.loadtxt('qtable.txt')
	print('Qtable')
	print(qtable)
	print('Waiting before testing...')
	time.sleep(10)
	print('beginning testing....')
	env.stack = []
	env.reset()

	for episode in range(5):
		# state = env.reset()
		state = 14
		step = 0
		done = False
		print("****************************************************")
		print("EPISODE ", episode)

		for step in range(50):
			# Take the action (index) that have the maximum expected future reward given that state
			action = np.argmax(qtable[state,:])
			
			new_state, reward, done = env.take_action(action)
			
			if done:
				break
			state = new_state

		input()
	