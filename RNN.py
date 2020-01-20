import numpy as np
import os 
import logging
import time

from keras.models import Sequential, load_model
from keras.layers import SimpleRNN
from keras.optimizers import Adam
from tqdm import tqdm

def load_one(path):	
	with open(path) as f:
		x = f.readlines()

	game = []
	for move in x:
		m = move.split(" ")
		line = []
		for i in m:
			if i == "\n":
				break
			line.append(float(i))
		game.append(line)

	return np.array(game)

def determine_winner(player1, player2): 
	#returns 0 if player1 wins, 1 if player2 wins, 0.5 if draw
	player1 = player1[-1]
	player2 = player2[-1]

	if (player1[0]+player1[7] == 72):
		if player1[0]<player1[7]:
			return 1
		elif player1[0]>player1[7]:
			return 0
		else:
			return 0.5
	else:
		if player2[0]<player2[7]:
			return 0
		elif player2[0]>player2[7]:
			return 1
		else:
			return 0.5


def datagen():
	path = "data/"
	filenames = os.listdir(path)

	for i in range(len(filenames)//2):
		p1path = path + "player1_" + str(i+1) + ".txt"
		p2path = path + "player2_" + str(i+1) + ".txt"
		player1 = load_one(p1path)
		player2 = load_one(p2path)
		
		winner = determine_winner(player1, player2)

		yield (player1.reshape(1, -1, player1.shape[1]), np.array([1-winner])), (player2.reshape(1, -1, player2.shape[1]), np.array([winner]))


def define_model():
	model = Sequential()
	model.add(SimpleRNN(units=1, activation='sigmoid', input_shape=(None, 14)))
	optim = Adam()
	model.compile(loss="binary_crossentropy", optimizer=optim)
	#model.summary()
	return model


if __name__ == "__main__":
	start_time = time.time()

	logging.disable(logging.WARNING)
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

	epochs = 25
	data = datagen()
	
	#model = define_model()

	model = load_model("models/AlphaPace.h5")
	print("model loaded")

	for (x1,y1),(x2,y2) in tqdm(data):
		model.fit(x1, y1, epochs=epochs, verbose=0)
		model.fit(x2, y2, epochs=epochs, verbose=0)
	
	model.save("models/AlphaPace.h5")
	print(round((time.time() - start_time),2), "s")
	
	print("allgood")
