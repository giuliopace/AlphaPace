import numpy as np
import os 
#from keras.models import Sequential
#from keras.layers import SimpleRNN


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

		yield (player1, np.array([1-winner])), (player2, np.array([winner]))




data = datagen()

print("\n\n", next(data))


#preprocess data

#model = Sequential()
#model.add(SimpleRNN(units=1, activation='sigmoid'))


