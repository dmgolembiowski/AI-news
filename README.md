# AI-news
Senior A.I. project to generate realistic news articles like those found on CNN, NYTimes, Fox News, etc. Future research will involve conducting surveys to evaluate people's ability to detect the difference between real news and dopplegangers.

	Long Short Term Memory (LSTM), a specialized recurrent neural network (RNN), was invented in response to a problem in machine-learning where “... it took a very long time… [to learn] to store information... via backpropagation… due to insufficient, decaying error backflow” (Hochreiter, Schmidhuber 1991). As its name implies, LSTM derives features from both Long-Term Memory (LTM) and Short-Term Memory (STM). 
LTM implementations use weighted gates to manage the error flow from a state cell’s input to every other state cell, as well as to the output layer. Before learned error signals make it through the input gate at a particular state cell, a gate outside the cell decides if the incoming error signal should be allowed to override the current contents or if the current values should be kept intact. Some configurations can even determine whether the error signal should be allowed to dampen or amplify other state cells. However, the drawback of using this model stems from its inability to remain teachable due to the exponentially-decaying error.
	On the other hand, STM implementations—also called the naive (greedy) approaches of Long Term Memory—suffer from bifurcating error signals because the sigmoid function causes oscillating weights and unstable learning.
LSTM expands on its predecessor by defining a high-level architecture that has an input layer, a hidden layer of Memory Cells Blocks (MCB)—which contain individual Memory Cells (MC) and self-associated Constant-Error Carousels (CEC), and finally an output layer. Each MCB uses many multiplicative non-output gates that allows the topology to have constant error flow through some number of memory cells in the same block, their own associated self-to-self connections (via their CEC), and signal transmission to other neighboring MCBs without manifesting the problems in the naive approach. LSTM needs this because every time new information makes it into a memory cell, it has to ask itself ‘should this new information be added to our memory?’ If yes, then its corresponding CEC has to remember a numerical value related to the logistic sigmoid activation function. Otherwise, that memory cell’s contents will not be protected from irrelevant inputs. Additionally, every memory cell has an input gate governing how much an incoming external signal will amplify or dampen its own contents, as well as the opposite relationship resulting from outgoing signals.
