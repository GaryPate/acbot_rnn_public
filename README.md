# acbot_rnn_public

Sample code from a much larger personal project using Tensorflow and Keras to perform predition on intra day time series to generate trading signals using deep neural nets.

acbot_control.py is the main orchestration file and instantiates a base class for training and testing of API data
acbot_rnn_module.py contains classes and functions for preparing features into tensors and generating Keras models
acbot_wrangle.py performs feature engineering and technical indicator creation from raw OHLC data 
