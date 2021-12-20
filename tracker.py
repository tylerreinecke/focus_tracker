import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import sys
from brainflow.data_filter import DataFilter 
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

LENGTH_EPOCH = 1024

def get_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, help='serial port', required=True, default='')
    args = parser.parse_args()
    return args

def setup_stream(args):
    # Device Parameters
    parameters = BrainFlowInputParams()
    parameters.serial_port = args.port
    # Muse 2 device Object
    device = BoardShim(BoardIds.MUSE_2_BOARD, parameters)
    device.prepare_session()
    device.start_stream(LENGTH_EPOCH * 4)
    return device

def beta_pipe(device : BoardShim):
    id = device.get_board_id()
    # Sampling rate for frequency calculations
    sampling_rate = BoardShim.get_sampling_rate(id)
    # Index of eeg channels
    eeg_channels = BoardShim.get_eeg_channels(id)
    # Functional data pipe that is run by main program
    def pipe(data):
        # Applies bandpass filter and returns avg band powers and std deviations
        bands = DataFilter.get_avg_band_powers(data, eeg_channels, sampling_rate, True)
        return bands[0][1]
    return pipe

def poll(device, pipe):
	# Poll data every second
    time.sleep(1)
    data = device.get_current_board_data(LENGTH_EPOCH)
    # Run pipeline on data
    metric = pipe(data)
    return metric

def end_stream(device):
    device.stop_stream()
    device.release_session()

def setup_plot():
	fig = plt.figure()
	ax = fig.add_subplot(1, 1,1)
	plt.ion()
	plt.title('Focus Score over Time')
	plt.ylabel('Focus Score')
	fig.show()
	fig.canvas.draw()
	return fig, ax

def update_plot(fig, ax, second, timeline, reading, scores):
	timeline.append(second)
	scores.append(reading)
	# Limit x and y lists to most recent 30 items
	timeline = timeline[-30:]
	scores = scores[-30:]
	ax.clear()
	ax.plot(timeline, scores, color='red')
	fig.canvas.draw()

if __name__ == '__main__':
	BoardShim.enable_dev_board_logger()
	device = setup_stream(get_args())
	pipe = beta_pipe(device)
	fig, ax = setup_plot()
	second = 1
	timeline = []
	scores = []
	try: 
		while True:
			# Get the piped data
			metric = poll(device, pipe)
			# Plot
			update_plot(fig, ax, second, timeline, metric, scores)
			second += 1
			print('Focus: ', metric)
	except KeyboardInterrupt:
		print('Interrupted')
	finally:
		end_stream(device)
		print('Exited')
		sys.exit()