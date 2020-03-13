import preprocess
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
plt.figure(figsize=(8, 4))

column_name = { 3 : "AF3",
                4 : "F7",
                5 : "F3",
                6 : "FC5",
                7 : "T7",
                8 : "P7",
                9 : "O1",
               10 : "O2",
               11 : "P8",
               12 : "T8",
               13 : "FC6",
               14 : "F4",
               15 : "F8",
               16 : "AF4"}

if __name__ == "__main__":
    data = preprocess.get_processed_data('eeg', 1, 12)

    # Time stamp is column 20
    x = data[20]
    x = x - x[0] # Start time from zero

    # EEG channels are 3-16
    for channel in range(3, 17):
        y = data[channel]

        plt.plot(x, y)
        plt.xlabel('Time ($s$)')
        plt.ylabel(column_name[channel] + " " + 'Volatge ($\mu V$)')
        plt.savefig(column_name[channel] + ".png", bbox_inches='tight')
        plt.clf()

