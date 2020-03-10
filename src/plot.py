import reader
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
plt.figure(figsize=(8, 4))

column_name = { 4 : "AF3",
                5 : "F7",
                6 : "F3",
                7 : "FC5",
                8 : "T7",
                9 : "P7",
               10 : "O1",
               11 : "O2",
               12 : "P8",
               13 : "T8",
               14 : "FC6",
               15 : "F4",
               16 : "F8",
               17 : "AF4"}

if __name__ == "__main__":
    data = reader.get_matlab_data('eeg', 1, 12)

    # Time stamp is column 20
    x = data[20]
    x = x - x[0] # Start time from zero

    # EEG channels are 4-17
    for channel in range(4, 18):
        y = data[channel]

        plt.plot(x, y)
        plt.xlabel('Time ($s$)')
        plt.ylabel(column_name[channel] + " " + 'Volatge ($\mu V$)')
        plt.savefig(column_name[channel] + ".png", bbox_inches='tight')
        plt.clf()

