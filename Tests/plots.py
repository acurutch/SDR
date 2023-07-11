import matplotlib as plt
import numpy as np

def plot_I_Q(baseband_symbols, baseband_signal_I_new, baseband_signal_Q_new, symbols_I, symbols_Q, known_preamble_symbols):
    ##################
    # Plot I symbols

    fig, axs = plt.subplots(2, 2)

    fig.suptitle("I Symbols")

    axs[0, 0].plot(np.real(baseband_symbols, 'r-'))
    axs[0, 0].set_title("Before Pulse Shaping")
    axs[0, 1].plot(baseband_signal_I_new, 'b-')
    axs[0, 1].plot(np.real(baseband_symbols), 'r-')
    axs[0, 1].set_title("After Pulse Shaping")
    axs[1, 0].plot(symbols_I)
    axs[1, 0].plot(np.real(baseband_symbols), 'r-')
    axs[1, 0].set_title("After Matched Filtering")

    axs[1,1].plot(symbols_I[0:len(known_preamble_symbols)],symbols_Q[0:len(known_preamble_symbols)], 'ro')
    axs[1,1].plot(symbols_I[len(known_preamble_symbols):],symbols_Q[len(known_preamble_symbols):], 'bo')
    axs[1,1].legend(['Preamble Symbols', 'Data Symbols'])
    axs[1,1].set_title("IQ Plot")

    major_ticks = np.arange(-4, 5, 2)
    axs[1,1].set_xticks(major_ticks)
    axs[1,1].set_yticks(major_ticks)

    axs[1,1].minorticks_on()

    minor_ticks = np.arange(-4, 5, 1)
    axs[1,1].set_xticks(minor_ticks, minor=True)
    axs[1,1].set_yticks(minor_ticks, minor=True)

    axs[1,1].grid(which='minor', linestyle=':', alpha=0.2)
    axs[1,1].grid(which='major', linestyle='-', alpha=0.4)

    axs[1,1].set_xlim([-4, 4])
    axs[1,1].set_ylim([-4, 4])

    fig.show()

    ##################
    # Plot Q symbols

    fig2, axs2 = plt.subplots(2, 2)

    fig2.suptitle("Q Symbols")

    axs2[0, 0].plot(np.imag(baseband_symbols), 'r-')
    axs2[0, 0].set_title("Before Pulse Shaping")
    axs2[0, 1].plot(baseband_signal_I_new, 'b-')
    axs2[0, 1].plot(np.imag(baseband_symbols), 'r-')
    axs2[0, 1].set_title("After Pulse Shaping")
    axs2[1, 0].plot(symbols_Q)
    axs2[1, 0].plot(np.imag(baseband_symbols), 'r-')
    axs2[1, 0].set_title("After Matched Filtering")

    axs2[1,1].plot(symbols_I[0:len(known_preamble_symbols)],symbols_Q[0:len(known_preamble_symbols)], 'ro')
    axs2[1,1].plot(symbols_I[len(known_preamble_symbols):],symbols_Q[len(known_preamble_symbols):], 'bo')
    axs2[1,1].legend(['Preamble Symbols', 'Data Symbols'])
    axs2[1,1].set_title("IQ Plot")

    major_ticks = np.arange(-4, 5, 2)
    axs2[1,1].set_xticks(major_ticks)
    axs2[1,1].set_yticks(major_ticks)

    axs2[1,1].minorticks_on()

    minor_ticks = np.arange(-4, 5, 1)
    axs2[1,1].set_xticks(minor_ticks, minor=True)
    axs2[1,1].set_yticks(minor_ticks, minor=True)

    axs2[1,1].grid(which='minor', linestyle=':', alpha=0.2)
    axs2[1,1].grid(which='major', linestyle='-', alpha=0.4)

    axs2[1,1].set_xlim([-4, 4])
    axs2[1,1].set_ylim([-4, 4])

    fig2.show()