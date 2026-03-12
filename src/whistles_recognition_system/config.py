
# %% NOT PUBLIC DATA:
NFFT = 2048
fs = 195312.5/2 # [Hz]
overlap_bin = 0.5
Tpx = NFFT*(1-overlap_bin)/fs
Fpx = fs/NFFT
Fpx_0 = 1955    # Frequency of the bottom pixel in the spectrogram [Hz]
Npxs = 608      # Total number of pixels in spectrogram height

# %% Parameters used in the spectrogram creation (image files):
Fpx = 62.5      # Frequency resolution of the spectrogram [Hz]
Tpx = 8e-3      # Time resolution of the spectrogram [s]
Fpx_0 = 2e3    # Frequency of the bottom pixel in the spectrogram [Hz]
Npxs = 448      # Total number of pixels in spectrogram height
