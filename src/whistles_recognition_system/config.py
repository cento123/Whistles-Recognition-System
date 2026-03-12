
# %% NOT PUBLIC DATA:
NFFT = 2048
fs = 195312.5/2 # [Hz]
overlap_bin = 0.5

Tbin = NFFT*(1-overlap_bin)/fs
Fbin = fs/NFFT
# %% Parameters used in the spectrogram creation (image files):
Fpx = Fbin      # Frequency resolution of the spectrogram [Hz]
Tpx = Tbin      # Time resolution of the spectrogram [s]
Fpx_0 = 1955    # Frequency of the bottom pixel in the spectrogram [Hz]
Npxs = 608      # Total number of pixels in spectrogram height