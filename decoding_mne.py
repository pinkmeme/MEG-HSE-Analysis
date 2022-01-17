import mne
import mneflow.optimize
import numpy as np

raw = mne.io.read_raw_fif("derivatives/megprep/tda_05/experiment_tda_05_tsss_mc.fif")
raw.plot()
ica = mne.preprocessing.ICA(n_components = 30, random_state = 0, max_iter = 200)
ica.fit(raw)
ica.plot_components(outlines="skirt")
bad_idx, scores = ica.find_bads_eog()
exclude = [0, 3, 6, 8, 13, 17, 18, 19]
ica2.exclude = exclude
raw = ica2.apply(raw.copy(), exclude = ica2.exclude)
events = mne.find_events(raw)
mne.viz.plot_events(events)
epochs = mne.Epochs(raw, events)

optimizer_params = dict(l1_lambda=2e-3,learn_rate=1e-3)
optimizer = mneflow.optimize.Optimizer(**optimizer_params)
lf_params = dict(n_ls= 64, #number of latent factors
              filter_length=17, #convolutional filter length in time samples
              pooling = 5, #pooling factor
              stride = 5, #stride parameter for pooling layer
              padding = 'SAME',
              dropout = .7,
              model_path = import_opt['savepath']) #path for storing the saved model
dataset = mneflow.Dataset(meta, train_batch = 200)
model = mneflow.models.LFCNN(dataset, optimizer)