# -*- coding: utf-8 -*-
"""

"""

import os
import glob
import mne
import numpy as np
from mne.preprocessing import ICA, read_ica
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
import matplotlib.pyplot as plt
import pandas as pd

def restore_events(mx_filtered, sb):

        # events

        events = mne.find_events(mx_filtered, min_duration = 0.003, stim_channel='STI101')
        #event_dict = {'search': 1, 'memorization': 2, 's_plus_m': 3, 'm_plus_s': 4,  'interr_cue': 9, 'instruction' : 11, 'fix_cross': 12} #'pre-trial_alpha': 10,
        #mx_filtered.plot(events=events)
        #fig = mne.viz.plot_events(events, sfreq=mx_filtered.info['sfreq'],first_samp=mx_filtered.first_samp, event_id=event_dict)
        #fig.subplots_adjust(right=0.7)

        st = np.where(events[:, 2] == 11)[0]
        intervals = events[(st+1), 0] - events[st, 0]
        print('markers delay range in samples: ', np.max(intervals) - np.min(intervals))
        time_interval = int(np.round(np.mean(intervals)))

        # restore events
        for i in np.where(events[:, 2] == 12)[0]:
                events = np.append(events, [[(events[i, 0] - 2 * time_interval), 0, 10]], axis=0)
                if events[(i-1), 2] == 11:
                        continue
                else:
                        events = np.append(events, [[events[i, 0] - time_interval, 0, 11]], axis=0)

        #event_dict = {'search': 1, 'memorization': 2, 's_plus_m': 3, 'm_plus_s': 4,  'interr_cue': 9, 'instruction' : 11, 'fix_cross': 12, 'pre-trial_alpha': 10}
        
        #fig = mne.viz.plot_events(events, sfreq=mx_filtered.info['sfreq'],first_samp=mx_filtered.first_samp, event_id=event_dict)
        #mx_filtered.plot(events=events)

        # sorting events by time column
        events = events[events[:, 0].argsort()]
        print('ev shape ', events.shape)
        
        # correct event markers according to the eye movements data :/
        
        events_task = mne.pick_events(events, include=[1, 2, 3, 4])
        events_else = mne.pick_events(events, include=[9, 10, 11, 12])
        # delta = 144 - len(events_task)
        # #sb, trials, event
        # sb_df = pd.DataFrame({'TASK':events})
        # sb_df['SUBJECT'] = int(sb)
        # sb_df['TRIAL'] = np.arange(1 + delta,145)
        
        eyes = pd.read_csv('/m/nbe/project/topdown/derivatives/eyetracking/eyemovements_script_all_3_pupil.csv', usecols = ['SUBJECT', 'TRIAL', 'TASK'])
        #eyes = eyes.loc[:,['SUBJECT', 'TRIAL', 'TASK']]
        eyes.drop_duplicates(keep='first',inplace=True) 
        eyes = eyes.query('SUBJECT == @sb')
        delta = 144 - events_task.shape[0]
        events_task[:, 2] = eyes['TASK'][delta:]
        
        del events
        events = np.append(events_task, events_else, axis=0)
        events = events[events[:, 0].argsort()]
        
        print('ev shape ', events.shape)
        # create markers to compare mixed and pure blocks
        
        # task_events = mne.pick_events(events, include=[1, 2, 3, 4])
        # n_trials = len(task_events)
        # new_markers = np.full((n_trials, 1), np.nan)

        # for row in np.arange(n_trials, 0, -6):
        #         if row <= 6:
        #                 new_markers[0:row] = 0
        #         else:
        #                 if (1 in task_events[row-6:row, 2]) and (2 in task_events[row-6:row, 2]):
        #                         new_markers[row-6:row] = np.array([30, 30, 30, 30, 30, 31]).reshape(6, 1)  # mixed block
        #                 else:
        #                         new_markers[row-6:row] = np.array([20, 20, 20, 20, 20, 21]).reshape(6, 1)  # pure block

        # new_markers = new_markers.astype(int)
        # task_events[:, 2] = new_markers.flat
        # block_events = mne.pick_events(events, include=[10, 11, 12])
        # block_events = np.append(block_events, task_events, axis=0)
        # block_events = block_events[block_events[:, 0].argsort()]
        return events #, block_events

def get_ica_projections(mx_filtered):
        # According to MNE tutorial: 1Hz high pass is often helpful for fitting ICA. Also let's do it on 10 min part of the recording 
        hpfilt_croped = mx_filtered.copy().crop(tmin = 150, tmax = 750)
        hpfilt_croped.load_data().filter(l_freq=1., h_freq=None)
        #hpfilt_croped.info['bads'].append('ECG063') 
        
        ecg_epochs = mne.preprocessing.create_ecg_epochs(hpfilt_croped)
        avg_ecg_epochs = ecg_epochs.average()
        #avg_ecg_epochs.plot_topomap(times=np.linspace(-0.05, 0.05, 11)) 
        
        eog_epochs = mne.preprocessing.create_eog_epochs(hpfilt_croped, ch_name = 'MEG0121')
        avg_eog_epochs = eog_epochs.average()
        #eog_epochs.average().plot_topomap(times=np.linspace(-0.1, 0.1, 9))

        method = 'fastica' 
        decim = 3  # we need sufficient statistics, not all time points -> saves time
        ica = ICA(n_components=0.99, method=method)
        #print(ica)
        reject = dict(mag=5e-12, grad=4000e-13)
        ica.fit(hpfilt_croped, decim=decim, reject=reject)
        
        ica.exclude = []
        # find which ICs match the ECG and EOG pattern
        ecg_indices, ecg_scores = ica.find_bads_ecg(hpfilt_croped, method='ctps') #ch_name='ECG063'
        eog_indices, eog_scores = ica.find_bads_eog(hpfilt_croped, ch_name = 'MEG0121')

        ica.exclude = ecg_indices + eog_indices
        
        
        # barplot of ICA component ECG (the first) and EOG (the second) match scores
        ica.plot_scores(ecg_scores, exclude=ecg_indices)
        ica.plot_scores(eog_scores, exclude=eog_indices)
        
        # plot diagnostics
        ica.plot_properties(hpfilt_croped, picks=ecg_indices)
        ica.plot_properties(hpfilt_croped, picks=eog_indices)
        
        # plot ICs applied to raw data, with ECG and EOG matches highlighted
        ica.plot_sources(hpfilt_croped)
        
        # plot ICs applied to the averaged ECG epochs, with ECG matches highlighted
        ica.plot_sources(avg_ecg_epochs)
        ica.plot_sources(avg_eog_epochs)
        
        #input("Press Enter to plot other components...")

        #ica.plot_properties(hpfilt_croped, picks = list(set(range(0, 20)) - set(ica.exclude)))
        ica.plot_properties(hpfilt_croped, picks = list(range(0,20)), reject =None)
        
        print('ica.exclude = ', ica.exclude, '\npress enter or type the new list')
        new_exclude = input()
        if len(new_exclude) == 0:
                pass
        else:
                new_exclude = new_exclude.split(",")
                new_exclude = [int(i) for i in new_exclude]
                ica.exclude = new_exclude
                
        # save ICA result                
        ica.save(file[:-4]+'-ica.fif')
        return ica


'''
path and conds
'''
rootdir_project = '/m/nbe/project/topdown'
path_preprocessed = '/m/nbe/project/topdown/derivatives/megprep'

path_maxfilter = '/m/nbe/project/topdown/neuromag/bin/util/x86_64-pc-linux-gnu/maxfilter-2.3'

os.chdir(path_preprocessed)
cond = 'experiment'
maxfilter_cond = 'tsss_mc_trans'
#%% ICA

'''
check raw files
'''

#for sb in range(5, 6):
#        subj_name = 'tda_' + str(sb).zfill(2)
#        #print(subj_name)
#        path_source = glob.glob(os.path.join(rootdir_project, 'MEG_raw_data', subj_name, '*'))[0]
#        for file in glob.glob(os.path.join(path_source, cond+'*')):
#                if not ('-1' in file):                    
#                        # read raw meg data
#                        raw = mne.io.read_raw_fif(file)
#                        #picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False)
#                        raw.plot(duration=10, group_by= 'selection', lowpass = 40)
#                        raw.close()

'''
find the files that maxfilter hasn't processed
if there are too many bad channels, than maxfilter doesn't produce output. find such cases
'''

### old
#files_to_check = []
#
#for sb in range(1, 33):
#        subj_name = 'tda_' + str(sb).zfill(2)
#        files_list =  glob.glob(os.path.join(path_preprocessed, subj_name, '*'))
#        for file in  (glob.glob(os.path.join(path_preprocessed, subj_name, '*.log'))):
#                if not (file[:-3] + 'fif' in files_list): 
#                        files_to_check.append(file)
#                        print(file)

#print(files_to_check)
#there are 3 files, that weren't processed (check sbj 02, 11, 32)



#sbj_to_check = []
#
#for sb in range(1, 33):
#        subj_name = 'tda_' + str(sb).zfill(2)
#        files_list =  glob.glob(os.path.join(path_preprocessed, subj_name, '*'))
#        for file in  (glob.glob(os.path.join(path_preprocessed, subj_name, '*.log'))):
#                if '_bad_ch' in file:
#                        continue
#                if not (file[:-3] + 'fif' in files_list): 
#                        sbj_to_check.append(sb)
#                        print(sb)
#                        break


'''
mne preprocessing
'''

#get ICA projections

for sb in range(25, 26): #sb_valid:
        subj_name = 'tda_' + str(sb).zfill(2)

        file_list =  glob.glob(os.path.join(path_preprocessed, subj_name, cond+'*'+maxfilter_cond+'*'+'fif'))
        ban_list = ['-1', 'ica', 'epo']
        file_list = [word for word in file_list if not any(bad in word for bad in ban_list)]
        
        file_list.sort(reverse=True)
        #print('subject: ', sb, '\nfiles: ', file_list)
        #in sb doesn't have MEG, take the next one 
        if len(file_list) == 0:
                continue
        
        if len(file_list) == 1:
                file = file_list[0] 
                
                # read raw meg data
                mx_filtered = mne.io.read_raw_fif(file, preload= True)
        
        else:
                file = file_list[0] 
                raws = [mne.io.read_raw_fif(f) for f in file_list]
                mx_filtered = mne.io.concatenate_raws(raws)
                map(mne.io.fiff.raw.Raw.close, raws) 
                del raws
                mx_filtered.load_data()
                
        # remove powerline noise
        # and lowpass
        #mx_filtered.filter(None, 40., fir_design='firwin')
        # or notch
        mx_filtered.notch_filter(np.arange(50, 101, 50), notch_widths= 2) 
        mx_filtered.filter(None, 70.)
        # take a look on psd
#        mx_filtered.plot_psd(tmax=30)
#        mx_filtered.plot_psd(fmin=2., fmax=100)

        
        # read the annotations of bad spans if it was already done
        annot_file = os.path.join(path_preprocessed, subj_name, os.path.basename(file)[:-4] +'_saved-annotations.csv')
        if os.path.exists(annot_file):
                annot_form_file = mne.read_annotations(annot_file)
                mx_filtered.set_annotations(annot_form_file)
                print('annotations added from the file')

        # remove noise before ICA
        fig = mx_filtered.plot(n_channels=34, highpass = 0.1, lowpass = 50, duration = 60)
        fig.canvas.key_press_event('a')
        mx_filtered.annotations.save(annot_file)

        # ICA
        ica = get_ica_projections(mx_filtered)
        mx_filtered.close()

#%%
# subjects 11, 13, 23 has more parts of the data
# for ICA the data was filtered 0,1 and cropped 600-1200, if no otherwise specified
# if there was no EOG, MEG0121 was used
ica_exclude = {}
ica_exclude['tda_05'] = [0, 9, 11, 42]
ica_exclude['tda_06'] = [0, 2, 6] #33?

ica_exclude['tda_07'] = [0, 1, 22] #no ECG and EOG channels # it's messy, but there is no flat parts, so I didn't reject anything
ica_exclude['tda_08'] = [2, 8, 18] #no ECG and EOG channels
ica_exclude['tda_09'] = [0, 15, 16, 12] #no ECG and EOG channels #on the interval 1520-2100s
ica_exclude['tda_11'] = [0, 5] #no ECG and EOG channels; on 1700-2300s #not clear, what was a heart component there
ica_exclude['tda_12'] = [0, 35, 22] #no ECG and EOG channels; on 250-850

ica_exclude['tda_13'] = [23, 29, 0, 14, 3] 
ica_exclude['tda_14'] = [0, 6, 21]
ica_exclude['tda_15'] = [0, 7, 18, 23] #on 1250-1850s
ica_exclude['tda_16'] = [0, 1, 23] #on 1240-1840s 

ica_exclude['tda_17'] = [10, 3, 26]  #on 310-890s
ica_exclude['tda_18'] = [32, 0, 11, 32] #1210-1790 
ica_exclude['tda_19'] = [3, 35, 38]#900-1500
ica_exclude['tda_20'] = [7, 23, 0] #on 900-1500s

ica_exclude['tda_22'] = [0, 20, 10]  # no ECG and EOG channels #on 200-800 
ica_exclude['tda_23'] = [1, 5, 18] #no ECG and EOG channels #on 600-1200s
ica_exclude['tda_24'] = [0, 3, 4] #no ECG and EOG channels 

ica_exclude['tda_25'] = [12, 0, 9] #on 150-750 #flat instead of one EOG pair
ica_exclude['tda_26'] = [0, 5, 13] #on 1400-2000
ica_exclude['tda_27'] = [0, 45, 24] #no ECG and EOG channels 
ica_exclude['tda_28'] = [0, 15, 1, 13] #no ECG and EOG channels #on 650-1250
ica_exclude['tda_29'] = [3, 4, 11, 35] #no ECG and EOG channels #on 650-1250
ica_exclude['tda_30'] = [0, 13, 16] #there is ECG channel here, but it's flat. and EOG epoch avg looks bad #650-1250s
ica_exclude['tda_31'] = [8, 19, 0, 10, 2] #600-1200s
ica_exclude['tda_32'] = [6, 24, 0, 11, 28]#350-1200 noise instead of eog epochs 


# apply ICA + get epochs

reject_criteria = dict(mag=4000e-15,     # 3000 fT
                       grad=4000e-13)    # 3000 fT/cm
                       #eog=200e-6)       # 200 μV

flat_criteria = dict(mag=1e-15,          # 1 fT
                     grad=1e-13)         # 1 fT/cm

#%%

for sb in range(5, 6): #ica_exclude:
        subj_name = 'tda_' + str(sb).zfill(2)
        print(subj_name)
        file_list =  glob.glob(os.path.join(path_preprocessed, subj_name, cond+'*'+maxfilter_cond+'*'+'fif'))
        ban_list = ['-1', 'ica', 'epo']
        file_list = [word for word in file_list if not any(bad in word for bad in ban_list)]
        
        file_list.sort(reverse=True)
        #print('subject: ', sb, '\nfiles: ', file_list)
        #in sb doesn't have MEG, take the next one 
        if len(file_list) == 0:
                continue
        
        if len(file_list) == 1:
                file = file_list[0] 
                
                # read raw meg data
                mx_filtered = mne.io.read_raw_fif(file, preload= True, verbose='CRITICAL')
        
        else:
                file = file_list[0] 
                raws = [mne.io.read_raw_fif(f, verbose='CRITICAL') for f in file_list]
                mx_filtered = mne.io.concatenate_raws(raws)
                map(mne.io.fiff.raw.Raw.close, raws) 
                del raws
                mx_filtered.load_data()
        
        # remove powerline noise
        # or notch
        mx_filtered.notch_filter(np.arange(50, 101, 50), notch_widths= 2)
        mx_filtered.filter(None, 70.)
                
        # get events
        events = restore_events(mx_filtered, sb)
        
        #get ica projections
        ica = read_ica(glob.glob(os.path.join(path_preprocessed, subj_name, '*ica*'))[0])
        if len(glob.glob(os.path.join(path_preprocessed, subj_name, '*ica'))) > 1:
                print('Someth went wrong! Found more then 1 file with ICA projections')
                                                      
        # apply ICA to the initial file               
        ica.apply(mx_filtered)



#       mx_filtered.plot(group_by='selection')
#                
#       plt.close('all')
        
        # Epochs

#        epochs_prestim = mne.Epochs(mx_filtered, mne.pick_events(events, include=10), baseline = None, tmin=0, tmax=.95, reject=reject_criteria, flat=flat_criteria)
#        #epochs_prestim.plot()
#        epochs_prestim.drop_bad()
#        #epochs_prestim.plot_drop_log()
#
#        print('bad epochs dropped (prestim) %.2f %%' % epochs_prestim.drop_log_stats())
#        
#        #better to concatinate for the classifer (?)
#        epochs_pure = mne.Epochs(mx_filtered, mne.pick_events(block_events, include=21), tmin=-0.2, tmax=4.5, reject=reject_criteria, flat=flat_criteria)
#        epochs_mixed = mne.Epochs(mx_filtered, mne.pick_events(block_events, include=31), tmin=-0.2, tmax=4.5, reject=reject_criteria, flat=flat_criteria)
#        
#        epochs_pure.drop_bad()
#        epochs_mixed.drop_bad()
#        print('bad epochs dropped (pure block) %.2f %%' % epochs_pure.drop_log_stats())
#        print('bad epochs dropped (mixed block) %.2f %%' % epochs_mixed.drop_log_stats())
#        
#        #epochs_prestim.plot_image(None, cmap='interactive', sigma=1., vmin=-250, vmax=250)
#        
#        epochs_fname = os.path.join(path_preprocessed, subj_name, os.path.basename(file)[:-4] +'-prestim_epo.fif')
#        epochs_prestim.save(epochs_fname, overwrite=True)
#        
#        epochs_fname = os.path.join(path_preprocessed, subj_name, os.path.basename(file)[:-4] +'-pure_bk_epo.fif')
#        epochs_pure.save(epochs_fname, overwrite=True)
#        
#        epochs_fname = os.path.join(path_preprocessed, subj_name, os.path.basename(file)[:-4] +'-mixed_bk_epo.fif')
#        epochs_mixed.save(epochs_fname, overwrite=True)
#       
        # remove when filtering and ICA re-did
        #mx_filtered.filter(None, 70.)#, fir_design='firwin'
        #mx_filtered.plot_psd()
        
        ###task epochs ###       
        # merged_events = mne.merge_events(events, [1, 3], 1)
        # merged_events = mne.merge_events(merged_events, [2, 4], 2)

        # epochs_task = mne.Epochs(mx_filtered, mne.pick_events(merged_events, include=[1, 2]), tmin=-0.5, tmax=4.5, reject=reject_criteria, flat=flat_criteria)
        # #epochs_task.plot()
        # epochs_task.drop_bad()
        # fig = epochs_task.plot_drop_log()
        # fig.canvas.set_window_title(subj_name)
        # plt.savefig('/m/nbe/project/topdown/plots/epoch_drop_logs/task/'+subj_name+'_drop_log.png')
        # epochs_task.plot()
        # print('bad epochs dropped (task) %.2f %%' % epochs_task.drop_log_stats())
        # destination_folder = '/m/nbe/project/topdown/derivatives/megepochs/task/'
        # epochs_fname = os.path.join(destination_folder, os.path.basename(file)[:-4] +'-task_epo.fif')
        # epochs_task.save(epochs_fname, overwrite=True)
        
        ### cue epochs, 88 = continue, 89 = switch ###
#        
#        for i in np.where(events[:, 2] == 1)[0]:
#                events[i+1,2] = 88
#        for i in np.where(events[:, 2] == 2)[0]:
#                events[i+1,2] = 88
#
#        for i in np.where(events[:, 2] == 3)[0]:
#                events[i+1,2] = 89
#        for i in np.where(events[:, 2] == 4)[0]:
#                events[i+1,2] = 89
#                
#                
#        epochs_cue = mne.Epochs(mx_filtered, mne.pick_events(events, include=[88,89]), tmin=-0.5, tmax=1, reject=reject_criteria, flat=flat_criteria)
#        #epochs_task.plot()
#        epochs_cue.drop_bad()
#        fig = epochs_cue.plot_drop_log()
#        fig.canvas.set_window_title(subj_name)
#        plt.savefig('/m/nbe/project/topdown/plots/epoch_drop_logs/cue_epo/'+subj_name+'_drop_log.png')
#        epochs_cue.plot()
#        print('bad epochs dropped (task) %.2f %%' % epochs_cue.drop_log_stats())
#        destination_folder = '/m/nbe/project/topdown/derivatives/megepochs/cue/'
#        epochs_fname = os.path.join(destination_folder, os.path.basename(file)[:-4] +'-cue_epo.fif')
#        epochs_cue.save(epochs_fname, overwrite=True)
        
        ###dummy: pretrial vs trial epochs ###       
        # merged_events = mne.merge_events(events, [1,2,3,4], 20)
        # merged_events[merged_events[:, 2] == 10, 0] += int(0.3*mx_filtered.info['sfreq'])
        # epoch_dummy = mne.Epochs(mx_filtered, mne.pick_events(merged_events, include=[10, 20]), tmin=0, tmax=0.7, reject=reject_criteria, flat=flat_criteria, baseline = None)

        # epoch_dummy.drop_bad()
        
        ##task second half ###

        events[np.where(events[:, 2] == 1)[0] + 1, 2] = 15 #ended with search
        events[np.where(events[:, 2] == 4)[0] + 1, 2] = 15
        events[np.where(events[:, 2] == 2)[0] + 1, 2] = 16 #ended with memo
        events[np.where(events[:, 2] == 3)[0] + 1, 2] = 16
        #print(events[:, 2])

        
        epochs_task_sh = mne.Epochs(mx_filtered, mne.pick_events(events, include=[15, 16]), tmin=-0.2, tmax=5, reject=reject_criteria, flat=flat_criteria)
        #epochs_task.plot()
        epochs_task_sh.drop_bad()
        print('bad epochs dropped (task) %.2f %%' % epochs_task_sh.drop_log_stats())
        destination_folder = '/m/nbe/project/topdown/derivatives/megepochs/task_second_half/'
        epochs_fname = os.path.join(destination_folder, os.path.basename(file)[:-4] +'-task_epo.fif')
        epochs_task_sh.save(epochs_fname, overwrite=True)

        mx_filtered.close()
        del mx_filtered, epochs_task_sh



#np.where([len(log) == 0 for log in epochs_task.drop_log])[0] 
#epochs_task.selection
#
#len(epochs_task.selection)

# epochs_task_rj = mne.Epochs(mx_filtered, mne.pick_events(events, include=[1, 2, 3, 4]), tmin=-0.2, tmax=4.5)
# epochs_task = mne.Epochs(mx_filtered, mne.pick_events(events, include=[1, 2, 3, 4]), tmin=-0.2, tmax=4.5, reject=reject_criteria, flat=flat_criteria)
# epochs_task.drop_bad()
# epochs_task_rj.drop(epochs_task.selection)

# epochs_task.plot()
# epochs_task_rj.plot()