# Import required general libraries
#
import matplotlib.pyplot as plt # library for plotting
import matplotlib
matplotlib.use('Qt5Agg')
# matplotlib.pyplot.ion()
import glob  # File pattern matching
import mne # library for EEG data analysis
import numpy as np # library for manging numerical data 
from PyQt5.QtWidgets import QFileDialog # library for creating dialogue windows
import plotly.graph_objs as go # library for graphic objects
import scipy 
import seaborn as sns
import re
import pandas as pd
from mne.channels import make_standard_montage
from mne_icalabel import label_components
import os
from pyprep.prep_pipeline import PrepPipeline
from pyprep.find_noisy_channels import NoisyChannels
from datetime import datetime
from scipy.spatial.distance import euclidean
import os
import sys
import glob
import io
import json
import re
import pickle
import random
import warnings
from datetime import datetime
from pathlib import Path
import time
import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import interp1d
from scipy.signal import resample
from scipy.spatial.distance import euclidean
from scipy.stats import (
    expon, gamma, laplace, linregress, norm, poisson, rayleigh, t, uniform
)
from statsmodels.tsa.arima.model import ARIMA
import matplotlib
matplotlib.use('Qt5Agg')  # GUI backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from tabulate import tabulate
from tqdm import tqdm
from tqdm.notebook import tqdm as notebook_tqdm
import plotly.graph_objs as go
import tkinter as tk
from tkinter import simpledialog
from PyQt5.QtWidgets import QFileDialog
import itertools
import mne
from mne import create_info, EvokedArray
from mne.channels import make_standard_montage
from mne.io import RawArray
import mne_connectivity
from mne_icalabel import label_components
from pyprep.find_noisy_channels import NoisyChannels
from pyprep.prep_pipeline import PrepPipeline
from fooof import FOOOF
from fooof.sim.gen import gen_power_spectrum
from fooof.sim.utils import set_random_seed
from fooof.plts.spectra import plot_spectra
from fooof.plts.annotate import plot_annotated_model
import tmspath_utils_adj


def import_modules():
    # ===============================
    # Import standard libraries
    # ===============================
    import os
    import sys
    import glob
    import io
    import json
    import re
    import pickle
    import random
    import warnings
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    from datetime import datetime
    from pathlib import Path
    import time
    
    # ===============================
    # Import scientific computing libraries
    # ===============================
    import numpy as np
    import pandas as pd
    import scipy
    from scipy.interpolate import interp1d
    from scipy.signal import resample
    from scipy.spatial.distance import euclidean
    from scipy.stats import (
        expon, gamma, laplace, linregress, norm, poisson, rayleigh, t, uniform
    )
    
    # ===============================
    # Import time series analysis
    # ===============================
    from statsmodels.tsa.arima.model import ARIMA
    
    # ===============================
    # Import data visualization libraries
    # ===============================
    import matplotlib
    matplotlib.use('Qt5Agg')  # GUI backend
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import seaborn as sns
    from tabulate import tabulate
    from tqdm import tqdm
    from tqdm.notebook import tqdm as notebook_tqdm
    import plotly.graph_objs as go
    
    # ===============================
    # GUI interaction (Tkinter + PyQt5)
    # ===============================
    import tkinter as tk
    from tkinter import simpledialog
    from PyQt5.QtWidgets import QFileDialog
    import itertools
    
    # ===============================
    # EEG data analysis (MNE + connectivity)
    # ===============================
    import mne
    from mne import create_info, EvokedArray
    from mne.channels import make_standard_montage
    from mne.io import RawArray
    import mne_connectivity
    from mne_icalabel import label_components
    
    # ===============================
    # EEG preprocessing tools
    # ===============================
    from pyprep.find_noisy_channels import NoisyChannels
    from pyprep.prep_pipeline import PrepPipeline
    
    # ===============================
    # FOOOF (spectral fitting)
    # ===============================
    from fooof import FOOOF
    from fooof.sim.gen import gen_power_spectrum
    from fooof.sim.utils import set_random_seed
    from fooof.plts.spectra import plot_spectra
    from fooof.plts.annotate import plot_annotated_model
    
    # ===============================
    # User-defined functions
    # ===============================
    import tmspath_utils as tmsu
    
    # ===============================
    # Init: date, timer, json
    # ===============================
    now = datetime.now()
    date = now.strftime("%Y%m%d%H%M%S")
    start_time = time.time()  # ⏱️ start timer
    
    return date, start_time

def make_json_serializable(d):
    new_d = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            new_d[k] = v
        elif isinstance(v, (list, tuple)):
            new_d[k] = [make_json_serializable({'v': x})['v'] if isinstance(x, dict) else str(x) for x in v]
        else:
            # Converti tutto il resto in stringa
            new_d[k] = str(v)
    return new_d

def directorySetup(json_data):
    import os
    import json
    from pathlib import Path

    sub=json_data["subject"]
    date=json_data["date"]

    if "experiment_dir" in json_data and json_data["experiment_dir"]:
        experiment_dir=json_data["experiment_dir"]
        print(f"📌 Using provided experiment_dir: {experiment_dir}")
    else:
        experiment_dir=os.path.join(json_data["mainDir"],f"{date}_{sub}")
        print(f"📁 Generated experiment_dir: {experiment_dir}")

    subdirs=[
        "1.basic",
        "2.trials",
        "3.detrend",
        os.path.join("3.detrend","examples"),
        "4.postICA",
        "5.Extra",
        os.path.join("5.Extra","FE"),
        "6.FOOOF",
        "7.pkls",
    ]

    os.makedirs(experiment_dir,exist_ok=True)

    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir,subdir),exist_ok=True)

    json_data["experiment_dir"]=experiment_dir
    json_data["rest_directory_structure"]=subdirs

    with open(Path(experiment_dir)/f"{sub}_pars.json","w") as json_file:
        json.dump(make_json_serializable(json_data),json_file,indent=4,sort_keys=True)

    return json_data,experiment_dir,sub

"""
def directorySetup(json_data):
    import os
    import json
    from pathlib import Path

    sub=json_data["subject"]
    date=json_data["date"]

    if "experiment_dir" in json_data and json_data["experiment_dir"]:
        experiment_dir=json_data["experiment_dir"]
        print(f"📌 Using provided experiment_dir: {experiment_dir}")
    else:
        experiment_dir=os.path.join(json_data["mainDir"],f"{date}_{sub}")
        print(f"📁 Generated experiment_dir: {experiment_dir}")

    subdirs=[
        "1.basic",
        "2.trials",
        "3.postICA",
        "4.Extra",
        os.path.join("4.Extra","FE"),
        "5.FOOOF",
        "7.pkls",

    ]

    os.makedirs(experiment_dir,exist_ok=True)

    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir,subdir),exist_ok=True)

    json_data["experiment_dir"]=experiment_dir
    json_data["rest_directory_structure"]=subdirs

    with open(Path(experiment_dir)/f"{sub}_pars.json","w") as json_file:
        json.dump(make_json_serializable(json_data),json_file,indent=4,sort_keys=True)

    return json_data,experiment_dir,sub
"""
def rest_paths(experiment_dir):
    from pathlib import Path

    base=Path(experiment_dir)

    paths={
        "basic":base/"1.basic",
        "trials":base/"2.trials",
        "detrend":base/"3.detrend",
        "detrend_examples":base/"3.detrend"/"examples",
        "postICA":base/"4.postICA",
        "extra":base/"5.Extra",
        "features":base/"5.Extra"/"FE",
        "fooof":base/"6.FOOOF",
        "pkls":base/"7.pkls",
    }

    for p in paths.values():
        p.mkdir(parents=True,exist_ok=True)

    return paths
"""
def rest_paths(experiment_dir):
    from pathlib import Path

    base=Path(experiment_dir)

    paths={
        "basic":base/"1.basic",
        "trials":base/"2.trials",
        "postICA":base/"3.postICA",
        "extra":base/"4.Extra",
        "features":base/"4.Extra"/"FE",
        "fooof":base/"5.FOOOF",
        "pkls":base/"7.pkls"
    }

    for p in paths.values():
        p.mkdir(parents=True,exist_ok=True)

    return paths
"""

def loadEDF(json_data, fileName):
    raw = mne.io.read_raw(fileName+'.EDF', preload=True)
    channel_types = raw.get_channel_types()
    raw.rename_channels({'EEG FPz': 'EEG Fpz'})
        
    if 'EMG' in raw.ch_names:
        print("Il canale 'EMG' è presente e verrà escluso.")
        raw.set_channel_types({'EMG': 'emg'})
        
        raw = raw.drop_channels('EMG')
        #print(raw.ch_names)
    else:
        print("Il canale 'EMG' non è presente.")
    
    rename_dict = {ch: ch.replace('EEG ', '') for ch in raw.ch_names}
    raw.rename_channels(rename_dict)
    print(raw.ch_names)
    
    return raw


def loadASCII(json_data, fileName):
    
    with open(fileName+'.asc', 'r') as f:
        lines = f.readlines()
    metaData=lines[0:11]
    
    header1, Patient_data, Trace_date, Start_seconds, Finish_seconds, SamplingRate_hz, potential_uV, _, channel_TR, channel_names, _ = metaData
    
    header1 = header1.strip('"')
    Patient_data = Patient_data.strip('"')
    Trace_date = Trace_date.strip('"')
    Start_seconds = Start_seconds.strip('"')
    Finish_seconds = Finish_seconds.strip('"')
    SamplingRate_hz = SamplingRate_hz.strip('"')
    potential_uV = potential_uV.strip('"')
    channel_TR = channel_TR.strip('"')
    channel_names = channel_names.strip('"')
    
    print(header1)
    print(Patient_data)
    print(Trace_date)
    print(Start_seconds)
    print(Finish_seconds)
    print(SamplingRate_hz)
    print(potential_uV)
    print(channel_TR)
    
    #print(channel_names)
    channel_names = channel_names.rstrip('\n')
    channel_names = [item.strip('"') for item in channel_names.split('", "')]
    #print(channel_names)
    channel_names = [name.replace('-RF', '') for name in channel_names]
    print(channel_names)
    
    with open(fileName+'_onlyData.asc', 'w') as f:
        f.writelines(lines[11::])
    
    df = pd.read_csv(fileName+'_onlyData.asc', 
                     sep='\t', 
                     decimal=',', 
                     names=channel_names, 
                     index_col=False,
                     #header=None,
                     #skiprows='None', 
                     dtype=np.float64)
    
    # check sampling rate
    for i in ['MK', 'EMG', 'TM']:
        if i in df.columns: print(f"Il canale {i} è presente nel DataFrame.")
        else: print(f"Il canale {i} non è presente nel DataFrame.")

    lenChannel = len(df[df.columns[0]].values)
    time = float(Finish_seconds[11:14])
    print(f'Esperimento ha durata time={time} con sr={lenChannel/time} e samples={lenChannel}')

    

    return df


def add_exp_artifact(epochs, json_data, experiment_dir, sub, tau_rise=0.005, tau_decay=0.1, gain=-3e-6*30, chans=None):
    data = epochs.get_data()  # shape (n_epochs, n_channels, n_times)
    sfreq = epochs.info['sfreq']
    times = epochs.times  # in seconds
    zero_idx = np.where(times >= 0)[0][0]
    t_post = times[zero_idx:] - times[zero_idx]

    # Artefatto esponenziale charge/discharge
    artifact = gain * (1 - np.exp(-t_post / tau_rise)) * np.exp(-t_post / tau_decay)

    chan_idxs = np.arange(data.shape[1]) if chans is None else [
        epochs.ch_names.index(ch) for ch in chans
    ]

    for trial in data:
        for ch in chan_idxs:
            trial[ch, zero_idx:] += artifact

    epochs_artifacted = epochs.copy()
    epochs_artifacted._data = data  # attenzione: modifica diretta

    # Save filtered raw and ICA model
    pkl_raw_path = Path(experiment_dir) / "7.pkls" / f"{sub}_epochs_artifacted.pkl"
    with open(pkl_raw_path, 'wb') as f:
        pickle.dump(epochs_artifacted, f)
            
    return epochs_artifacted


def clean_continuum_channels(raw,json_data,experiment_dir,sub,seedChans=None):
    import mne
    import numpy as np
    import json
    import pickle
    from pathlib import Path

    if seedChans is None:
        seedChans=json_data.get("seedChans",[])

    seg_duration=float(json_data.get("rest_segment_duration",5.0))
    seg_overlap=float(json_data.get("rest_segment_overlap",0.0))

    print(f"🧠 [{sub}] REST continuum cleaning")
    print(f"   Segment duration: {seg_duration}s")
    print(f"   Segment overlap: {seg_overlap}s")

    raw_clean=raw.copy().pick("eeg")
    # raw_clean.set_eeg_reference("average")

    temp_epochs=mne.make_fixed_length_epochs(
        raw_clean,
        duration=seg_duration,
        overlap=seg_overlap,
        preload=True
    )

    temp_epochs=temp_epochs.pick("eeg")
    # temp_epochs.set_eeg_reference("average")

    n_segments_before=len(temp_epochs)
    n_channels_before=len(temp_epochs.ch_names)

    original_selection=temp_epochs.selection.copy()

    if json_data.get("do_chan_trials_selection_automatic",False):
        print("🤖 REST automatic segment/channel rejection")

        data=temp_epochs.get_data()

        chan_var=np.var(data,axis=(0,2))
        c_low=np.percentile(chan_var,5)
        c_high=np.percentile(chan_var,95)

        bad_channels=[
            ch for ch,var in zip(temp_epochs.ch_names,chan_var)
            if var<c_low or var>c_high
        ]
        bad_channels=[ch for ch in bad_channels if ch not in seedChans]

        seg_var=np.var(data,axis=(1,2))
        s_low=np.percentile(seg_var,5)
        s_high=np.percentile(seg_var,95)

        bad_segments=np.where((seg_var<s_low)|(seg_var>s_high))[0].tolist()

        temp_epochs.info["bads"]=bad_channels

        if len(bad_segments)>0:
            keep_idx=np.where(~np.isin(np.arange(len(temp_epochs)),bad_segments))[0]
            temp_epochs=temp_epochs[keep_idx]

    else:
        print("🖱️ REST manual artifact rejection")

        user_bad_channels=[
            ch for ch in json_data.get("bad_channels",[])
            if ch in temp_epochs.ch_names
        ]
        temp_epochs.info["bads"]=user_bad_channels

        fig=temp_epochs.plot(
            butterfly=False,
            n_epochs=int(json_data.get("rest_gui_n_epochs",10)),
            n_channels=len(temp_epochs.ch_names),
            block=True,
            use_opengl=True,
            scalings={"eeg":float(json_data.get("rest_gui_scaling",50e-6))}
        )

        bad_channels=list(temp_epochs.info["bads"])

        kept_selection=temp_epochs.selection.copy()
        bad_segments=sorted(list(set(original_selection)-set(kept_selection)))

    raw_clean.info["bads"]=bad_channels

    # if len(bad_channels)>0:
    #    print(f"🧩 [{sub}] Interpolo bad channels sul continuo: {bad_channels}")
    #    raw_clean.interpolate_bads(reset_bads=False)

    if len(bad_segments)>0:
        print(f"🧹 [{sub}] Annotazione segmenti bad sul continuo: {len(bad_segments)}")

        onsets=[]
        durations=[]
        descriptions=[]

        step=seg_duration-seg_overlap
        for idx in bad_segments:
            onset=float(idx*step)
            if onset+seg_duration <= raw_clean.times[-1]:
                onsets.append(onset)
                durations.append(seg_duration)
                descriptions.append("BAD_REST_SEGMENT")

        bad_ann=mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions
        )

        raw_clean.set_annotations(raw_clean.annotations+bad_ann)

    json_data["rest_cleaning_mode"]="continuum_cleaning_via_5s_segments"
    json_data["rest_segment_duration"]=seg_duration
    json_data["rest_segment_overlap"]=seg_overlap
    json_data["bad_channels"]=bad_channels
    json_data["bad_trials"]=bad_segments
    json_data["rest_segments_tot"]=int(n_segments_before)
    json_data["rest_segments_rejected"]=int(len(bad_segments))
    json_data["rest_segments_selected"]=int(n_segments_before-len(bad_segments))
    json_data["channels_tot"]=int(n_channels_before)
    json_data["channels_selected"]=int(len(raw_clean.ch_names))
    json_data["ch_names_after_cleaning"]=list(raw_clean.ch_names)

    paths=rest_paths(experiment_dir)

    with open(paths["pkls"]/f"{sub}_segments_REST_5s.pkl","wb") as f:
        pickle.dump(temp_epochs,f)

    with open(paths["pkls"]/f"{sub}_raw_REST_clean.pkl","wb") as f:
        pickle.dump(raw_clean,f)

    fig=raw_clean.plot_psd(
        fmin=json_data.get("l_freq",raw_clean.info["highpass"]),
        fmax=json_data.get("broad_band_h_freq",raw_clean.info["lowpass"]),
        xscale="log",
        show=False
    )
    fig.savefig(paths["basic"]/f"{sub}_PSD_raw_REST_clean.png")
    plt.close(fig)

    with open(Path(experiment_dir)/f"{sub}_pars.json","w") as json_file:
        json.dump(make_json_serializable(json_data),json_file,indent=4,sort_keys=True)

    print("✅ clean_continuum_channels completed")
    print(f"   Segments rejected: {json_data['rest_segments_rejected']} / {json_data['rest_segments_tot']}")
    print(f"   Bad channels: {bad_channels}")

    return raw_clean,temp_epochs,json_data



"""
def computeBasicSteps_old08072026(raw, events, json_data, experiment_dir, sub, 
                      FIGSIZE=(13, 6), 
                      computeFOOOF=True,
                     ):

    if json_data['do_pulseArtifactRej']:
        print(f"🔧 [{sub}] Step 1: Rimozione artefatto TMS + PSD")
        raw = remove_tms_artifact_and_plot_psd(
            raw=raw,
            events=events,
            json_data=json_data,
            experiment_dir=experiment_dir,
            sub=sub,
            figsize=FIGSIZE,
            do_plot=False,
            ica_continuum=json_data['do_ica_continuum'],
        )

    if json_data['do_filter_and_plot_raw']:
        print(f"🔧 [{sub}] Step 2: Filtro broad-band e notch")
        raw = filter_and_plot_raw(
            raw=raw,
            json_data=json_data,
            experiment_dir=experiment_dir,
            sub=sub,
            figsize=FIGSIZE,
        )


    if json_data['do_clean_trials_channels']:
        eeg_type=json_data.get("eeg_type",json_data.get("EEGTYPE","tms")).lower()
    
        if eeg_type=="rest":
            print(f"🔧 [{sub}] Step 3 REST: Pulizia continuum su segmenti da 5 s")
            raw,temp_epochs,json_data=clean_continuum_channels(
                raw=raw,
                json_data=json_data,
                experiment_dir=experiment_dir,
                sub=sub
            )
        else:
            print(f"🔧 [{sub}] Step 3 TMS: Pulizia epoche e canali artefattati")
            temp_epochs,json_data=clean_trials_channels(
                raw=raw,
                events=events,
                json_data=json_data,
                experiment_dir=experiment_dir,
                sub=sub
            )

    if json_data['do_prepare_epochs']:
        eeg_type=json_data.get("eeg_type",json_data.get("EEGTYPE","tms")).lower()
    
        if eeg_type=="rest":
            print(f"🔧 [{sub}] Step 4 REST: mantengo il continuo pulito come input finale")
            epochs=raw
            json_data["epochs_object_type"]="Raw"
            json_data["TEP_ID_events"]="REST_continuum_clean"
        else:
            print(f"🔧 [{sub}] Step 4 TMS: Creazione oggetto Epochs finale")
            epochs,json_data=prepare_epochs(raw,events,temp_epochs,json_data,experiment_dir,sub)
    else:
        epochs = raw
        temp_epochs = raw
    print(f"✅ [{sub}] Completato. Informazioni finali su Epochs:")
    print(epochs.info)

    if computeFOOOF:
        print(f"🔧 [{sub}] Step 5: FOOOF computation")
        df = extract_psd_features(epochs, 'preDetrend', experiment_dir, json_data)
    
    # Salva parametri
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
            json.dump(json_data, json_file, indent=4, sort_keys=True)
        
    if json_data['do_artifact']:
        epochs = add_exp_artifact(epochs,json_data, experiment_dir, sub, 
                                  tau_rise=json_data['do_artifact_rise'], 
                                  tau_decay=json_data['do_artifact_decay'], 
                                  gain=json_data['do_artifact_gain'], 
                                  chans=json_data['do_artifact_chans'])
        basicPlots(epochs, 
                   json_data, experiment_dir, 
                   sub, key='epochs_artifacted', subPath='2.Detrend', show=False)


    detrendedEpochs, json_data = computeDetrendSteps(epochs, 
                                            json_data, experiment_dir, sub, 
                                            computeFOOOF=computeFOOOF)

    return raw, epochs, detrendedEpochs, temp_epochs, json_data
"""

def detrend_rest_raw_poly(raw_clean,json_data,experiment_dir,sub):
    import mne
    import numpy as np
    import pickle
    import json
    import matplotlib.pyplot as plt
    from pathlib import Path

    order=int(json_data.get("rest_detrend_order",json_data.get("detrend_noWindowedOrder",1)))

    print(f"🧼 [{sub}] REST detrend continuo polinomiale, ordine={order}")

    raw_detrended=raw_clean.copy()
    picks=mne.pick_types(raw_detrended.info,eeg=True,exclude=[])

    data=raw_detrended.get_data(picks=picks)
    sfreq=raw_detrended.info["sfreq"]

    x=np.arange(data.shape[1],dtype=float)
    x=(x-x.mean())/x.std()

    bad_mask=np.zeros(data.shape[1],dtype=bool)

    for ann in raw_detrended.annotations:
        desc=str(ann["description"])
        if desc.startswith("BAD"):
            start=int(round(ann["onset"]*sfreq))
            stop=int(round((ann["onset"]+ann["duration"])*sfreq))
            start=max(start,0)
            stop=min(stop,data.shape[1])
            bad_mask[start:stop]=True

    good=~bad_mask

    data_dt=np.zeros_like(data)
    trends=np.zeros_like(data)

    for i in range(data.shape[0]):
        y=data[i]

        if np.sum(good)>order+1:
            coeff=np.polyfit(x[good],y[good],order)
            trend=np.polyval(coeff,x)
            data_dt[i]=y-trend
            trends[i]=trend
        else:
            data_dt[i]=y-np.mean(y)
            trends[i]=np.mean(y)

    raw_detrended._data[picks]=data_dt

    json_data["rest_detrend_done"]=True
    json_data["rest_detrend_type"]="raw_continuum_poly"
    json_data["rest_detrend_order"]=int(order)
    json_data["rest_detrend_bad_segments_excluded_from_fit"]=True
    json_data["rest_raw_detrended_n_times"]=int(raw_detrended.n_times)
    json_data["rest_raw_detrended_sfreq"]=float(raw_detrended.info["sfreq"])
    json_data["rest_raw_detrended_duration_sec"]=float(raw_detrended.times[-1])
    json_data["detrended_object_type"]="Raw_REST_clean_detrended"

    paths=rest_paths(experiment_dir)

    with open(paths["pkls"]/f"{sub}_raw_REST_detrended.pkl","wb") as f:
        pickle.dump(raw_detrended,f)

    fig=raw_clean.plot_psd(
        fmin=json_data.get("l_freq",raw_clean.info["highpass"]),
        fmax=json_data.get("broad_band_h_freq",raw_clean.info["lowpass"]),
        xscale="log",
        show=False
    )
    fig.savefig(paths["detrend"]/f"{sub}_PSD_before_REST_poly_order{order}.png")
    plt.close(fig)

    fig=raw_detrended.plot_psd(
        fmin=json_data.get("l_freq",raw_detrended.info["highpass"]),
        fmax=json_data.get("broad_band_h_freq",raw_detrended.info["lowpass"]),
        xscale="log",
        show=False
    )
    fig.savefig(paths["detrend"]/f"{sub}_PSD_after_REST_poly_order{order}.png")
    plt.close(fig)

    ch_examples=json_data.get("rest_detrend_example_chans",raw_detrended.ch_names[:min(6,len(raw_detrended.ch_names))])
    times=raw_clean.times

    for ch in ch_examples:
        if ch not in raw_detrended.ch_names:
            continue
    
        idx = raw_detrended.ch_names.index(ch)
    
        y_before = data[idx] * 1e6
        y_after = data_dt[idx] * 1e6
        y_trend = trends[idx] * 1e6
    
        y_before_plot = y_before.copy()
        y_after_plot = y_after.copy()
        y_trend_plot = y_trend.copy()
    
        y_before_plot[bad_mask] = np.nan
        y_after_plot[bad_mask] = np.nan
        y_trend_plot[bad_mask] = np.nan
    
        fig, ax = plt.subplots(figsize=(14, 5))
    
        ax.plot(
            times,
            y_before_plot,
            linewidth=0.5,
            label="clean before detrend, good samples only"
        )
    
        ax.plot(
            times,
            y_trend_plot,
            linewidth=2,
            label=f"poly trend order={order}, fitted on good samples"
        )
    
        ax.plot(
            times,
            y_after_plot,
            linewidth=0.5,
            label="after detrend, good samples only"
        )
    
        for ann in raw_detrended.annotations:
            if str(ann["description"]).startswith("BAD"):
                ax.axvspan(
                    ann["onset"],
                    ann["onset"] + ann["duration"],
                    color="red",
                    alpha=0.15,
                    linewidth=0
                )
    
        ax.set_title(f"{sub} REST detrend example - {ch}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude [µV]")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.2)
    
        fig.savefig(
            paths["detrend_examples"] / f"{sub}_REST_poly_order{order}_{ch}.png",
            dpi=150,
            bbox_inches="tight"
        )

    plt.close(fig)

    with open(Path(experiment_dir)/f"{sub}_pars.json","w") as json_file:
        json.dump(make_json_serializable(json_data),json_file,indent=4,sort_keys=True)

    print(f"✅ [{sub}] REST raw continuo detrendato salvato")
    return raw_detrended,json_data

def make_rest_fake_epochs(raw_detrended,json_data,experiment_dir,sub,
                          note="REST_fake_after_detrend",
                          subPath=None):
    import mne
    import pickle
    import json
    import numpy as np
    from pathlib import Path

    print(f"🧩 [{sub}] REST fake epoching: {note}")

    paths=rest_paths(experiment_dir)

    if subPath is None:
        subPath=Path("2.trials")/"postDetrend"

    events_fake=make_rest_events(raw_detrended,json_data)

    baseline=(
        json_data.get("baseline_cor_tmin",None),
        json_data.get("baseline_cor_tmax",None)
    )

    if baseline[0] is None or baseline[1] is None:
        baseline=None
    elif baseline[0] < json_data["epochs_timewindow_min"]:
        baseline=(json_data["epochs_timewindow_min"],baseline[1])

    epochs_fake=mne.Epochs(
        raw_detrended,
        events_fake,
        event_id=int(json_data.get("rest_event_id",999)),
        tmin=float(json_data["epochs_timewindow_min"]),
        tmax=float(json_data["epochs_timewindow_max"]),
        baseline=baseline,
        detrend=None,
        preload=True,
        reject_by_annotation=True
    )

    epochs_fake=epochs_fake.pick("eeg")
    # epochs_fake=epochs_fake.set_eeg_reference("average")

    if json_data.get("r_sfreq",None) is not None:
        epochs_fake=epochs_fake.resample(float(json_data["r_sfreq"]))

    json_data["epochs_object_type"]="Epochs_REST_fake_after_continuum_detrend"
    json_data["TEP_ID_events"]=note
    json_data["rest_fake_epochs_after_detrend"]=True
    json_data["rest_fake_epochs_n"]=int(len(epochs_fake))
    json_data["rest_fake_epochs_channels"]=int(len(epochs_fake.ch_names))
    json_data["rest_fake_epochs_ch_names"]=list(epochs_fake.ch_names)

    with open(paths["pkls"]/f"{sub}_epochs_{note}.pkl","wb") as f:
        pickle.dump(epochs_fake,f)


    npy_events_path=paths["pkls"]/f"{sub}_events_{note}.npy"
    np.save(npy_events_path,events_fake)

    basicPlots(
        epochs_fake,
        json_data,
        experiment_dir,
        sub,
        key=note,
        subPath=str(subPath),
        show=False
    )

    for ch in epochs_fake.ch_names:
        plotTrialTepVariability(
            epochs_fake,
            json_data,
            experiment_dir,
            sub,
            chanNAME=ch,
            operator=np.mean,
            save=True,
            parDir="postDetrend"
        )

    with open(Path(experiment_dir)/f"{sub}_pars.json","w") as json_file:
        json.dump(make_json_serializable(json_data),json_file,indent=4,sort_keys=True)

    print(f"✅ [{sub}] Fake epochs REST salvate in 3.trials/postDetrend e 7.pkls")
    return epochs_fake,events_fake,json_data

def auto_detect_rest_bad_channels_segments(
    raw,
    json_data,
    experiment_dir,
    sub,
    window_sec=1.0,
    channel_z_threshold=3.5,
    segment_z_threshold=3.5,
    protect_seed_channels=True,
    save=True
):
    import json
    import pickle
    import numpy as np
    import pandas as pd
    from pathlib import Path

    paths=rest_paths(experiment_dir)

    out_dir=paths["trials"]/"REST_auto_clean"
    out_dir.mkdir(parents=True,exist_ok=True)

    raw_out=raw.copy().pick("eeg")
    raw_out.load_data()

    sfreq=float(raw_out.info["sfreq"])
    n_times=int(raw_out.n_times)
    duration_sec=float(n_times/sfreq)

    window_sec=float(window_sec)

    if window_sec<=0:
        raise ValueError(
            "window_sec deve essere > 0"
        )

    start_exclusion_sec=float(
        json_data.get(
            "rest_crop_start_sec",
            0.0
        )
    )

    end_exclusion_sec=float(
        json_data.get(
            "rest_crop_end_sec",
            0.0
        )
    )

    if (
        start_exclusion_sec<0
        or end_exclusion_sec<0
    ):
        raise ValueError(
            "Le finestre iniziale e finale "
            "devono essere >= 0."
        )

    analysis_start_sample=int(
        round(
            start_exclusion_sec*sfreq
        )
    )

    analysis_stop_sample=int(
        round(
            n_times
            -end_exclusion_sec*sfreq
        )
    )

    analysis_start_sample=max(
        0,
        min(
            analysis_start_sample,
            n_times
        )
    )

    analysis_stop_sample=max(
        analysis_start_sample,
        min(
            analysis_stop_sample,
            n_times
        )
    )

    usable_samples=int(
        analysis_stop_sample
        -analysis_start_sample
    )

    if usable_samples<=0:
        raise ValueError(
            "Nessun campione disponibile dopo "
            "l'esclusione delle finestre "
            "iniziale e finale."
        )

    usable_duration_sec=float(
        usable_samples/sfreq
    )

    window_samples=max(
        1,
        int(
            round(
                window_sec*sfreq
            )
        )
    )

    n_full_windows=int(
        usable_samples//window_samples
    )

    remainder_samples=int(
        usable_samples
        -n_full_windows*window_samples
    )

    remainder_sec=float(
        remainder_samples/sfreq
    )

    print(
        f"🤖 [{sub}] REST automatic detection"
    )

    print(
        f"   Total duration: "
        f"{duration_sec:.6f} s"
    )

    print(
        f"   Initial excluded window: "
        f"{start_exclusion_sec:.6f} s"
    )

    print(
        f"   Final excluded window: "
        f"{end_exclusion_sec:.6f} s"
    )

    print(
        f"   Usable duration: "
        f"{usable_duration_sec:.6f} s"
    )

    print(
        f"   Full windows: "
        f"{n_full_windows} × "
        f"{window_sec:.6f} s"
    )

    print(
        f"   Final remainder: "
        f"{remainder_sec:.6f} s"
    )

    existing_bad_mask=np.zeros(
        n_times,
        dtype=bool
    )

    for annotation in raw_out.annotations:
        description=str(
            annotation["description"]
        )

        if not description.startswith("BAD"):
            continue

        start=int(
            round(
                float(
                    annotation["onset"]
                )*sfreq
            )
        )

        stop=int(
            round(
                (
                    float(
                        annotation["onset"]
                    )
                    +float(
                        annotation["duration"]
                    )
                )*sfreq
            )
        )

        start=max(
            0,
            min(
                start,
                n_times
            )
        )

        stop=max(
            start,
            min(
                stop,
                n_times
            )
        )

        existing_bad_mask[
            start:stop
        ]=True

    data=raw_out.get_data()

    def robust_zscore(values):
        values=np.asarray(
            values,
            dtype=float
        )

        result=np.full(
            values.shape,
            np.nan,
            dtype=float
        )

        finite=np.isfinite(
            values
        )

        if np.sum(finite)<2:
            return result

        median=float(
            np.median(
                values[finite]
            )
        )

        mad=float(
            np.median(
                np.abs(
                    values[finite]
                    -median
                )
            )
        )

        if (
            not np.isfinite(mad)
            or mad<=0
        ):
            standard_deviation=float(
                np.std(
                    values[finite]
                )
            )

            if (
                not np.isfinite(
                    standard_deviation
                )
                or standard_deviation<=0
            ):
                result[finite]=0.0
                return result

            result[finite]=(
                values[finite]
                -median
            )/standard_deviation

            return result

        result[finite]=(
            0.67448975
            *(
                values[finite]
                -median
            )
            /mad
        )

        return result

    valid_channel_mask=np.zeros(
        n_times,
        dtype=bool
    )

    valid_channel_mask[
        analysis_start_sample:
        analysis_stop_sample
    ]=True

    valid_channel_mask&=~existing_bad_mask

    if np.sum(valid_channel_mask)<2:
        raise ValueError(
            "Non ci sono abbastanza campioni validi "
            "per valutare i canali."
        )

    channel_rows=[]
    channel_log_variance=[]

    for channel_index,channel_name in enumerate(
        raw_out.ch_names
    ):
        signal=np.asarray(
            data[channel_index],
            dtype=float
        )

        valid_signal=signal[
            valid_channel_mask
        ]

        valid_signal=valid_signal[
            np.isfinite(
                valid_signal
            )
        ]

        if valid_signal.size<2:
            variance=np.nan
            peak_to_peak=np.nan
            flat_fraction=np.nan

        else:
            variance=float(
                np.var(
                    valid_signal
                )
            )

            peak_to_peak=float(
                np.ptp(
                    valid_signal
                )
            )

            differences=np.abs(
                np.diff(
                    valid_signal
                )
            )

            flat_fraction=(
                float(
                    np.mean(
                        differences<1e-12
                    )
                )
                if differences.size>0
                else 1.0
            )

        log_variance=(
            float(
                np.log10(
                    max(
                        variance,
                        np.finfo(float).tiny
                    )
                )
            )
            if np.isfinite(variance)
            else np.nan
        )

        channel_log_variance.append(
            log_variance
        )

        channel_rows.append({
            "channel":channel_name,
            "channel_index":int(
                channel_index
            ),
            "variance":variance,
            "log10_variance":log_variance,
            "peak_to_peak":peak_to_peak,
            "flat_fraction":flat_fraction
        })

    channel_log_variance=np.asarray(
        channel_log_variance,
        dtype=float
    )

    channel_z=robust_zscore(
        channel_log_variance
    )

    seed_channels=set(
        json_data.get(
            "seedChans",
            []
        )
    )

    auto_bad_channels=[]

    for index,row in enumerate(
        channel_rows
    ):
        robust_z=(
            float(
                channel_z[index]
            )
            if np.isfinite(
                channel_z[index]
            )
            else np.nan
        )

        bad_by_variance=bool(
            np.isfinite(
                channel_z[index]
            )
            and abs(
                channel_z[index]
            )>=float(
                channel_z_threshold
            )
        )

        bad_by_flatness=bool(
            np.isfinite(
                row["flat_fraction"]
            )
            and row["flat_fraction"]>=0.95
        )

        is_seed=bool(
            row["channel"] in seed_channels
        )

        is_bad=bool(
            bad_by_variance
            or bad_by_flatness
        )

        if (
            protect_seed_channels
            and is_seed
        ):
            is_bad=False

        row["robust_z"]=robust_z
        row["bad_by_variance"]=bad_by_variance
        row["bad_by_flatness"]=bad_by_flatness

        row["protected_seed"]=bool(
            protect_seed_channels
            and is_seed
        )

        row["auto_bad"]=is_bad

        if is_bad:
            auto_bad_channels.append(
                row["channel"]
            )

    existing_bad_channels=[
        channel
        for channel in raw_out.info.get(
            "bads",
            []
        )
        if channel in raw_out.ch_names
    ]

    final_bad_channels=sorted(
        set(existing_bad_channels)
        |set(auto_bad_channels)
    )

    raw_out.info["bads"]=final_bad_channels

    good_channel_indices=[
        index
        for index,channel_name in enumerate(
            raw_out.ch_names
        )
        if channel_name not in final_bad_channels
    ]

    if len(good_channel_indices)==0:
        raise RuntimeError(
            "Tutti i canali sono stati "
            "classificati come bad."
        )

    windows=[]

    for window_index in range(
        n_full_windows
    ):
        start_sample=(
            analysis_start_sample
            +window_index*window_samples
        )

        stop_sample=min(
            start_sample+window_samples,
            analysis_stop_sample
        )

        windows.append({
            "window_index":int(
                window_index
            ),
            "start_sample":int(
                start_sample
            ),
            "stop_sample":int(
                stop_sample
            ),
            "is_remainder":False
        })

    if remainder_samples>0:
        start_sample=(
            analysis_start_sample
            +n_full_windows*window_samples
        )

        stop_sample=analysis_stop_sample

        windows.append({
            "window_index":int(
                len(windows)
            ),
            "start_sample":int(
                start_sample
            ),
            "stop_sample":int(
                stop_sample
            ),
            "is_remainder":True
        })

    if len(windows)==0:
        raise ValueError(
            "Non è stata creata alcuna "
            "finestra REST."
        )

    covered_samples=sum(
        window["stop_sample"]
        -window["start_sample"]
        for window in windows
    )

    if covered_samples!=usable_samples:
        raise RuntimeError(
            "Le finestre non coprono tutta "
            "la regione utilizzabile: "
            f"{covered_samples} != "
            f"{usable_samples} campioni."
        )

    segment_rows=[]
    segment_log_variance=[]

    for window in windows:
        start_sample=int(
            window["start_sample"]
        )

        stop_sample=int(
            window["stop_sample"]
        )

        if stop_sample<=start_sample:
            continue

        existing_fraction=float(
            np.mean(
                existing_bad_mask[
                    start_sample:stop_sample
                ]
            )
        )

        segment=data[
            good_channel_indices,
            start_sample:stop_sample
        ]

        if segment.size==0:
            variance=np.nan
            rms=np.nan
            peak_to_peak=np.nan

        else:
            channel_variances=np.var(
                segment,
                axis=1
            )

            channel_rms=np.sqrt(
                np.mean(
                    segment**2,
                    axis=1
                )
            )

            channel_peak_to_peak=np.ptp(
                segment,
                axis=1
            )

            variance=float(
                np.median(
                    channel_variances
                )
            )

            rms=float(
                np.median(
                    channel_rms
                )
            )

            peak_to_peak=float(
                np.median(
                    channel_peak_to_peak
                )
            )

        log_variance=(
            float(
                np.log10(
                    max(
                        variance,
                        np.finfo(float).tiny
                    )
                )
            )
            if np.isfinite(variance)
            else np.nan
        )

        start_sec=float(
            start_sample/sfreq
        )

        stop_sec=float(
            stop_sample/sfreq
        )

        segment_rows.append({
            "window_index":int(
                window["window_index"]
            ),
            "start_sample":start_sample,
            "stop_sample":stop_sample,
            "start_sec":start_sec,
            "stop_sec":stop_sec,
            "duration_sec":float(
                (
                    stop_sample-start_sample
                )/sfreq
            ),
            "is_remainder":bool(
                window["is_remainder"]
            ),
            "existing_bad_fraction":existing_fraction,
            "median_variance":variance,
            "log10_median_variance":log_variance,
            "median_rms":rms,
            "median_peak_to_peak":peak_to_peak
        })

        segment_log_variance.append(
            log_variance
        )

    segment_log_variance=np.asarray(
        segment_log_variance,
        dtype=float
    )

    reference_mask=np.asarray(
        [
            (
                row[
                    "existing_bad_fraction"
                ]==0
                and np.isfinite(
                    row[
                        "log10_median_variance"
                    ]
                )
            )
            for row in segment_rows
        ],
        dtype=bool
    )

    segment_z=np.full(
        len(segment_rows),
        np.nan,
        dtype=float
    )

    if np.sum(reference_mask)>=2:
        segment_z[
            reference_mask
        ]=robust_zscore(
            segment_log_variance[
                reference_mask
            ]
        )

    else:
        print(
            "⚠️ Troppo poche finestre pulite "
            "per calcolare gli outlier temporali."
        )

    auto_bad_windows=[]

    for index,row in enumerate(
        segment_rows
    ):
        robust_z=(
            float(
                segment_z[index]
            )
            if np.isfinite(
                segment_z[index]
            )
            else np.nan
        )

        already_bad=bool(
            row[
                "existing_bad_fraction"
            ]>0
        )

        auto_bad=bool(
            not already_bad
            and np.isfinite(
                segment_z[index]
            )
            and abs(
                segment_z[index]
            )>=float(
                segment_z_threshold
            )
        )

        row["robust_z"]=robust_z
        row["already_bad"]=already_bad
        row["auto_bad"]=auto_bad

        if auto_bad:
            auto_bad_windows.append(
                row
            )

    new_annotations=raw_out.annotations.copy()

    for row in auto_bad_windows:
        new_annotations.append(
            onset=float(
                row["start_sec"]
            ),
            duration=float(
                row["duration_sec"]
            ),
            description="BAD_REST_AUTO"
        )

    raw_out.set_annotations(
        new_annotations
    )

    df_channels=pd.DataFrame(
        channel_rows
    )

    df_segments=pd.DataFrame(
        segment_rows
    )

    json_data[
        "rest_auto_detection_enabled"
    ]=True

    json_data[
        "rest_auto_window_sec"
    ]=float(window_sec)

    json_data[
        "rest_auto_window_samples"
    ]=int(window_samples)

    json_data[
        "rest_auto_full_windows"
    ]=int(n_full_windows)

    json_data[
        "rest_auto_total_windows"
    ]=int(len(windows))

    json_data[
        "rest_auto_remainder_samples"
    ]=int(remainder_samples)

    json_data[
        "rest_auto_remainder_sec"
    ]=float(remainder_sec)

    json_data[
        "rest_auto_remainder_included"
    ]=bool(
        remainder_samples>0
    )

    json_data[
        "rest_auto_analysis_start_sample"
    ]=int(analysis_start_sample)

    json_data[
        "rest_auto_analysis_stop_sample"
    ]=int(analysis_stop_sample)

    json_data[
        "rest_auto_analysis_start_sec"
    ]=float(
        analysis_start_sample/sfreq
    )

    json_data[
        "rest_auto_analysis_stop_sec"
    ]=float(
        analysis_stop_sample/sfreq
    )

    json_data[
        "rest_auto_usable_samples"
    ]=int(usable_samples)

    json_data[
        "rest_auto_covered_samples"
    ]=int(covered_samples)

    json_data[
        "rest_auto_usable_duration_sec"
    ]=float(usable_duration_sec)

    json_data[
        "rest_auto_channel_z_threshold"
    ]=float(channel_z_threshold)

    json_data[
        "rest_auto_segment_z_threshold"
    ]=float(segment_z_threshold)

    json_data[
        "rest_auto_protect_seed_channels"
    ]=bool(protect_seed_channels)

    json_data[
        "rest_auto_bad_channels"
    ]=list(auto_bad_channels)

    json_data[
        "bad_channels"
    ]=list(final_bad_channels)

    json_data[
        "rest_auto_bad_windows"
    ]=[
        int(
            row["window_index"]
        )
        for row in auto_bad_windows
    ]

    json_data[
        "rest_auto_bad_windows_tot"
    ]=int(
        len(auto_bad_windows)
    )

    json_data[
        "rest_auto_bad_seconds"
    ]=float(
        sum(
            row["duration_sec"]
            for row in auto_bad_windows
        )
    )

    json_data[
        "rest_auto_annotation_description"
    ]="BAD_REST_AUTO"

    if save:
        channel_csv=(
            out_dir
            /f"{sub}_auto_channels.csv"
        )

        windows_csv=(
            out_dir
            /f"{sub}_auto_windows.csv"
        )

        df_channels.to_csv(
            channel_csv,
            index=False
        )

        df_segments.to_csv(
            windows_csv,
            index=False
        )

        with open(
            out_dir
            /f"{sub}_raw_autoAnnotated.pkl",
            "wb"
        ) as file:
            pickle.dump(
                raw_out,
                file
            )

        raw_out.save(
            out_dir
            /f"{sub}_raw_autoAnnotated.fif",
            overwrite=True
        )

        json_data[
            "rest_auto_channels_csv"
        ]=str(channel_csv)

        json_data[
            "rest_auto_windows_csv"
        ]=str(windows_csv)

        with open(
            Path(experiment_dir)
            /f"{sub}_pars.json",
            "w"
        ) as file:
            json.dump(
                make_json_serializable(
                    json_data
                ),
                file,
                indent=4,
                sort_keys=True
            )

    print(
        f"✅ [{sub}] REST automatic detection completed"
    )

    print(
        f"   Auto bad channels: "
        f"{auto_bad_channels}"
    )

    print(
        f"   Final bad channels: "
        f"{final_bad_channels}"
    )

    print(
        f"   Auto bad windows: "
        f"{len(auto_bad_windows)} / "
        f"{len(windows)}"
    )

    print(
        f"   Auto bad seconds: "
        f"{json_data['rest_auto_bad_seconds']:.3f}"
    )

    print(
        f"   Covered samples: "
        f"{covered_samples} / "
        f"{usable_samples}"
    )

    return (
        raw_out,
        df_channels,
        df_segments,
        json_data
    )

def computeBasicSteps(raw, events, json_data, experiment_dir, sub,
                      FIGSIZE=(13, 6),
                      computeFOOOF=False):

    import json
    import pickle
    from pathlib import Path

    experiment_dir = str(
        Path(json_data.get("experiment_dir", experiment_dir))
        .expanduser()
        .resolve()
    )
    json_data["experiment_dir"] = experiment_dir

    eeg_type = json_data.get("eeg_type", json_data.get("EEGTYPE", "tms")).lower()

    if eeg_type != "rest":
        raise ValueError(
            "Questa computeBasicSteps è REST-only. "
            "Per TEP usa tmspath_utils.py."
        )

    print(f"🧠 [{sub}] REST basic pipeline: raw continuum only + free annotations")

    paths = rest_paths(experiment_dir)

    if json_data.get("do_filter_and_plot_raw", True):
        print(f"🔧 [{sub}] Step 1 REST: filtro continuo")
        raw = filter_and_plot_raw(
            raw=raw,
            json_data=json_data,
            experiment_dir=experiment_dir,
            sub=sub,
            figsize=FIGSIZE,
            subPath="1.basic"
        )

    if json_data.get("do_clean_trials_channels",True):
        raw_for_annotation=raw.copy().pick("eeg")
    
        if json_data.get(
            "do_chan_trials_selection_automatic",
            False
        ):
            print(
                f"🔧 [{sub}] Step 2A REST: "
                "rilevamento automatico canali e finestre BAD"
            )
    
            (
                raw_for_annotation,
                df_auto_channels,
                df_auto_windows,
                json_data
            )=auto_detect_rest_bad_channels_segments(
                raw=raw_for_annotation,
                json_data=json_data,
                experiment_dir=experiment_dir,
                sub=sub,
                window_sec=float(
                    json_data.get(
                        "rest_auto_window_sec",
                        1.0
                    )
                ),
                channel_z_threshold=float(
                    json_data.get(
                        "rest_auto_channel_z_threshold",
                        3.5
                    )
                ),
                segment_z_threshold=float(
                    json_data.get(
                        "rest_auto_segment_z_threshold",
                        3.5
                    )
                ),

                protect_seed_channels=bool(
                    json_data.get(
                        "rest_auto_protect_seed_channels",
                        True
                    )
                ),
                save=True
            )
    
        else:
            print(
                f"⏭️ [{sub}] REST: "
                "rilevamento automatico disattivato"
            )
    
        print(
            f"🔧 [{sub}] Step 2B REST: "
            "revisione manuale delle annotazioni e dei canali"
        )
    
        raw_clean,json_data=annotate_rest_bad_segments_free(
            raw=raw_for_annotation,
            json_data=json_data,
            experiment_dir=experiment_dir,
            sub=sub,
            h_freq_vis=json_data.get(
                "rest_annotation_h_freq_vis",
                60
            ),
            scaling=json_data.get(
                "rest_gui_scaling",
                50e-6
            ),
            save=True,
            note="free_annotations"
        )

    
    else:
        print(f"⏭️ [{sub}] REST: salto annotazione BAD")
        raw_clean = raw.copy().pick("eeg")
        raw_clean.info["bads"] = json_data.get("bad_channels", [])

        json_data["rest_bad_segment_mode"] = "none"
        json_data["rest_free_annotations_tot"] = int(len(raw_clean.annotations))
        json_data["rest_free_bad_annotations_tot"] = 0
        json_data["rest_free_bad_seconds"] = 0.0
        json_data["rest_free_bad_channels"] = list(raw_clean.info.get("bads", []))

    if json_data.get("do_rest_detrend", False):
        print(f"🔧 [{sub}] Step 3 REST: detrend polinomiale continuo / DC correction")
        raw_clean, json_data = detrend_rest_raw_poly(
            raw_clean=raw_clean,
            json_data=json_data,
            experiment_dir=experiment_dir,
            sub=sub
        )
    else:
        print(f"⏭️ [{sub}] REST: salto detrend continuo / DC correction")

    json_data["rest_pipeline_mode"] = "raw_continuum_filter_freeAnnotation_detrend"
    json_data["raw_object_type"] = "Raw_REST_clean_freeAnnotated"
    json_data["segments_object_type"] = "None"
    json_data["epochs_object_type"] = "None"
    json_data["rest_fixed_epochs_created"] = False
    json_data["rest_segment_duration"] = None
    json_data["rest_segment_overlap"] = None

    if json_data.get("do_rest_detrend", False):
        json_data["raw_object_type"] = "Raw_REST_clean_freeAnnotated_detrended"

    bad_annotations = [
        ann for ann in raw_clean.annotations
        if str(ann["description"]).startswith("BAD")
    ]

    json_data["bad_channels"] = list(raw_clean.info.get("bads", []))
    json_data["rest_bad_annotations_tot"] = int(len(bad_annotations))
    json_data["rest_bad_seconds"] = float(sum([ann["duration"] for ann in bad_annotations]))

    with open(paths["pkls"] / f"{sub}_raw_REST_clean.pkl", "wb") as f:
        pickle.dump(raw_clean, f)

    raw_clean.save(
        paths["pkls"] / f"{sub}_raw_REST_clean.fif",
        overwrite=True
    )

    ann_df = raw_clean.annotations.to_data_frame()
    ann_df.to_csv(
        paths["trials"] / f"{sub}_REST_annotations_final.csv",
        index=False
    )

    with open(Path(experiment_dir) / f"{sub}_pars.json", "w") as json_file:
        json.dump(
            make_json_serializable(json_data),
            json_file,
            indent=4,
            sort_keys=True
        )

    print(f"✅ [{sub}] REST basic preprocessing completato")
    print(f"   raw_clean:        {type(raw_clean)}")
    print(f"   bad channels:     {json_data.get('bad_channels', [])}")
    print(f"   BAD annotations:  {json_data.get('rest_bad_annotations_tot', 0)}")
    print(f"   BAD seconds:      {json_data.get('rest_bad_seconds', 0):.2f}")
    print(f"   rest detrend:     {json_data.get('do_rest_detrend', False)}")
    print(f"   detrend order:    {json_data.get('rest_detrend_order', 'NA')}")
    print(f"   fixed epochs:     False")
    print(f"   5s segments:      False")

    return raw_clean, json_data

def make_rest_fixed_epochs(raw_clean,json_data,experiment_dir,sub):
    import mne
    import pickle
    import json
    from pathlib import Path

    paths=rest_paths(experiment_dir)

    seg_duration=float(json_data.get("rest_segment_duration",5.0))
    seg_overlap=float(json_data.get("rest_segment_overlap",0.0))

    print(f"🧩 [{sub}] Creo epochs REST finali da {seg_duration}s")

    epochs_5s=mne.make_fixed_length_epochs(
        raw_clean,
        duration=seg_duration,
        overlap=seg_overlap,
        preload=True,
        reject_by_annotation=True
    )

    epochs_5s=epochs_5s.pick("eeg")

    if json_data.get("r_sfreq",None) is not None:
        epochs_5s=epochs_5s.resample(float(json_data["r_sfreq"]))

    json_data["epochs_object_type"]="Epochs_REST_fixed_5s"
    json_data["rest_epochs_duration"]=float(seg_duration)
    json_data["rest_epochs_overlap"]=float(seg_overlap)
    json_data["rest_epochs_n"]=int(len(epochs_5s))
    json_data["rest_epochs_channels"]=int(len(epochs_5s.ch_names))
    json_data["rest_epochs_ch_names"]=list(epochs_5s.ch_names)

    with open(paths["pkls"]/f"{sub}_epochs_REST_5s.pkl","wb") as f:
        pickle.dump(epochs_5s,f)

    basicPlots(
        epochs_5s,
        json_data,
        experiment_dir,
        sub,
        key="REST_5s",
        subPath="2.trials",
        show=False
    )

    for ch in epochs_5s.ch_names:
        plotTrialTepVariability(
            epochs_5s,
            json_data,
            experiment_dir,
            sub,
            chanNAME=ch,
            operator=np.mean,
            save=True,
            parDir="."
        )

    with open(Path(experiment_dir)/f"{sub}_pars.json","w") as json_file:
        json.dump(make_json_serializable(json_data),json_file,indent=4,sort_keys=True)

    print(f"✅ [{sub}] Epochs REST 5s salvati")
    return epochs_5s,json_data


def make_synthetic_rest_raw_like(
    raw,
    base_freq=10.0,
    jitter_hz=5.0,
    amplitude_uv=20.0,
    noise_uv=5.0,
    add_aperiodic=True,
    aperiodic_exponent=1.5,
    aperiodic_amplitude_uv=8.0,
    aperiodic_fmin=0.5,
    add_drift=True,
    drift_type="quadratic",
    drift_amplitude_uv=50.0,
    seed=42,
    random_phase=True
):
    import numpy as np

    rng=np.random.default_rng(seed)

    raw_out=raw.copy().load_data()

    eeg_picks=[
        i
        for i,ch_type in enumerate(raw_out.get_channel_types())
        if ch_type=="eeg"
    ]

    sfreq=float(raw_out.info["sfreq"])
    n_times=int(raw_out.n_times)
    t=np.arange(n_times,dtype=float)/sfreq

    data=raw_out.get_data().copy()

    fft_freqs=np.fft.rfftfreq(
        n_times,
        d=1.0/sfreq
    )

    aperiodic_scale=np.zeros_like(
        fft_freqs,
        dtype=float
    )

    if add_aperiodic:
        valid=fft_freqs>=float(aperiodic_fmin)

        aperiodic_scale[valid]=(
            fft_freqs[valid]
            **(-float(aperiodic_exponent)/2.0)
        )

    t_norm=t/t[-1] if t[-1]>0 else np.zeros_like(t)

    if add_drift:
        if drift_type=="linear":
            drift_profile=t_norm
        elif drift_type=="quadratic":
            drift_profile=t_norm**2
        elif drift_type=="cubic":
            drift_profile=t_norm**3
        elif drift_type=="sinusoidal":
            drift_profile=np.sin(
                2.0*np.pi*0.02*t
            )
        else:
            raise ValueError(
                "drift_type deve essere: "
                "'linear', 'quadratic', 'cubic' o 'sinusoidal'"
            )

        drift_profile-=np.mean(drift_profile)
        max_abs=np.max(np.abs(drift_profile))

        if max_abs>0:
            drift_profile/=max_abs

        drift_profile*=float(drift_amplitude_uv)*1e-6

    else:
        drift_profile=np.zeros_like(t)

    channel_freqs={}
    channel_phases={}
    channel_drift_sign={}

    for idx in eeg_picks:
        ch=raw_out.ch_names[idx]

        frequency=base_freq+rng.uniform(
            -jitter_hz,
            jitter_hz
        )
        frequency=max(0.1,float(frequency))

        phase=(
            rng.uniform(0.0,2.0*np.pi)
            if random_phase
            else 0.0
        )

        sinusoid=(
            float(amplitude_uv)
            *1e-6
            *np.sin(
                2.0*np.pi*frequency*t+phase
            )
        )

        white_noise=rng.normal(
            loc=0.0,
            scale=float(noise_uv)*1e-6,
            size=n_times
        )

        signal=sinusoid+white_noise

        if add_aperiodic:
            spectrum=(
                rng.normal(size=fft_freqs.size)
                +1j*rng.normal(size=fft_freqs.size)
            )*aperiodic_scale

            spectrum[0]=0.0

            if n_times%2==0:
                spectrum[-1]=spectrum[-1].real+0j

            aperiodic=np.fft.irfft(
                spectrum,
                n=n_times
            )

            aperiodic-=np.mean(aperiodic)

            std=np.std(aperiodic)

            if np.isfinite(std) and std>0:
                aperiodic/=std

            aperiodic*=(
                float(aperiodic_amplitude_uv)
                *1e-6
            )

            signal+=aperiodic

        drift_sign=rng.choice([-1.0,1.0])
        signal+=drift_sign*drift_profile

        data[idx,:]=signal

        channel_freqs[ch]=float(frequency)
        channel_phases[ch]=float(phase)
        channel_drift_sign[ch]=float(drift_sign)

    raw_out._data=data

    synth_info={
        "synthetic_rest_base_freq":float(base_freq),
        "synthetic_rest_jitter_hz":float(jitter_hz),
        "synthetic_rest_amplitude_uv":float(amplitude_uv),
        "synthetic_rest_noise_uv":float(noise_uv),
        "synthetic_rest_add_aperiodic":bool(add_aperiodic),
        "synthetic_rest_aperiodic_exponent":float(
            aperiodic_exponent
        ),
        "synthetic_rest_aperiodic_amplitude_uv":float(
            aperiodic_amplitude_uv
        ),
        "synthetic_rest_aperiodic_fmin":float(
            aperiodic_fmin
        ),
        "synthetic_rest_add_drift":bool(add_drift),
        "synthetic_rest_drift_type":str(drift_type),
        "synthetic_rest_drift_amplitude_uv":float(
            drift_amplitude_uv
        ),
        "synthetic_rest_seed":int(seed),
        "synthetic_rest_random_phase":bool(random_phase),
        "synthetic_rest_channel_freqs_hz":channel_freqs,
        "synthetic_rest_channel_phases_rad":channel_phases,
        "synthetic_rest_channel_drift_sign":channel_drift_sign
    }

    return raw_out,synth_info


def crop_rest_edges(raw,json_data):
    start_sec=float(json_data.get("rest_crop_start_sec",0.0))
    end_sec=float(json_data.get("rest_crop_end_sec",0.0))

    duration=float(raw.times[-1])

    if start_sec<0 or end_sec<0:
        raise ValueError(
            "rest_crop_start_sec e rest_crop_end_sec devono essere >= 0"
        )

    if start_sec+end_sec>=duration:
        raise ValueError(
            f"Crop non valido: start={start_sec}s + end={end_sec}s "
            f">= durata={duration:.3f}s"
        )

    tmin=start_sec
    tmax=duration-end_sec

    raw_out=raw.copy().crop(
        tmin=tmin,
        tmax=tmax,
        include_tmax=True
    )

    json_data["rest_crop_applied"]=bool(
        start_sec>0 or end_sec>0
    )
    json_data["rest_crop_start_sec"]=start_sec
    json_data["rest_crop_end_sec"]=end_sec
    json_data["rest_duration_before_crop_sec"]=duration
    json_data["rest_duration_after_crop_sec"]=float(
        raw_out.times[-1]
    )
    json_data["rest_n_times_after_crop"]=int(
        raw_out.n_times
    )

    print(
        f"✂️ REST crop: rimossi {start_sec:.3f}s iniziali "
        f"e {end_sec:.3f}s finali"
    )
    print(
        f"   Durata: {duration:.3f}s → "
        f"{raw_out.times[-1]:.3f}s"
    )

    return raw_out,json_data

def computeBasicSteps_20072026(raw,events,json_data,experiment_dir,sub,
                      FIGSIZE=(13,6),
                      computeFOOOF=False):

    import json
    import pickle
    import mne
    from pathlib import Path

    experiment_dir=str(Path(json_data.get("experiment_dir",experiment_dir)).expanduser().resolve())
    json_data["experiment_dir"]=experiment_dir

    eeg_type=json_data.get("eeg_type",json_data.get("EEGTYPE","tms")).lower()

    if eeg_type!="rest":
        raise ValueError(
            "Questa computeBasicSteps è REST-only. "
            "Per TEP usa tmspath_utils.py."
        )

    print(f"🧠 [{sub}] REST basic pipeline: raw continuum only")

    paths=rest_paths(experiment_dir)

    if json_data.get("do_filter_and_plot_raw",True):
        print(f"🔧 [{sub}] Step 1 REST: filtro continuo")
        raw=filter_and_plot_raw(
            raw=raw,
            json_data=json_data,
            experiment_dir=experiment_dir,
            sub=sub,
            figsize=FIGSIZE,
            subPath="1.basic"
        )

    if json_data.get("do_clean_trials_channels",True):
        print(f"🔧 [{sub}] Step 2 REST: cleaning su segmenti da 5 s")
        raw_clean,segments_5s,json_data=clean_continuum_channels(
            raw=raw,
            json_data=json_data,
            experiment_dir=experiment_dir,
            sub=sub
        )
    else:
        print(f"⏭️ [{sub}] REST: salto cleaning")
        raw_clean=raw.copy().pick("eeg")

        segments_5s=mne.make_fixed_length_epochs(
            raw_clean,
            duration=float(json_data.get("rest_segment_duration",5.0)),
            overlap=float(json_data.get("rest_segment_overlap",0.0)),
            preload=True,
            reject_by_annotation=True
        )

        segments_5s=segments_5s.pick("eeg")

    if json_data.get("do_rest_detrend",False):
        print(f"🔧 [{sub}] Step 3 REST: detrend polinomiale continuo / DC correction")
        raw_clean,json_data=detrend_rest_raw_poly(
            raw_clean=raw_clean,
            json_data=json_data,
            experiment_dir=experiment_dir,
            sub=sub
        )
    else:
        print(f"⏭️ [{sub}] REST: salto detrend continuo / DC correction")

    json_data["rest_pipeline_mode"]="raw_continuum_filter_clean_detrend"
    json_data["raw_object_type"]="Raw_REST_clean"
    json_data["segments_object_type"]="Epochs_REST_5s_segments_for_cleaning_only"
    json_data["epochs_object_type"]="None"
    json_data["rest_fixed_epochs_created"]=False

    if json_data.get("do_rest_detrend",False):
        json_data["raw_object_type"]="Raw_REST_clean_detrended"

    with open(paths["pkls"]/f"{sub}_raw_REST_clean.pkl","wb") as f:
        pickle.dump(raw_clean,f)

    with open(paths["pkls"]/f"{sub}_segments_REST_5s_cleaning.pkl","wb") as f:
        pickle.dump(segments_5s,f)

    with open(Path(experiment_dir)/f"{sub}_pars.json","w") as json_file:
        json.dump(make_json_serializable(json_data),json_file,indent=4,sort_keys=True)

    print(f"✅ [{sub}] REST basic preprocessing completato")
    print(f"   raw_clean:       {type(raw_clean)}")
    print(f"   segments_5s:     {type(segments_5s)}")
    print(f"   bad channels:    {json_data.get('bad_channels',[])}")
    print(f"   bad segments:    {json_data.get('rest_segments_rejected',0)} / {json_data.get('rest_segments_tot','NA')}")
    print(f"   rest detrend:    {json_data.get('do_rest_detrend',False)}")
    print(f"   detrend order:   {json_data.get('rest_detrend_order','NA')}")
    print(f"   fixed epochs:    False")

    return raw_clean,segments_5s,json_data


def add_aperiodic_component(raw_obj, exponent=1.5, amplitude_uv=8.0, fmin=0.5, seed=42):
    data=raw_obj.get_data()
    n_channels,n_samples=data.shape
    sfreq=raw_obj.info["sfreq"]

    freqs=np.fft.rfftfreq(n_samples,d=1.0/sfreq)

    scale=np.zeros_like(freqs)
    valid=freqs>=fmin
    scale[valid]=freqs[valid]**(-exponent/2.0)

    rng=np.random.default_rng(seed)
    noise=np.empty_like(data)

    for ch in range(n_channels):
        spectrum=(
            rng.normal(size=freqs.size)
            +1j*rng.normal(size=freqs.size)
        )*scale

        spectrum[0]=0.0

        x=np.fft.irfft(spectrum,n=n_samples)
        std=np.std(x)

        if std>0:
            x=x/std

        noise[ch]=x*amplitude_uv*1e-6

    raw_out=raw_obj.copy()
    raw_out._data=data+noise

    return raw_out


def load_and_prepare_raw_data(fileName, json_data, experiment_dir, sub):
    import os
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import mne
    from tmspath_utils_rest import loadASCII, loadEDF
    json_data, experiment_dir, sub = directorySetup(json_data)
    
    experiment_dir=json_data["experiment_dir"]
    sub=json_data["subject"]

    if 'sourceData' not in json_data:
        raise KeyError("⚠️ 'sourceData' non è definito in json_data.")

    source = json_data['sourceData']
    dataType = json_data['dataType']

    json_data['pulse_artifact_rej_smoothingvalue'] = 0.002
    saveNote = ''
    raw, events = None, None
  
    def save_layout_and_metadata(raw,note):
        import os
        import json
        import matplotlib.pyplot as plt
        from pathlib import Path
    
        basic_dir=Path(experiment_dir).expanduser().resolve()/"1.basic"
        basic_dir.mkdir(parents=True,exist_ok=True)
    
        safe_note=str(note).replace("\\","_").replace("/","_").replace(":","_")
        safe_sub=str(sub).replace("\\","_").replace("/","_").replace(":","_")
    
        layout_path=basic_dir/f"{safe_sub}_{safe_note}_scalplayout.png"
    
        print(f"[DEBUG] Saving layout to: {layout_path}")
        print(f"[DEBUG] Parent exists: {layout_path.parent.exists()}")
    
        fig=raw.plot_sensors(show_names=True,show=False)
        fig.savefig(str(layout_path),dpi=300,bbox_inches="tight")
        plt.close(fig)
    
        pars_path=Path(experiment_dir).expanduser().resolve()/f"{safe_sub}_pars.json"
        with open(pars_path,"w") as json_file:
            json.dump(make_json_serializable(json_data),json_file,indent=4,sort_keys=True)

    # === CASO SIMS ===
    if source == 'SIMS':
        json_data['pulse_artifact_rej_timewindow_min'] = -0.002  # not used in sims
        json_data['pulse_artifact_rej_timewindow_max'] = 0.008  # not used in sims
        #json_data['detrend_minTimeWindowOffset'] = json_data['pulse_artifact_rej_timewindow_max'] # not used in sims

        # fileName = f"{json_data['mainDir']}\\{json_data['subject']}.fif"
        epochs = mne.read_epochs(fileName, preload=True)
        basicPlots(epochs, json_data, experiment_dir, sub, key='epochsOK', subPath='1.basic')
        with open(Path(experiment_dir) / '7.pkls' / f'{sub}_epochsOK.pkl', 'wb') as f:
            pickle.dump(epochs, f)
        json_data['sfreq'] = epochs.info['sfreq']
        json_data['r_sfreq'] = 512
        raw = epochs
        events = raw.events
        json_data['TEP_ID_events'] = 'no_events'
        save_layout_and_metadata(raw, 'no_events')
        return raw, events, json_data

    """
    # === CASO MAYER ===
    if source == 'MAYER':
        # json_data['pulse_artifact_rej_timewindow_min'] = -0.002
        # json_data['pulse_artifact_rej_timewindow_max'] = 0.008
        # json_data['detrend_minTimeWindowOffset'] = json_data['pulse_artifact_rej_timewindow_max']
        if dataType == 'ASCII':
            df = loadASCII(fileName, fileName)
            data = df.values.T * 1e-6
            ASCII_events = np.where(df['MK'] == df['MK'].unique()[1])[0]
            raw = loadEDF(json_data, fileName)
            do_runame=False
            if do_runame:
                rename_dict = {'T3': 'FT7', 'T4': 'FT8', 'T5': 'TP7', 'T6': 'TP8'}<
                raw.rename_channels(rename_dict)
            raw.set_montage('easycap-M1', verbose=True)
            raw._data = data[:len(raw.ch_names)]
            json_data['ch_names'] = raw.ch_names
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            TMScode = np.unique(events[:, 2])[0]
            events = events[events[:, 2] == TMScode]
            saveNote = 'EDF_events'
            json_data['TEP_ID_events'] = saveNote
            json_data['sfreq'] = raw.info['sfreq']
            save_layout_and_metadata(raw, saveNote)
            return raw, events, json_data
    """
    # === CASO MAYER === NEW (07/07/2026) update for rest
    if source=="MAYER" and dataType=="ASCII":
        df_ascii=loadASCII(fileName,fileName)
        data=df_ascii.values.T*1e-6
    
        eeg_type=json_data.get(
            "eeg_type",
            json_data.get("EEGTYPE","tms")
        ).lower()
    
        raw=loadEDF(json_data,fileName)
    
        rename_case={
            "FPz":"Fpz",
            "Fc1":"FC1",
            "Fc2":"FC2",
            "Af4":"AF4",
            "Fc5":"FC5",
            "Fc6":"FC6",
            "Po3":"PO3",
            "Tp9":"TP9",
            "Tp10":"TP10",
            "Po4":"PO4",
            "Cp1":"CP1",
            "Cp2":"CP2",
            "Cp5":"CP5",
            "Cp6":"CP6"
        }
    
        raw.rename_channels({
            old:new
            for old,new in rename_case.items()
            if old in raw.ch_names
        })
    
        ch_types={}
    
        if "TM" in raw.ch_names:
            ch_types["TM"]="misc"
    
        if "MK" in raw.ch_names:
            ch_types["MK"]="stim"
    
        if ch_types:
            raw.set_channel_types(ch_types)
    
        montage_df=pd.read_csv("./outputFile.csv")
    
        scale=0.095-0.005
        ch_pos={}
    
        for _,row in montage_df.iterrows():
            ch=str(row["labels"])
    
            if ch not in raw.ch_names:
                continue
    
            x=-float(row["Y"])*scale
            y=float(row["X"])*scale
            z=float(row["Z"])*scale
    
            if ch in ["TP9","TP10"]:
                radius=np.sqrt(x**2+y**2)
                max_radius_tp=0.086-0.050*0.50
    
                if radius>max_radius_tp:
                    shrink=max_radius_tp/radius
                    x*=shrink
                    y*=shrink
    
            ch_pos[ch]=np.array([x,y,z],dtype=float)
    
        if "TP9" in ch_pos:
            print("TP9:",ch_pos["TP9"])
    
        if "TP10" in ch_pos:
            print("TP10:",ch_pos["TP10"])
    
        montage=mne.channels.make_dig_montage(
            ch_pos=ch_pos,
            coord_frame="head"
        )
    
        raw.set_montage(
            montage,
            on_missing="warn"
        )
    
        if data.shape[0]<len(raw.ch_names):
            raise ValueError(
                f"I dati ASCII contengono {data.shape[0]} canali, "
                f"ma il Raw ne contiene {len(raw.ch_names)}."
            )
    
        raw._data=data[:len(raw.ch_names)]
    
        json_data["ch_names"]=list(raw.ch_names)
        json_data["sfreq"]=float(raw.info["sfreq"])
    
        if eeg_type=="rest" and json_data.get(
            "use_synthetic_rest_signal",
            False
        ):
            print("🧪 EEG REST synthetic mode")
            print("   Original MAYER file used for metadata, montage and duration")
            print("   Raw data replaced with synthetic REST signal")
    
            raw,synth_info=make_synthetic_rest_raw_like(
                raw,
                base_freq=float(
                    json_data.get(
                        "synthetic_rest_base_freq",
                        10.0
                    )
                ),
                jitter_hz=float(
                    json_data.get(
                        "synthetic_rest_jitter_hz",
                        5.0
                    )
                ),
                amplitude_uv=float(
                    json_data.get(
                        "synthetic_rest_amplitude_uv",
                        20.0
                    )
                ),
                noise_uv=float(
                    json_data.get(
                        "synthetic_rest_noise_uv",
                        5.0
                    )
                ),
                add_aperiodic=bool(
                    json_data.get(
                        "synthetic_rest_add_aperiodic",
                        True
                    )
                ),
                aperiodic_exponent=float(
                    json_data.get(
                        "synthetic_rest_aperiodic_exponent",
                        1.5
                    )
                ),
                aperiodic_amplitude_uv=float(
                    json_data.get(
                        "synthetic_rest_aperiodic_amplitude_uv",
                        8.0
                    )
                ),
                aperiodic_fmin=float(
                    json_data.get(
                        "synthetic_rest_aperiodic_fmin",
                        0.5
                    )
                ),
                add_drift=bool(
                    json_data.get(
                        "synthetic_rest_add_drift",
                        True
                    )
                ),
                drift_type=str(
                    json_data.get(
                        "synthetic_rest_drift_type",
                        "quadratic"
                    )
                ),
                drift_amplitude_uv=float(
                    json_data.get(
                        "synthetic_rest_drift_amplitude_uv",
                        50.0
                    )
                ),
                seed=int(
                    json_data.get(
                        "synthetic_rest_seed",
                        42
                    )
                ),
                random_phase=bool(
                    json_data.get(
                        "synthetic_rest_random_phase",
                        True
                    )
                )
            )
    
            json_data.update(synth_info)
            json_data["data_replaced_with_synthetic_rest"]=True
            json_data["synthetic_rest_duration_sec"]=float(
                raw.times[-1]
            )
            json_data["synthetic_rest_n_times"]=int(
                raw.n_times
            )
            json_data["synthetic_rest_sfreq"]=float(
                raw.info["sfreq"]
            )
    
        else:
            json_data["data_replaced_with_synthetic_rest"]=False
    
        if eeg_type=="rest":
            crop_start_sec=float(
                json_data.get(
                    "rest_crop_start_sec",
                    0.0
                )
            )
    
            crop_end_sec=float(
                json_data.get(
                    "rest_crop_end_sec",
                    0.0
                )
            )
    
            crop_edges=bool(
                json_data.get(
                    "rest_crop_edges",
                    False
                )
            )
    
            preselect_edges=bool(
                json_data.get(
                    "rest_preselect_edges_as_bad",
                    True
                )
            )
    
            duration_before=float(raw.times[-1])
    
            if crop_start_sec<0 or crop_end_sec<0:
                raise ValueError(
                    "rest_crop_start_sec e rest_crop_end_sec "
                    "devono essere >= 0"
                )
    
            if crop_start_sec+crop_end_sec>=duration_before:
                raise ValueError(
                    f"Intervalli REST non validi: "
                    f"start={crop_start_sec}s, "
                    f"end={crop_end_sec}s, "
                    f"durata={duration_before:.3f}s"
                )
    
            crop_applied=bool(
                crop_edges
                and (
                    crop_start_sec>0
                    or crop_end_sec>0
                )
            )
    
            if crop_applied:
                raw=raw.copy().crop(
                    tmin=crop_start_sec,
                    tmax=duration_before-crop_end_sec,
                    include_tmax=True
                )
    
                print(
                    f"✂️ REST crop reale: rimossi "
                    f"{crop_start_sec:.3f}s iniziali e "
                    f"{crop_end_sec:.3f}s finali"
                )
    
                print(
                    f"   Durata: {duration_before:.3f}s → "
                    f"{raw.times[-1]:.3f}s"
                )
    
            elif preselect_edges:
                annotations=raw.annotations.copy()
            
                existing_descriptions=set(
                    str(x)
                    for x in annotations.description
                )
            
                if (
                    crop_start_sec>0
                    and "BAD_REST_START" not in existing_descriptions
                ):
                    annotations.append(
                        onset=0.0,
                        duration=crop_start_sec,
                        description="BAD_REST_START"
                    )
            
                if (
                    crop_end_sec>0
                    and "BAD_REST_END" not in existing_descriptions
                ):
                    annotations.append(
                        onset=duration_before-crop_end_sec,
                        duration=crop_end_sec,
                        description="BAD_REST_END"
                    )
            
                raw.set_annotations(annotations)
            
                print(
                    f"📌 REST bordi preselezionati come BAD: "
                    f"inizio={crop_start_sec:.3f}s, "
                    f"fine={crop_end_sec:.3f}s"
                )
            json_data["rest_crop_edges"]=crop_edges
            json_data["rest_preselect_edges_as_bad"]=preselect_edges
            json_data["rest_crop_applied"]=crop_applied
            json_data["rest_crop_start_sec"]=crop_start_sec
            json_data["rest_crop_end_sec"]=crop_end_sec
            json_data["rest_duration_before_crop_sec"]=duration_before
            json_data["rest_duration_after_crop_sec"]=float(
                raw.times[-1]
            )
            json_data["rest_n_times_after_crop"]=int(
                raw.n_times
            )
    
            json_data[
                "rest_edges_preselected_as_bad"
            ]=bool(
                preselect_edges
                and not crop_applied
                and (
                    crop_start_sec>0
                    or crop_end_sec>0
                )
            )
    
            print("🧠 EEG REST: continuo puro")
    
            events=None
            saveNote="REST_continuum"
    
            json_data["TEP_ID_events"]=saveNote
            json_data["epoching_mode"]="REST_continuum"
            json_data["sfreq"]=float(raw.info["sfreq"])
    
            save_layout_and_metadata(
                raw,
                saveNote
            )
    
            return raw,events,json_data
    
        events,event_id=mne.events_from_annotations(
            raw,
            verbose=False
        )
    
        if len(events)==0:
            raise ValueError(
                "Nessun evento trovato. "
                "Se i dati sono resting-state, "
                "imposta json_data['eeg_type']='rest'."
            )
    
        tm_code=np.unique(events[:,2])[0]
        events=events[events[:,2]==tm_code]
    
        saveNote="EDF_events"
    
        json_data["TEP_ID_events"]=saveNote
        json_data["sfreq"]=float(raw.info["sfreq"])
    
        save_layout_and_metadata(
            raw,
            saveNote
        )
    
        return raw,events,json_data

    # === CASO CHALFONT ===
    if 'Chalfont' in source:
        json_data['pulse_artifact_rej_timewindow_min'] = -0.002 * 1.5
        json_data['pulse_artifact_rej_timewindow_max'] = 0.008 * 1.5
        #json_data['detrend_minTimeWindowOffset'] = json_data['pulse_artifact_rej_timewindow_max']
        if dataType == 'VHDR':
            raw = mne.io.read_raw_brainvision(f'{fileName}.vhdr', eog=['VEOG', 'HEOG'], preload=True)
            try:
                raw.set_montage('easycap-M1', verbose=True)
            except ValueError as e:
                if 'channel positions not present' in str(e):
                    raw.set_channel_types({'65': 'misc', '66': 'misc'})
                    raw.set_montage('easycap-M1', on_missing='ignore', verbose=True)
                else:
                    raise
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            TMScode = 1015
            events = events[events[:, 2] == TMScode]
            saveNote = 'NGH_events'
            json_data['TEP_ID_events'] = saveNote
            json_data['sfreq'] = raw.info['sfreq']
            save_layout_and_metadata(raw, saveNote)
            return raw, events, json_data

    # === CASO UNIMI ===
    if 'UNIMI' in source:
        #json_data['pulse_artifact_rej_timewindow_min'] = -0.002
        #json_data['pulse_artifact_rej_timewindow_max'] = 0.008
        #json_data['detrend_minTimeWindowOffset'] = json_data['pulse_artifact_rej_timewindow_max']
        if dataType == 'VHDR':
            raw = mne.io.read_raw_brainvision(f'{fileName}.vhdr', eog=['VEOG', 'HEOG'], preload=True)
            raw.set_montage('easycap-M1', verbose=True)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            TMScode = 1128
            events = events[events[:, 2] == TMScode]
            saveNote = 'MI_events'
            if Path(fileName).stem == 'prova_Betta_0002':
                shift_sec = - ((0.0001 * 4) + 0.002)
                shift_samples = int(shift_sec * raw.info['sfreq'])
                events[:, 0] += shift_samples
            json_data['TEP_ID_events'] = saveNote
            json_data['sfreq'] = raw.info['sfreq']
            save_layout_and_metadata(raw, saveNote)
            return raw, events, json_data
            
        if dataType == 'ASCII':
            df = loadASCII(fileName, fileName)
            data = df.values.T 
            ASCII_events = np.where(df['MK'] == df['MK'].unique()[1])[0]
            raw = loadEDF(json_data, fileName)
            rename_dict = {'T3': 'FT7', 'T4': 'FT8', 'T5': 'TP7', 'T6': 'TP8'}
            raw.rename_channels(rename_dict)
            raw.set_montage('easycap-M1', verbose=True)
            raw._data = data[:len(raw.ch_names)]
            json_data['ch_names'] = raw.ch_names
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            TMScode = np.unique(events[:, 2])[0]
            events = events[events[:, 2] == TMScode]
            saveNote = 'EDF_events'
            json_data['TEP_ID_events'] = saveNote
            json_data['sfreq'] = raw.info['sfreq']

            save_layout_and_metadata(raw, saveNote)
            return raw, events, json_data

    raise ValueError(f"⚠️ Origine dati '{source}' non riconosciuta o mal configurata.")

def load_and_prepare_raw_data_20072026(fileName, json_data, experiment_dir, sub):
    import os
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import mne
    from tmspath_utils_rest import loadASCII, loadEDF
    json_data, experiment_dir, sub = directorySetup(json_data)
    
    experiment_dir=json_data["experiment_dir"]
    sub=json_data["subject"]

    if 'sourceData' not in json_data:
        raise KeyError("⚠️ 'sourceData' non è definito in json_data.")

    source = json_data['sourceData']
    dataType = json_data['dataType']

    json_data['pulse_artifact_rej_smoothingvalue'] = 0.002
    saveNote = ''
    raw, events = None, None
  
    def save_layout_and_metadata(raw,note):
        import os
        import json
        import matplotlib.pyplot as plt
        from pathlib import Path
    
        basic_dir=Path(experiment_dir).expanduser().resolve()/"1.basic"
        basic_dir.mkdir(parents=True,exist_ok=True)
    
        safe_note=str(note).replace("\\","_").replace("/","_").replace(":","_")
        safe_sub=str(sub).replace("\\","_").replace("/","_").replace(":","_")
    
        layout_path=basic_dir/f"{safe_sub}_{safe_note}_scalplayout.png"
    
        print(f"[DEBUG] Saving layout to: {layout_path}")
        print(f"[DEBUG] Parent exists: {layout_path.parent.exists()}")
    
        fig=raw.plot_sensors(show_names=True,show=False)
        fig.savefig(str(layout_path),dpi=300,bbox_inches="tight")
        plt.close(fig)
    
        pars_path=Path(experiment_dir).expanduser().resolve()/f"{safe_sub}_pars.json"
        with open(pars_path,"w") as json_file:
            json.dump(make_json_serializable(json_data),json_file,indent=4,sort_keys=True)

    # === CASO SIMS ===
    if source == 'SIMS':
        json_data['pulse_artifact_rej_timewindow_min'] = -0.002  # not used in sims
        json_data['pulse_artifact_rej_timewindow_max'] = 0.008  # not used in sims
        #json_data['detrend_minTimeWindowOffset'] = json_data['pulse_artifact_rej_timewindow_max'] # not used in sims

        # fileName = f"{json_data['mainDir']}\\{json_data['subject']}.fif"
        epochs = mne.read_epochs(fileName, preload=True)
        basicPlots(epochs, json_data, experiment_dir, sub, key='epochsOK', subPath='1.basic')
        with open(Path(experiment_dir) / '7.pkls' / f'{sub}_epochsOK.pkl', 'wb') as f:
            pickle.dump(epochs, f)
        json_data['sfreq'] = epochs.info['sfreq']
        json_data['r_sfreq'] = 512
        raw = epochs
        events = raw.events
        json_data['TEP_ID_events'] = 'no_events'
        save_layout_and_metadata(raw, 'no_events')
        return raw, events, json_data

    """
    # === CASO MAYER ===
    if source == 'MAYER':
        # json_data['pulse_artifact_rej_timewindow_min'] = -0.002
        # json_data['pulse_artifact_rej_timewindow_max'] = 0.008
        # json_data['detrend_minTimeWindowOffset'] = json_data['pulse_artifact_rej_timewindow_max']
        if dataType == 'ASCII':
            df = loadASCII(fileName, fileName)
            data = df.values.T * 1e-6
            ASCII_events = np.where(df['MK'] == df['MK'].unique()[1])[0]
            raw = loadEDF(json_data, fileName)
            do_runame=False
            if do_runame:
                rename_dict = {'T3': 'FT7', 'T4': 'FT8', 'T5': 'TP7', 'T6': 'TP8'}
                raw.rename_channels(rename_dict)
            raw.set_montage('easycap-M1', verbose=True)
            raw._data = data[:len(raw.ch_names)]
            json_data['ch_names'] = raw.ch_names
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            TMScode = np.unique(events[:, 2])[0]
            events = events[events[:, 2] == TMScode]
            saveNote = 'EDF_events'
            json_data['TEP_ID_events'] = saveNote
            json_data['sfreq'] = raw.info['sfreq']
            save_layout_and_metadata(raw, saveNote)
            return raw, events, json_data
    """
    # === CASO MAYER === NEW (07/07/2026) update for rest
    if source == 'MAYER':
        if dataType == 'ASCII':
            df = loadASCII(fileName, fileName)
            data = df.values.T * 1e-6
            ASCII_events = np.where(df['MK'] == df['MK'].unique()[1])[0] if json_data['eeg_type'] != 'rest' else None
            raw = loadEDF(json_data, fileName)
            rename_case = {
                'FPz': 'Fpz',
                'Fc1': 'FC1',
                'Fc2': 'FC2',
                'Af4': 'AF4',
                'Fc5': 'FC5',
                'Fc6': 'FC6',
                'Po3': 'PO3',
                'Tp9': 'TP9',
                'Tp10': 'TP10',
                'Po4': 'PO4',
                'Cp1': 'CP1',
                'Cp2': 'CP2',
                'Cp5': 'CP5',
                'Cp6': 'CP6'
            }
            raw.rename_channels({
                k: v for k, v in rename_case.items()
                if k in raw.ch_names
            })
            ch_types = {}
            if 'TM' in raw.ch_names:
                ch_types['TM'] = 'misc'
            if 'MK' in raw.ch_names:
                ch_types['MK'] = 'stim'
            if ch_types:
                raw.set_channel_types(ch_types)

            df = pd.read_csv("./outputFile.csv")
            scale = 0.095-0.005
            ch_pos = {}
            
            for _, row in df.iterrows():
                ch = row["labels"]
                if ch not in raw.ch_names:
                    continue
            
                x = -float(row["Y"]) * scale
                y =  float(row["X"]) * scale
                z =  float(row["Z"]) * scale
            
                if ch in ["TP9", "TP10"]:
                    r = np.sqrt(x**2 + y**2)
                    max_radius_tp = 0.086-0.050*0.50   # prova 0.088, 0.086, 0.084
                    if r > max_radius_tp:
                        shrink = max_radius_tp / r
                        x *= shrink
                        y *= shrink
            
                ch_pos[ch] = np.array([x, y, z])
            
            print("TP9:", ch_pos["TP9"])
            print("TP10:", ch_pos["TP10"])
            
            montage = mne.channels.make_dig_montage(
                ch_pos=ch_pos,
                coord_frame="head"
            )
            raw.set_montage(montage, on_missing="warn")

            """
            #montage_ref = mne.channels.make_standard_montage('standard_1020')
            desired_channels = montage_new.ch_names
            #ref_positions = montage_ref.get_positions()['ch_pos']
            ref_positions = montage_new.get_positions()['ch_pos']
            desired_positions = {
                ch: ref_positions[ch]
                for ch in desired_channels
                if ch in ref_positions
            }
            missing = [ch for ch in desired_channels if ch not in ref_positions]
            print("Missing from standard_1020:", missing)
            montage = mne.channels.make_dig_montage(
                ch_pos=desired_positions,
                coord_frame='head'
            )
            raw.set_montage(montage, on_missing='warn')
            """
            # raw._data = data[:len(raw.ch_names)]
            # json_data['ch_names'] = raw.ch_names
            # events, event_id = mne.events_from_annotations(raw, verbose=False)
            # TMScode = np.unique(events[:, 2])[0]
            #events = events[events[:, 2] == TMScode]
            # saveNote = 'EDF_events'
            # json_data['TEP_ID_events'] = saveNote
            # json_data['sfreq'] = raw.info['sfreq']
            #save_layout_and_metadata(raw, saveNote)

            raw._data = data[:len(raw.ch_names)]
            json_data['ch_names'] = raw.ch_names
            json_data['sfreq'] = raw.info['sfreq']
            eeg_type = json_data.get('eeg_type', json_data.get('EEGTYPE', 'tms')).lower()
            if eeg_type == 'rest':
                print("🧠 EEG REST: continuo puro")
                events=None
                saveNote="REST_continuum"
                json_data["TEP_ID_events"]=saveNote
                json_data["epoching_mode"]="REST_continuum"
                save_layout_and_metadata(raw,saveNote)
                return raw,events,json_data
                
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            if len(events) == 0:
                raise ValueError(
                    "Nessun evento trovato. Se questi dati sono resting-state, imposta json_data['eeg_type']='rest'."
                )
            TMScode = np.unique(events[:, 2])[0]
            events = events[events[:, 2] == TMScode]
            saveNote = 'EDF_events'
            json_data['TEP_ID_events'] = saveNote
            save_layout_and_metadata(raw, saveNote)
            return raw, events, json_data
        
        return raw, events, json_data

    # === CASO CHALFONT ===
    if 'Chalfont' in source:
        json_data['pulse_artifact_rej_timewindow_min'] = -0.002 * 1.5
        json_data['pulse_artifact_rej_timewindow_max'] = 0.008 * 1.5
        #json_data['detrend_minTimeWindowOffset'] = json_data['pulse_artifact_rej_timewindow_max']
        if dataType == 'VHDR':
            raw = mne.io.read_raw_brainvision(f'{fileName}.vhdr', eog=['VEOG', 'HEOG'], preload=True)
            try:
                raw.set_montage('easycap-M1', verbose=True)
            except ValueError as e:
                if 'channel positions not present' in str(e):
                    raw.set_channel_types({'65': 'misc', '66': 'misc'})
                    raw.set_montage('easycap-M1', on_missing='ignore', verbose=True)
                else:
                    raise
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            TMScode = 1015
            events = events[events[:, 2] == TMScode]
            saveNote = 'NGH_events'
            json_data['TEP_ID_events'] = saveNote
            json_data['sfreq'] = raw.info['sfreq']
            save_layout_and_metadata(raw, saveNote)
            return raw, events, json_data

    # === CASO UNIMI ===
    if 'UNIMI' in source:
        #json_data['pulse_artifact_rej_timewindow_min'] = -0.002
        #json_data['pulse_artifact_rej_timewindow_max'] = 0.008
        #json_data['detrend_minTimeWindowOffset'] = json_data['pulse_artifact_rej_timewindow_max']
        if dataType == 'VHDR':
            raw = mne.io.read_raw_brainvision(f'{fileName}.vhdr', eog=['VEOG', 'HEOG'], preload=True)
            raw.set_montage('easycap-M1', verbose=True)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            TMScode = 1128
            events = events[events[:, 2] == TMScode]
            saveNote = 'MI_events'
            if Path(fileName).stem == 'prova_Betta_0002':
                shift_sec = - ((0.0001 * 4) + 0.002)
                shift_samples = int(shift_sec * raw.info['sfreq'])
                events[:, 0] += shift_samples
            json_data['TEP_ID_events'] = saveNote
            json_data['sfreq'] = raw.info['sfreq']
            save_layout_and_metadata(raw, saveNote)
            return raw, events, json_data
            
        if dataType == 'ASCII':
            df = loadASCII(fileName, fileName)
            data = df.values.T 
            ASCII_events = np.where(df['MK'] == df['MK'].unique()[1])[0]
            raw = loadEDF(json_data, fileName)
            rename_dict = {'T3': 'FT7', 'T4': 'FT8', 'T5': 'TP7', 'T6': 'TP8'}
            raw.rename_channels(rename_dict)
            raw.set_montage('easycap-M1', verbose=True)
            raw._data = data[:len(raw.ch_names)]
            json_data['ch_names'] = raw.ch_names
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            TMScode = np.unique(events[:, 2])[0]
            events = events[events[:, 2] == TMScode]
            saveNote = 'EDF_events'
            json_data['TEP_ID_events'] = saveNote
            json_data['sfreq'] = raw.info['sfreq']

            save_layout_and_metadata(raw, saveNote)
            return raw, events, json_data

    raise ValueError(f"⚠️ Origine dati '{source}' non riconosciuta o mal configurata.")


def make_rest_events(raw,json_data):
    import numpy as np

    sfreq=raw.info["sfreq"]

    tmin=float(json_data.get("epochs_timewindow_min",-0.1))
    tmax=float(json_data.get("epochs_timewindow_max",0.4))
    spacing=float(json_data.get("rest_event_spacing",1.0))
    event_id=int(json_data.get("rest_event_id",999))

    first_time=float(json_data.get("rest_first_event_time",abs(tmin)+0.5))
    last_time=raw.times[-1]-tmax-0.5

    if last_time<=first_time:
        raise ValueError(
            f"REST troppo corto per eventi fittizi: first={first_time}, last={last_time}, durata={raw.times[-1]}"
        )

    event_times=np.arange(first_time,last_time,spacing)
    event_samples=(event_times*sfreq).astype(int)

    events=np.column_stack([
        event_samples,
        np.zeros(len(event_samples),dtype=int),
        np.full(len(event_samples),event_id,dtype=int)
    ])

    json_data["rest_event_spacing"]=float(spacing)
    json_data["rest_event_id"]=int(event_id)
    json_data["rest_n_events"]=int(len(events))
    json_data["rest_first_event_time"]=float(event_times[0])
    json_data["rest_last_event_time"]=float(event_times[-1])
    json_data["epoching_mode"]="REST_fake_events"

    print(f"🧠 REST fake events: {len(events)} eventi, spacing={spacing}s, event_id={event_id}")

    return events


def ICAcontinuum_manual_from_computeBasicSteps(
    basic_output,
    experiment_dir=None,
    sub=None,
    label_prob_threshold=0.80,
    autoReject=True,
    manualCheck=True,
    show_components=True,
    show_all_properties=False,
    property_picks=None,
    save=True
):
    import matplotlib
    matplotlib.use("Qt5Agg")

    import mne
    import numpy as np
    import pickle
    import json
    import matplotlib.pyplot as plt
    from pathlib import Path
    from datetime import datetime
    from mne.preprocessing import ICA
    from mne_icalabel import label_components

    raw_clean,epochs_5s,segments_5s,json_data=basic_output

    if experiment_dir is None:
        experiment_dir=json_data["experiment_dir"]

    if sub is None:
        sub=json_data["subject"]

    experiment_dir=str(Path(experiment_dir).expanduser().resolve())
    json_data["experiment_dir"]=experiment_dir

    paths=rest_paths(experiment_dir)

    raw_ica=raw_clean.copy().pick("eeg")

    bad_channels=[
        ch for ch in json_data.get("bad_channels",[])
        if ch in raw_ica.ch_names
    ]

    raw_ica.info["bads"]=bad_channels

    picks_ica=mne.pick_types(
        raw_ica.info,
        eeg=True,
        exclude="bads"
    )

    print("Bad channels esclusi da ICA:",bad_channels)
    print("Canali usati per ICA:",len(picks_ica))

    raw_ica.set_eeg_reference("average")

    n_components=len(picks_ica)-1

    ica=ICA(
        n_components=n_components,
        method="fastica",
        random_state=42,
        max_iter="auto"
    )

    ica.fit(
        raw_ica,
        picks=picks_ica,
        reject_by_annotation=True
    )

    print(ica)

    raw_ica_fit=raw_ica.copy().pick(
        list(ica.ch_names)
    )

    ic_labels=label_components(
        raw_ica_fit,
        ica,
        method="iclabel"
    )

    labels=ic_labels["labels"]
    probs=ic_labels["y_pred_proba"]

    for i,label in enumerate(labels):
        prob=float(np.max(np.atleast_1d(probs[i])))
        print(i,label,prob)

    artifact_tags=[
        "eye blink",
        "muscle artifact",
        "heart beat",
        "line noise",
        "channel noise"
    ]

    iclabel_suggested=[]
    
    for i,label in enumerate(labels):
        probability=float(
            np.max(
                np.atleast_1d(
                    probs[i]
                )
            )
        )
    
        if (
            label in artifact_tags
            and probability>=label_prob_threshold
        ):
            iclabel_suggested.append(i)
    
    print(
        "Componenti artefattuali suggerite da ICLabel:",
        iclabel_suggested
    )
    
    if autoReject:
        auto_exclude=iclabel_suggested.copy()
    else:
        auto_exclude=[]
    
    ica.exclude=[]

    if manualCheck:
        final_exclude,kept,topo_selected,psd_selected=(
            select_ica_components_topo_psd_gui(
                ica=ica,
                raw_ica=raw_ica_fit,
                labels=labels,
                probabilities=probs,
                preselected=iclabel_suggested,
                combine_mode="PSD_FINAL",
                ncols_topo=8,
                ncols_psd=4,
                nrows_psd=3,
                fmin=0.5,
                fmax=50
            )
        )
        
        ica.exclude=[
            int(x)
            for x in final_exclude
        ]
    
        print("Componenti selezionate da topografia:",topo_selected)
        print("Componenti selezionate da PSD:",psd_selected)
        print("Componenti finali escluse:",ica.exclude)
        print("Componenti tenute:",kept)

    if show_components:
        n_comp=ica.n_components_

        for start in range(0,n_comp,20):
            stop=min(start+20,n_comp)

            ica.plot_components(
                picks=list(range(start,stop)),
                inst=raw_ica_fit,
                ch_type="eeg",
                plot_std=True,
                psd_args=dict(fmin=0.5,fmax=50),
                show=True
            )

    if property_picks is not None:
        for ic in property_picks:
            print("IC",ic,labels[ic],float(np.max(np.atleast_1d(probs[ic]))))

            ica.plot_properties(
                inst=raw_ica_fit,
                picks=[ic],
                dB=False,
                plot_std=True,
                log_scale=True,
                psd_args=dict(fmin=0.5,fmax=50),
                reject_by_annotation=True,
                show=True
            )

    if show_all_properties:
        for ic in range(ica.n_components_):
            print("IC",ic,labels[ic],float(np.max(np.atleast_1d(probs[ic]))))

            ica.plot_properties(
                inst=raw_ica,
                picks=[ic],
                dB=False,
                plot_std=True,
                log_scale=True,
                psd_args=dict(fmin=0.5,fmax=50),
                reject_by_annotation=True,
                show=True
            )

    manual_exclude=[]
    
    if not manualCheck:
        manual=input("Componenti da escludere, separate da virgola: ")
    
        if manual.strip()!="":
            manual_exclude=[
                int(x.strip())
                for x in manual.split(",")
                if x.strip().isdigit()
            ]
        else:
            manual_exclude=[]
    
        ica.exclude=sorted(set(list(ica.exclude)+manual_exclude))
    
    print("Componenti finali escluse:",ica.exclude)

    postICA_raw_continuum=raw_ica.copy()

    ica.apply(postICA_raw_continuum)

    postICA_raw_continuum.plot(
        n_channels=min(32,len(postICA_raw_continuum.ch_names)),
        scalings={"eeg":50e-6}
    )

    postICA_final=postICA_raw_continuum.copy()

    postICA_final.info["bads"]=bad_channels

    if len(bad_channels)>0:
        postICA_final.interpolate_bads(reset_bads=True)

    postICA_final.set_eeg_reference("average")

    timestamp=datetime.now().strftime("%Y%m%d%H%M%S")

    json_data["ICA_fit_space"]="raw_continuum"
    json_data["ICA_components_tot"]=int(ica.n_components_)
    json_data["ICA_autoExcludedComponents"]=[int(x) for x in auto_exclude]
    json_data["ICA_manualAddedComponents"]=[int(x) for x in manual_exclude]
    json_data["ICA_excludedComponents"]=[int(x) for x in ica.exclude]
    json_data["ICA_labels"]=[str(x) for x in labels]
    json_data["ICA_label_probabilities"]=[
        float(np.max(np.atleast_1d(p)))
        for p in probs
    ]
    json_data["ICA_bad_channels_excluded_from_fit"]=bad_channels
    json_data["channel_interpolation_timing"]="after_ica"
    json_data["reference_after_ica"]="average"
    json_data["ICA_timestamp"]=timestamp
    if manualCheck:
        json_data["ICA_topoSelectedComponents"]=[int(x) for x in topo_selected]
        json_data["ICA_psdSelectedComponents"]=[int(x) for x in psd_selected]
        json_data["ICA_manualCombinationMode"]="OR"
        json_data["ICA_keptComponents"]=[int(x) for x in kept]
    if save:
        ica_path=paths["pkls"]/f"{timestamp}_{sub}_ica_model_continuum.pkl"
        raw_path=paths["pkls"]/f"{timestamp}_{sub}_postICA_raw_continuum.pkl"
        final_path=paths["pkls"]/f"{timestamp}_{sub}_postICA_final.pkl"

        with open(ica_path,"wb") as f:
            pickle.dump(ica,f)

        with open(raw_path,"wb") as f:
            pickle.dump(postICA_raw_continuum,f)

        with open(final_path,"wb") as f:
            pickle.dump(postICA_final,f)

        postICA_final.save(
            paths["pkls"]/f"{timestamp}_{sub}_postICA_final.fif",
            overwrite=True
        )

        with open(Path(experiment_dir)/f"{sub}_pars.json","w") as f:
            json.dump(make_json_serializable(json_data),f,indent=4,sort_keys=True)

        print("Salvato:")
        print(ica_path)
        print(raw_path)
        print(final_path)

    return postICA_final,postICA_raw_continuum,ica,json_data


def run_rest_ica(
    raw_clean,
    json_data,
    experiment_dir,
    sub
):
    from pathlib import Path
    import json
    import matplotlib.pyplot as plt

    if not json_data.get("do_ica",False):
        print("⏭️ ICA REST disattivata")
        return raw_clean,None,None,None,json_data

    experiment_dir=Path(
        json_data.get(
            "experiment_dir",
            experiment_dir
        )
    ).expanduser().resolve()

    json_data["experiment_dir"]=str(
        experiment_dir
    )

    plt.close("all")

    temp_raw_clean=raw_clean.copy().resample(
        sfreq=float(
            json_data["r_sfreq"]
        )
    )

    (
        postICA_final,
        postICA_raw_continuum,
        ica,
        json_data
    )=ICAcontinuum_manual_from_computeBasicSteps(
        basic_output=(
            temp_raw_clean,
            None,
            None,
            json_data
        ),
        experiment_dir=str(experiment_dir),
        sub=sub,
        label_prob_threshold=float(
            json_data.get(
                "do_label_prob_threshold",
                0.80
            )
        ),
        autoReject=bool(
            json_data.get(
                "do_ica_automaticRej",
                True
            )
        ),
        manualCheck=bool(
            json_data.get(
                "do_ica_manualCheck",
                True
            )
        ),
        show_components=False,
        show_all_properties=False,
        property_picks=None,
        save=True
    )

    plt.close("all")

    paths=rest_paths(
        experiment_dir
    )

    timestamp=json_data.get(
        "ICA_timestamp",
        "no_timestamp"
    )

    if timestamp is None or str(timestamp).strip()=="":
        timestamp="no_timestamp"

    postica_dir=(
        Path(paths["postICA"])
        /str(timestamp)
    ).expanduser().resolve()

    removed_dir=(
        postica_dir
        /"removed_components"
    )

    kept_dir=(
        postica_dir
        /"kept_components"
    )

    components_dir=(
        postica_dir
        /"components_batches"
    )

    for directory in [
        postica_dir,
        removed_dir,
        kept_dir,
        components_dir
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    print(
        "Saving postICA outputs to:",
        postica_dir
    )

    json_data["postICA_dir"]=str(
        postica_dir
    )

    def save_fig_safe(
        fig,
        path,
        dpi=300
    ):
        path=Path(
            path
        ).expanduser().resolve()

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if isinstance(fig,(list,tuple)):
            saved_paths=[]

            for index,item in enumerate(fig):
                indexed_path=path.with_name(
                    f"{path.stem}_{index}{path.suffix}"
                )

                item.savefig(
                    indexed_path,
                    dpi=dpi,
                    bbox_inches="tight"
                )

                plt.close(item)

                saved_paths.append(
                    str(indexed_path)
                )

            return saved_paths

        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight"
        )

        plt.close(fig)

        return str(path)

    def save_psd_plot(
        raw_obj,
        label
    ):
        psd=raw_obj.compute_psd(
            method="welch",
            fmin=float(
                json_data.get(
                    "l_freq",
                    0.5
                )
            ),
            fmax=float(
                json_data.get(
                    "h_freq",
                    45
                )
            ),
            reject_by_annotation=True,
            n_per_seg=int(
                json_data.get(
                    "rest_psd_n_per_seg",
                    2000
                )
            ),
            n_overlap=int(
                json_data.get(
                    "rest_psd_n_overlap",
                    0
                )
            ),
            n_fft=int(
                json_data.get(
                    "rest_psd_n_fft",
                    2048
                )
            )
        )

        fig=psd.plot(
            dB=True,
            xscale="log",
            average=True,
            show=False
        )

        out_path=(
            postica_dir
            /f"{sub}_{label}_PSD.png"
        )

        saved_path=save_fig_safe(
            fig,
            out_path,
            dpi=300
        )

        return psd,saved_path

    (
        psd_postICA_raw,
        psd_postICA_raw_path
    )=save_psd_plot(
        postICA_raw_continuum,
        "postICA_raw_continuum"
    )

    (
        psd_postICA_final,
        psd_postICA_final_path
    )=save_psd_plot(
        postICA_final,
        "postICA_final"
    )

    json_data[
        "postICA_raw_continuum_PSD_png"
    ]=psd_postICA_raw_path

    json_data[
        "postICA_final_PSD_png"
    ]=psd_postICA_final_path

    save_preview_plots=bool(
        json_data.get(
            "save_postICA_preview_plots",
            True
        )
    )

    if save_preview_plots:
        fig=postICA_raw_continuum.plot(
            n_channels=min(
                32,
                len(
                    postICA_raw_continuum.ch_names
                )
            ),
            duration=float(
                json_data.get(
                    "ica_preview_duration",
                    20
                )
            ),
            scalings={
                "eeg":float(
                    json_data.get(
                        "rest_gui_scaling",
                        50e-6
                    )
                )
            },
            show=False,
            block=False
        )

        save_fig_safe(
            fig,
            postica_dir
            /f"{sub}_postICA_raw_continuum_preview.png",
            dpi=300
        )

        fig=postICA_final.plot(
            n_channels=min(
                32,
                len(
                    postICA_final.ch_names
                )
            ),
            duration=float(
                json_data.get(
                    "ica_preview_duration",
                    20
                )
            ),
            scalings={
                "eeg":float(
                    json_data.get(
                        "rest_gui_scaling",
                        50e-6
                    )
                )
            },
            show=False,
            block=False
        )

        save_fig_safe(
            fig,
            postica_dir
            /f"{sub}_postICA_final_preview.png",
            dpi=300
        )

    raw_ica_for_plots=(
        temp_raw_clean
        .copy()
        .pick("eeg")
        .pick(
            list(
                ica.ch_names
            )
        )
    )

    save_component_batches=bool(
        json_data.get(
            "save_ica_component_batches",
            True
        )
    )

    if save_component_batches:
        for start in range(
            0,
            ica.n_components_,
            20
        ):
            stop=min(
                start+20,
                ica.n_components_
            )

            try:
                figs=ica.plot_components(
                    picks=list(
                        range(
                            start,
                            stop
                        )
                    ),
                    inst=raw_ica_for_plots,
                    ch_type="eeg",
                    plot_std=True,
                    psd_args={
                        "fmin":0.5,
                        "fmax":50
                    },
                    show=False
                )

                if not isinstance(
                    figs,
                    list
                ):
                    figs=[figs]

                for index,fig in enumerate(figs):
                    save_fig_safe(
                        fig,
                        components_dir
                        /(
                            f"{sub}_ICA_components_"
                            f"{start:03d}_{stop:03d}_"
                            f"{index}.png"
                        ),
                        dpi=300
                    )

            except Exception as error:
                print(
                    "⚠️ Could not save ICA components "
                    f"{start}-{stop}: {error}"
                )

            finally:
                plt.close("all")

    excluded=sorted(
        int(component)
        for component in ica.exclude
    )

    kept=[
        component
        for component in range(
            ica.n_components_
        )
        if component not in excluded
    ]

    json_data[
        "ICA_excludedComponents"
    ]=excluded

    json_data[
        "ICA_keptComponents"
    ]=kept

    def save_component_properties(
        component_indices,
        output_dir,
        tag
    ):
        for component in component_indices:
            try:
                figs=ica.plot_properties(
                    raw_ica_for_plots,
                    picks=[component],
                    psd_args={
                        "fmin":0.5,
                        "fmax":50
                    },
                    reject_by_annotation=True,
                    show=False
                )

                if not isinstance(
                    figs,
                    list
                ):
                    figs=[figs]

                for index,fig in enumerate(figs):
                    save_fig_safe(
                        fig,
                        output_dir
                        /(
                            f"IC{component:03d}_"
                            f"{tag}_{index}.png"
                        ),
                        dpi=200
                    )

            except Exception as error:
                print(
                    f"⚠️ Could not save {tag} "
                    f"IC {component}: {error}"
                )

            finally:
                plt.close("all")

    save_component_properties_plots=bool(
        json_data.get(
            "save_ica_component_properties",
            True
        )
    )

    if save_component_properties_plots:
        save_component_properties(
            excluded,
            removed_dir,
            "removed"
        )

        save_component_properties(
            kept,
            kept_dir,
            "kept"
        )

    json_data[
        "postICA_figures_saved"
    ]=True

    json_data[
        "postICA_removed_components_dir"
    ]=str(
        removed_dir
    )

    json_data[
        "postICA_kept_components_dir"
    ]=str(
        kept_dir
    )

    json_data[
        "postICA_components_batches_dir"
    ]=str(
        components_dir
    )

    with open(
        experiment_dir/f"{sub}_pars.json",
        "w"
    ) as file:
        json.dump(
            make_json_serializable(
                json_data
            ),
            file,
            indent=4,
            sort_keys=True
        )

    raw_clean_afterICA=(
        postICA_final
        .copy()
        .resample(
            sfreq=float(
                json_data["sfreq"]
            )
        )
    )

    plt.close("all")

    return (
        raw_clean_afterICA,
        postICA_final,
        postICA_raw_continuum,
        ica,
        json_data
    )
    

def _component_grid_selector(fig_builder, n_components, title="Select components", ncols=5):
    import math
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    selected=set()
    page=0
    per_page=ncols*2
    n_pages=math.ceil(n_components/per_page)

    state={"done":False}

    def draw_page():
        fig.clf()
        start=page*per_page
        stop=min(start+per_page,n_components)
        inds=list(range(start,stop))

        fig.suptitle(f"{title} | page {page+1}/{n_pages} | selected: {sorted(selected)}", fontsize=12)

        axes=[]
        nrows=2
        for k,ic in enumerate(inds):
            ax=fig.add_subplot(nrows,ncols,k+1)
            fig_builder(ax,ic)

            if ic in selected:
                for spine in ax.spines.values():
                    spine.set_color("red")
                    spine.set_linewidth(3)
            else:
                for spine in ax.spines.values():
                    spine.set_color("black")
                    spine.set_linewidth(1)

            ax.set_title(f"IC {ic}", fontsize=10)
            ax._ic_index=ic
            axes.append(ax)

        ax_prev=fig.add_axes([0.10,0.01,0.10,0.05])
        ax_next=fig.add_axes([0.22,0.01,0.10,0.05])
        ax_done=fig.add_axes([0.78,0.01,0.12,0.05])

        btn_prev=Button(ax_prev,"Prev")
        btn_next=Button(ax_next,"Next")
        btn_done=Button(ax_done,"Done")

        def prev(event):
            nonlocal page
            if page>0:
                page-=1
                draw_page()
                fig.canvas.draw_idle()

        def next_(event):
            nonlocal page
            if page<n_pages-1:
                page+=1
                draw_page()
                fig.canvas.draw_idle()

        def done(event):
            state["done"]=True
            plt.close(fig)

        btn_prev.on_clicked(prev)
        btn_next.on_clicked(next_)
        btn_done.on_clicked(done)

        def onclick(event):
            if event.inaxes is None:
                return
            if hasattr(event.inaxes,"_ic_index"):
                ic=event.inaxes._ic_index
                if ic in selected:
                    selected.remove(ic)
                else:
                    selected.add(ic)
                draw_page()
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", onclick)
        fig.tight_layout(rect=[0,0.08,1,0.95])

    fig=plt.figure(figsize=(16,8))
    draw_page()
    plt.show(block=True)

    return sorted(selected)


def _make_topomap_builder(ica,raw_ica):
    import mne

    ica_ch_names=list(ica.ch_names)

    ica_picks=mne.pick_channels(
        raw_ica.ch_names,
        include=ica_ch_names,
        ordered=True
    )

    ica_info=mne.pick_info(
        raw_ica.info,
        sel=ica_picks,
        copy=True
    )

    components=ica.get_components()

    if components.shape[0]!=len(ica_info.ch_names):
        raise ValueError(
            f"Mismatch ICA/topomap: "
            f"components={components.shape[0]}, "
            f"info={len(ica_info.ch_names)}"
        )

    def builder(ax,ic):
        mne.viz.plot_topomap(
            components[:,ic],
            ica_info,
            axes=ax,
            show=False,
            contours=6
        )

    return builder

def _make_psd_builder(ica, raw_ica, fmin=0.5, fmax=50):
    import numpy as np

    sources=ica.get_sources(raw_ica).get_data()
    sfreq=raw_ica.info["sfreq"]

    def compute_psd(sig):
        x=sig-np.mean(sig)
        n=len(x)
        freqs=np.fft.rfftfreq(n,d=1/sfreq)
        psd=(np.abs(np.fft.rfft(x))**2)/n
        mask=(freqs>=fmin)&(freqs<=fmax)
        return freqs[mask],psd[mask]

    cache={}
    for ic in range(sources.shape[0]):
        cache[ic]=compute_psd(sources[ic])

    def builder(ax, ic):
        freqs,psd=cache[ic]
        ax.plot(freqs,psd,lw=1)
        ax.set_xlim(fmin,fmax)
        ax.set_yscale("log")
        ax.set_xlabel("Hz",fontsize=8)
        ax.set_ylabel("PSD",fontsize=8)
        ax.tick_params(labelsize=8)

    return builder


def ICAcontinuum_topo_psd_selector(
    basic_output,
    experiment_dir=None,
    sub=None,
    combine_mode="OR",
    autoReject=False,
    label_prob_threshold=0.80,
    save=True
):
    import matplotlib
    matplotlib.use("Qt5Agg")

    import mne
    import numpy as np
    import pickle
    import json
    import matplotlib.pyplot as plt
    from pathlib import Path
    from datetime import datetime
    from mne.preprocessing import ICA
    from mne_icalabel import label_components

    raw_clean,epochs_5s,segments_5s,json_data=basic_output

    if experiment_dir is None:
        experiment_dir=json_data["experiment_dir"]
    if sub is None:
        sub=json_data["subject"]

    experiment_dir=str(Path(experiment_dir).expanduser().resolve())
    json_data["experiment_dir"]=experiment_dir

    paths=rest_paths(experiment_dir)

    timestamp=datetime.now().strftime("%Y%m%d%H%M%S")
    postica_dir=paths["postICA"]/timestamp
    postica_dir.mkdir(parents=True,exist_ok=True)

    overview_dir=Path(experiment_dir)/"CONTINUUM_ICA_OVERVIEWS"
    overview_dir.mkdir(parents=True,exist_ok=True)

    raw_ica=raw_clean.copy().pick("eeg")

    bad_channels=[ch for ch in json_data.get("bad_channels",[]) if ch in raw_ica.ch_names]
    raw_ica.info["bads"]=bad_channels

    picks_ica=mne.pick_types(raw_ica.info,eeg=True,exclude="bads")

    print("Bad channels esclusi da ICA:",bad_channels)
    print("Canali usati per ICA:",len(picks_ica))

    raw_ica.set_eeg_reference("average")

    n_components=len(picks_ica)-1

    ica=ICA(
        n_components=n_components,
        method="fastica",
        random_state=42,
        max_iter="auto"
    )

    ica.fit(
        raw_ica,
        picks=picks_ica,
        reject_by_annotation=True
    )

    print(ica)

    ic_labels=label_components(raw_ica,ica,method="iclabel")
    labels=ic_labels["labels"]
    probs=ic_labels["y_pred_proba"]

    for i,label in enumerate(labels):
        prob=float(np.max(np.atleast_1d(probs[i])))
        print(i,label,prob)

    auto_exclude=[]
    if autoReject:
        artifact_tags=["eye blink","muscle artifact","heart beat","line noise","channel noise"]
        for i,label in enumerate(labels):
            prob=float(np.max(np.atleast_1d(probs[i])))
            if label in artifact_tags and prob>=label_prob_threshold:
                auto_exclude.append(i)

    print("ICLabel suggests:",auto_exclude)

    topo_builder=_make_topomap_builder(ica,raw_ica)
    psd_builder=_make_psd_builder(ica,raw_ica,fmin=0.5,fmax=50)

    print("GUI 1: selezione tramite topografie")
    topo_selected=_component_grid_selector(
        topo_builder,
        n_components=ica.n_components_,
        title="Select ICA components from TOPOGRAPHY",
        ncols=5
    )

    print("Selezionate da topografia:",topo_selected)

    print("GUI 2: selezione tramite PSD")
    psd_selected=_component_grid_selector(
        psd_builder,
        n_components=ica.n_components_,
        title="Select ICA components from PSD",
        ncols=5
    )

    print("Selezionate da PSD:",psd_selected)

    topo_set=set(topo_selected)
    psd_set=set(psd_selected)
    auto_set=set(auto_exclude)

    mode=combine_mode.upper()
    if mode=="AND":
        manual_set=topo_set & psd_set
    else:
        manual_set=topo_set | psd_set

    final_exclude=sorted(auto_set | manual_set)
    kept=[ic for ic in range(ica.n_components_) if ic not in final_exclude]

    ica.exclude=final_exclude

    print("combine_mode:",mode)
    print("Final excluded:",final_exclude)
    print("Kept:",kept)

    postICA_raw_continuum=raw_ica.copy()
    ica.apply(postICA_raw_continuum)

    postICA_final=postICA_raw_continuum.copy()
    postICA_final.info["bads"]=bad_channels

    if len(bad_channels)>0:
        postICA_final.interpolate_bads(reset_bads=True)

    postICA_final.set_eeg_reference("average")

    json_data["ICA_fit_space"]="raw_continuum"
    json_data["ICA_components_tot"]=int(ica.n_components_)
    json_data["ICA_labels"]=[str(x) for x in labels]
    json_data["ICA_label_probabilities"]=[float(np.max(np.atleast_1d(p))) for p in probs]
    json_data["ICA_autoExcludedComponents"]=[int(x) for x in auto_exclude]
    json_data["ICA_topoSelectedComponents"]=[int(x) for x in topo_selected]
    json_data["ICA_psdSelectedComponents"]=[int(x) for x in psd_selected]
    json_data["ICA_manualCombinationMode"]=mode
    json_data["ICA_excludedComponents"]=[int(x) for x in final_exclude]
    json_data["ICA_keptComponents"]=[int(x) for x in kept]
    json_data["ICA_bad_channels_excluded_from_fit"]=bad_channels
    json_data["channel_interpolation_timing"]="after_ica"
    json_data["reference_after_ica"]="average"
    json_data["ICA_timestamp"]=timestamp

    if save:
        pkl_dir=paths["pkls"]
        pkl_dir.mkdir(parents=True,exist_ok=True)

        with open(pkl_dir/f"{timestamp}_{sub}_ica_model_continuum.pkl","wb") as f:
            pickle.dump(ica,f)

        with open(pkl_dir/f"{timestamp}_{sub}_postICA_raw_continuum.pkl","wb") as f:
            pickle.dump(postICA_raw_continuum,f)

        with open(pkl_dir/f"{timestamp}_{sub}_postICA_final.pkl","wb") as f:
            pickle.dump(postICA_final,f)

        postICA_final.save(
            pkl_dir/f"{timestamp}_{sub}_postICA_final.fif",
            overwrite=True
        )

        removed_dir=postica_dir/"removed_components"
        kept_dir=postica_dir/"kept_components"
        removed_dir.mkdir(parents=True,exist_ok=True)
        kept_dir.mkdir(parents=True,exist_ok=True)

        # salva immagini "ricche" tipo plot_properties nel 3.postICA
        for ic in final_exclude:
            figs=ica.plot_properties(
                raw_ica,
                picks=[ic],
                psd_args=dict(fmin=0.5,fmax=50),
                reject_by_annotation=True,
                show=False
            )
            if not isinstance(figs,list):
                figs=[figs]
            for j,fig in enumerate(figs):
                fig.savefig(removed_dir/f"IC{ic:03d}_removed_{labels[ic]}_{j}.png",dpi=150,bbox_inches="tight")
                plt.close(fig)

        for ic in kept:
            figs=ica.plot_properties(
                raw_ica,
                picks=[ic],
                psd_args=dict(fmin=0.5,fmax=50),
                reject_by_annotation=True,
                show=False
            )
            if not isinstance(figs,list):
                figs=[figs]
            for j,fig in enumerate(figs):
                fig.savefig(kept_dir/f"IC{ic:03d}_kept_{labels[ic]}_{j}.png",dpi=150,bbox_inches="tight")
                plt.close(fig)

        # salva overview generali "fuori cartella" 3.postICA
        n_comp=ica.n_components_
        for start in range(0,n_comp,20):
            stop=min(start+20,n_comp)

            fig=ica.plot_components(
                picks=list(range(start,stop)),
                inst=raw_ica,
                ch_type="eeg",
                plot_std=True,
                psd_args=dict(fmin=0.5,fmax=50),
                show=False
            )
            if isinstance(fig,list):
                for j,f in enumerate(fig):
                    f.savefig(overview_dir/f"ALL_topomaps_page_{start:03d}_{stop:03d}_{j}.png",dpi=150,bbox_inches="tight")
                    plt.close(f)
            else:
                fig.savefig(overview_dir/f"ALL_topomaps_page_{start:03d}_{stop:03d}.png",dpi=150,bbox_inches="tight")
                plt.close(fig)

        # overview PSD di tutte le componenti
        for ic in range(n_comp):
            figs=ica.plot_properties(
                raw_ica,
                picks=[ic],
                psd_args=dict(fmin=0.5,fmax=50),
                reject_by_annotation=True,
                show=False
            )
            if not isinstance(figs,list):
                figs=[figs]
            for j,fig in enumerate(figs):
                fig.savefig(overview_dir/f"ALL_properties_IC{ic:03d}_{labels[ic]}_{j}.png",dpi=150,bbox_inches="tight")
                plt.close(fig)

        with open(Path(experiment_dir)/f"{sub}_pars.json","w") as f:
            json.dump(make_json_serializable(json_data),f,indent=4,sort_keys=True)

        print("Saved in:",postica_dir)
        print("Overview saved in:",overview_dir)

    return postICA_final,postICA_raw_continuum,ica,json_data


def select_ica_components_by_topomap_buttons(
    ica,
    raw_ica,
    labels=None,
    probabilities=None,
    preselected=None,
    ncols=8,
    title="Select ICs by TOPOGRAPHY"
):
    import math
    import numpy as np
    import mne
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    ica_ch_names=list(ica.ch_names)

    missing=[
        ch
        for ch in ica_ch_names
        if ch not in raw_ica.ch_names
    ]

    if missing:
        raise ValueError(
            f"Canali ICA assenti nel Raw: {missing}"
        )

    raw_ica_fit=raw_ica.copy().pick(
        ica_ch_names
    )

    components=ica.get_components()

    if components.shape[0]!=len(raw_ica_fit.ch_names):
        raise ValueError(
            f"Mismatch topomap: "
            f"component rows={components.shape[0]}, "
            f"Info channels={len(raw_ica_fit.ch_names)}"
        )

    n_comp=int(ica.n_components_)
    nrows=int(math.ceil(n_comp/ncols))

    if labels is None:
        labels=["unknown"]*n_comp
    else:
        labels=[str(x) for x in labels]

    if probabilities is None:
        probabilities=[np.nan]*n_comp
    else:
        probabilities=[
            float(np.max(np.atleast_1d(x)))
            for x in probabilities
        ]

    if len(labels)!=n_comp:
        raise ValueError(
            f"Labels ICLabel: {len(labels)} != componenti: {n_comp}"
        )

    if len(probabilities)!=n_comp:
        raise ValueError(
            f"Probabilità ICLabel: "
            f"{len(probabilities)} != componenti: {n_comp}"
        )

    selected=set()

    if preselected is not None:
        selected={
            int(ic)
            for ic in preselected
            if 0<=int(ic)<n_comp
        }

    label_aliases={
        "muscle artifact":"muscle",
        "eye blink":"eye",
        "heart beat":"heart",
        "line noise":"line",
        "channel noise":"channel",
        "brain":"brain",
        "other":"other"
    }

    def short_label(ic):
        label=str(labels[ic]).strip().lower()
        return label_aliases.get(label,label)

    def button_text(ic):
        action="REMOVE" if ic in selected else "KEEP"
        label=short_label(ic)
        probability=probabilities[ic]

        if np.isfinite(probability):
            prob_text=f"{probability:.2f}"
        else:
            prob_text="NA"

        return (
            f"{action} IC{ic}\n"
            f"{label} {prob_text}"
        )

    fig=plt.figure(
        figsize=(
            2.4*ncols,
            2.9*nrows
        )
    )

    fig.suptitle(
        title,
        fontsize=14
    )

    axes=[]
    buttons=[]

    for ic in range(n_comp):
        ax=fig.add_subplot(
            nrows,
            ncols,
            ic+1
        )

        axes.append(ax)

        mne.viz.plot_topomap(
            components[:,ic],
            raw_ica_fit.info,
            axes=ax,
            show=False,
            contours=6
        )

        ax.set_title("")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(
        rect=[0,0.05,1,0.94]
    )

    fig.canvas.draw()

    def update_button(button,ic):
        button.label.set_text(
            button_text(ic)
        )
    
        button.label.set_fontsize(8)
    
        if ic in selected:
            button.ax.set_facecolor(
                "salmon"
            )
    
            for spine in axes[ic].spines.values():
                spine.set_color("red")
                spine.set_linewidth(3)
    
        else:
            button.ax.set_facecolor(
                "lightgray"
            )
    
            for spine in axes[ic].spines.values():
                spine.set_color("black")
                spine.set_linewidth(1)

    for ic,ax in enumerate(axes):
        pos=ax.get_position()

        button_ax=fig.add_axes([
            pos.x0,
            pos.y1+0.002,
            pos.width,
            0.035
        ])

        button=Button(
            button_ax,
            button_text(ic)
        )

        buttons.append(button)

        def make_callback(ic,button):
            def callback(event):
                if ic in selected:
                    selected.remove(ic)
                else:
                    selected.add(ic)

                update_button(
                    button,
                    ic
                )

                fig.canvas.draw_idle()

            return callback

        button.on_clicked(
            make_callback(ic,button)
        )

        update_button(
            button,
            ic
        )

    done_ax=fig.add_axes([
        0.43,
        0.005,
        0.14,
        0.035
    ])

    done_button=Button(
        done_ax,
        "DONE"
    )

    done_button.on_clicked(
        lambda event:plt.close(fig)
    )

    plt.show(block=True)

    return sorted(selected)

def select_ica_components_by_psd_buttons(
    ica,
    raw_ica,
    labels=None,
    probabilities=None,
    preselected=None,
    ncols=4,
    nrows=3,
    fmin=0.5,
    fmax=50,
    title="Select ICs by PSD"
):
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    raw_ica_fit=raw_ica.copy().pick(
        list(ica.ch_names)
    )

    sources=ica.get_sources(
        raw_ica_fit
    ).get_data()

    sfreq=float(
        raw_ica_fit.info["sfreq"]
    )

    n_comp=int(
        ica.n_components_
    )

    if sources.shape[0]!=n_comp:
        raise ValueError(
            f"Sorgenti ICA: {sources.shape[0]} "
            f"!= componenti: {n_comp}"
        )

    if labels is None:
        labels=["unknown"]*n_comp
    else:
        labels=[str(x) for x in labels]

    if probabilities is None:
        probabilities=[np.nan]*n_comp
    else:
        probabilities=[
            float(
                np.max(
                    np.atleast_1d(x)
                )
            )
            for x in probabilities
        ]

    if len(labels)!=n_comp:
        raise ValueError(
            f"Labels ICLabel: {len(labels)} != {n_comp}"
        )

    if len(probabilities)!=n_comp:
        raise ValueError(
            f"Probabilità ICLabel: "
            f"{len(probabilities)} != {n_comp}"
        )

    selected=set()

    if preselected is not None:
        selected={
            int(ic)
            for ic in preselected
            if 0<=int(ic)<n_comp
        }

    label_aliases={
        "muscle artifact":"muscle",
        "eye blink":"eye",
        "heart beat":"heart",
        "line noise":"line",
        "channel noise":"channel",
        "brain":"brain",
        "other":"other"
    }

    def short_label(ic):
        label=str(
            labels[ic]
        ).strip().lower()

        return label_aliases.get(
            label,
            label
        )

    def button_text(ic):
        action=(
            "REMOVE"
            if ic in selected
            else "KEEP"
        )

        label=short_label(ic)
        probability=probabilities[ic]

        if np.isfinite(probability):
            probability_text=f"{probability:.2f}"
        else:
            probability_text="NA"

        return (
            f"{action} IC{ic}\n"
            f"{label} {probability_text}"
        )

    def style_button(
        button,
        ic
    ):
        button.label.set_text(
            button_text(ic)
        )

        button.label.set_fontsize(
            8
        )

        if ic in selected:
            button.color="salmon"
            button.hovercolor="lightsalmon"
        else:
            button.color="lightgray"
            button.hovercolor="silver"

        button.ax.set_facecolor(
            button.color
        )

    per_page=int(
        ncols*nrows
    )

    n_pages=max(
        1,
        int(
            math.ceil(
                n_comp/per_page
            )
        )
    )

    state={
        "page":0
    }

    psd_cache={}
    widget_refs=[]

    def get_psd(ic):
        if ic in psd_cache:
            return psd_cache[ic]

        signal=np.asarray(
            sources[ic],
            dtype=float
        )

        signal=signal-np.nanmean(
            signal
        )

        n_samples=len(signal)

        freqs=np.fft.rfftfreq(
            n_samples,
            d=1.0/sfreq
        )

        spectrum=np.fft.rfft(
            signal
        )

        psd=(
            np.abs(spectrum)**2
        )/n_samples

        mask=(
            (freqs>=float(fmin))
            &(freqs<=float(fmax))
        )

        psd_cache[ic]=(
            freqs[mask],
            psd[mask]
        )

        return psd_cache[ic]

    fig=plt.figure(
        figsize=(
            4*ncols,
            3.2*nrows
        )
    )

    def clear_dynamic_axes():
        for axis in list(
            fig.axes
        ):
            fig.delaxes(
                axis
            )

        widget_refs.clear()

    def draw_page():
        clear_dynamic_axes()

        page=int(
            state["page"]
        )

        start=page*per_page

        stop=min(
            start+per_page,
            n_comp
        )

        current_components=list(
            range(
                start,
                stop
            )
        )

        fig.suptitle(
            f"{title} | "
            f"page {page+1}/{n_pages} | "
            f"selected={sorted(selected)}",
            fontsize=14
        )

        plot_axes=[]

        for position,ic in enumerate(
            current_components
        ):
            axis=fig.add_subplot(
                nrows,
                ncols,
                position+1
            )

            freqs,psd_values=get_psd(
                ic
            )

            axis.plot(
                freqs,
                psd_values,
                linewidth=1
            )

            axis.set_xlim(
                float(fmin),
                float(fmax)
            )

            axis.set_yscale(
                "log"
            )

            axis.set_xlabel(
                "Hz",
                fontsize=8
            )

            axis.set_ylabel(
                "PSD",
                fontsize=8
            )

            axis.tick_params(
                labelsize=8
            )

            axis.grid(
                True,
                alpha=0.3
            )

            if ic in selected:
                for spine in axis.spines.values():
                    spine.set_color(
                        "red"
                    )

                    spine.set_linewidth(
                        3
                    )
            else:
                for spine in axis.spines.values():
                    spine.set_color(
                        "black"
                    )

                    spine.set_linewidth(
                        1
                    )

            plot_axes.append(
                (
                    ic,
                    axis
                )
            )

        plt.tight_layout(
            rect=[
                0,
                0.10,
                1,
                0.92
            ]
        )

        fig.canvas.draw()

        for ic,axis in plot_axes:
            axis_position=axis.get_position()

            button_axis=fig.add_axes([
                axis_position.x0,
                axis_position.y1+0.002,
                axis_position.width,
                0.040
            ])

            button=Button(
                button_axis,
                button_text(ic)
            )

            style_button(
                button,
                ic
            )

            def make_callback(
                component
            ):
                def callback(event):
                    if component in selected:
                        selected.remove(
                            component
                        )
                    else:
                        selected.add(
                            component
                        )

                    draw_page()

                    fig.canvas.draw_idle()

                return callback

            button.on_clicked(
                make_callback(ic)
            )

            widget_refs.append(
                button
            )

        previous_axis=fig.add_axes([
            0.20,
            0.015,
            0.10,
            0.045
        ])

        next_axis=fig.add_axes([
            0.32,
            0.015,
            0.10,
            0.045
        ])

        done_axis=fig.add_axes([
            0.70,
            0.015,
            0.12,
            0.045
        ])

        previous_button=Button(
            previous_axis,
            "< Prev"
        )

        next_button=Button(
            next_axis,
            "Next >"
        )

        done_button=Button(
            done_axis,
            "DONE"
        )

        previous_button.color="lightgray"
        previous_button.hovercolor="silver"

        next_button.color="lightgray"
        next_button.hovercolor="silver"

        done_button.color="lightgray"
        done_button.hovercolor="silver"

        previous_button.ax.set_facecolor(
            previous_button.color
        )

        next_button.ax.set_facecolor(
            next_button.color
        )

        done_button.ax.set_facecolor(
            done_button.color
        )

        def previous_page(event):
            if state["page"]>0:
                state["page"]-=1

                draw_page()

                fig.canvas.draw_idle()

        def next_page(event):
            if state["page"]<n_pages-1:
                state["page"]+=1

                draw_page()

                fig.canvas.draw_idle()

        def close_gui(event):
            plt.close(
                fig
            )

        previous_button.on_clicked(
            previous_page
        )

        next_button.on_clicked(
            next_page
        )

        done_button.on_clicked(
            close_gui
        )

        widget_refs.extend([
            previous_button,
            next_button,
            done_button
        ])

    draw_page()

    plt.show(
        block=True
    )

    return sorted(
        selected
    )

def select_ica_components_topo_psd_gui(
    ica,
    raw_ica,
    labels=None,
    probabilities=None,
    preselected=None,
    combine_mode="PSD_FINAL",
    ncols_topo=8,
    ncols_psd=4,
    nrows_psd=3,
    fmin=0.5,
    fmax=50
):
    topo_selected=select_ica_components_by_topomap_buttons(
        ica=ica,
        raw_ica=raw_ica,
        labels=labels,
        probabilities=probabilities,
        preselected=preselected,
        ncols=ncols_topo,
        title=(
            "TOPOGRAPHY SELECTION — "
            "all ICA components with ICLabel"
        )
    )

    psd_selected=select_ica_components_by_psd_buttons(
        ica=ica,
        raw_ica=raw_ica,
        labels=labels,
        probabilities=probabilities,
        preselected=topo_selected,
        ncols=ncols_psd,
        nrows=nrows_psd,
        fmin=fmin,
        fmax=fmax,
        title=(
            "PSD SELECTION — "
            "all ICA components with ICLabel"
        )
    )

    mode=str(
        combine_mode
    ).upper()

    topo_set=set(
        topo_selected
    )

    psd_set=set(
        psd_selected
    )

    if mode=="AND":
        final_selected=sorted(
            topo_set & psd_set
        )

    elif mode=="OR":
        final_selected=sorted(
            topo_set | psd_set
        )

    elif mode=="TOPO_FINAL":
        final_selected=sorted(
            topo_set
        )

    else:
        final_selected=sorted(
            psd_set
        )

    kept=[
        ic
        for ic in range(
            int(ica.n_components_)
        )
        if ic not in final_selected
    ]

    print(
        "ICLabel preselected:",
        sorted(preselected or [])
    )

    print(
        "Topography selected:",
        topo_selected
    )

    print(
        "PSD selected:",
        psd_selected
    )

    print(
        "Combination mode:",
        mode
    )

    print(
        "Final excluded:",
        final_selected
    )

    print(
        "Kept:",
        kept
    )

    return (
        final_selected,
        kept,
        topo_selected,
        psd_selected
    )

def ICAprocessing_continuum(raw_clean,json_data,experiment_dir,sub,
                            autoReject=True,
                            manualCheck=True,
                            computeFOOOF=False):

    from pathlib import Path
    from datetime import datetime
    import pickle
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from mne.preprocessing import ICA
    from mne_icalabel import label_components

    paths=rest_paths(experiment_dir)

    timestamp=datetime.now().strftime("%Y%m%d%H%M%S")
    ica_dir=paths["ica"]/timestamp
    ica_dir.mkdir(parents=True,exist_ok=True)

    print(f"⚙️ [{sub}] ICA continuo REST")

    raw_ica_input=raw_clean.copy().pick("eeg")
    # raw_ica_input.set_eeg_reference("average")

    n_components=json_data.get("n_components",None)

    ica=ICA(
        n_components=n_components,
        method=json_data.get("ica_method","fastica"),
        random_state=int(json_data.get("ica_random_state",42)),
        max_iter=json_data.get("ica_max_iter","auto")
    )

    print(f"⚙️ [{sub}] Fit ICA su Raw continuo")
    ica.fit(raw_ica_input)

    labels=None
    probs_all=None
    auto_excluded=[]

    if autoReject:
        print(f"🏷️ [{sub}] ICLabel automatic rejection")

        ic_labels=label_components(raw_ica_input,ica,method="iclabel")
        labels=ic_labels["labels"]
        probs_all=ic_labels["y_pred_proba"]

        artifact_tags=json_data.get(
            "ica_artifact_tags",
            [
                "eye blink",
                "muscle artifact",
                "heart beat",
                "line noise",
                "channel noise"
            ]
        )

        label_prob_threshold=float(json_data.get("do_label_prob_threshold",0.8))

        for i,label in enumerate(labels):
            probs=np.array(probs_all[i],ndmin=1)
            max_prob=float(probs.max())

            if label in artifact_tags and max_prob>=label_prob_threshold:
                print(f"❌ IC {i}: {label} prob={max_prob:.2f} excluded")
                auto_excluded.append(i)
            else:
                print(f"✅ IC {i}: {label} prob={max_prob:.2f} kept")

    ica.exclude=sorted(set([int(x) for x in auto_excluded]))

    json_data["ICA_autoExcludedComponents"]=[int(x) for x in ica.exclude]

    all_components=list(range(ica.n_components_))
    kept_components=[i for i in all_components if i not in ica.exclude]

    if len(ica.exclude)>0:
        fig=ica.plot_components(picks=ica.exclude,show=False)
        if isinstance(fig,list):
            for j,f in enumerate(fig):
                f.savefig(ica_dir/f"{sub}_AUTO_excluded_ICAs_{j}.png",dpi=150,bbox_inches="tight")
                plt.close(f)
        else:
            fig.savefig(ica_dir/f"{sub}_AUTO_excluded_ICAs.png",dpi=150,bbox_inches="tight")
            plt.close(fig)

    if len(kept_components)>0:
        fig=ica.plot_components(picks=kept_components,show=False)
        if isinstance(fig,list):
            for j,f in enumerate(fig):
                f.savefig(ica_dir/f"{sub}_AUTO_kept_ICAs_{j}.png",dpi=150,bbox_inches="tight")
                plt.close(f)
        else:
            fig.savefig(ica_dir/f"{sub}_AUTO_kept_ICAs.png",dpi=150,bbox_inches="tight")
            plt.close(fig)

    components_dir=ica_dir/"components"
    components_dir.mkdir(parents=True,exist_ok=True)

    for idx in all_components:
        tag="unknown"
        if labels is not None:
            tag=str(labels[idx]).replace("/","_").replace(" ","")

        fig=ica.plot_components(picks=idx,show=False)

        if isinstance(fig,list):
            for j,f in enumerate(fig):
                f.savefig(components_dir/f"component_{idx}_{tag}_view{j}.png",dpi=150,bbox_inches="tight")
                plt.close(f)
        else:
            fig.savefig(components_dir/f"component_{idx}_{tag}.png",dpi=150,bbox_inches="tight")
            plt.close(fig)

    if manualCheck:
        try:
            print(f"🖱️ [{sub}] Manual ICA check MNE-native su Raw continuo")
    
            raw_ica_input.load_data()
    
            ica.plot_components(
                picks=list(range(ica.n_components_)),
                show=True
            )
    
            ica.plot_sources(
                raw_ica_input,
                show=True,
                block=True
            )
    
            print(f"📌 Componenti escluse dopo GUI: {ica.exclude}")
    
            manual_input=input(
                "Componenti ICA da escludere, separate da virgola. "
                "Invio per mantenere la selezione GUI: "
            )
    
            if manual_input.strip()!="":
                manual_exclude=[
                    int(x.strip())
                    for x in manual_input.split(",")
                    if x.strip().isdigit()
                ]
                ica.exclude=sorted(set(list(ica.exclude)+manual_exclude))
    
            print(f"📌 Componenti escluse finali: {ica.exclude}")
    
        except Exception as e:
            print(f"⚠️ [{sub}] Manual ICA saltata: {e}")
    
    
    ica.exclude=sorted(set([int(x) for x in ica.exclude]))

    print(f"🧼 [{sub}] Applico ICA, componenti escluse: {ica.exclude}")
    postICA_raw=ica.apply(raw_ica_input.copy())

    json_data["ICA_object_type"]="Raw_REST_continuum"
    json_data["ICA_timestamp"]=timestamp
    json_data["ICA_components_tot"]=int(ica.n_components_)
    json_data["ICA_excludedComponents"]=[int(x) for x in ica.exclude]
    json_data["ICA_includedComponents_tot"]=int(ica.n_components_-len(ica.exclude))
    json_data["postICA_object_type"]="Raw_REST_continuum"

    if labels is not None:
        json_data["ICA_labels"]=[str(x) for x in labels]

    with open(paths["pkls"]/f"{timestamp}_{sub}_ica_model_REST_continuum.pkl","wb") as f:
        pickle.dump(ica,f)

    with open(paths["pkls"]/f"{timestamp}_{sub}_postICA_raw_REST_continuum.pkl","wb") as f:
        pickle.dump(postICA_raw,f)

    fig=postICA_raw.plot_psd(
        fmin=float(json_data.get("l_freq",0.5)),
        fmax=float(json_data.get("h_freq",45)),
        xscale="log",
        show=False
    )
    fig.savefig(ica_dir/f"{sub}_postICA_raw_REST_continuum_PSD.png",dpi=150,bbox_inches="tight")
    plt.close(fig)

    fig=postICA_raw.plot(
        n_channels=min(32,len(postICA_raw.ch_names)),
        duration=float(json_data.get("ica_preview_duration",20)),
        scalings={"eeg":float(json_data.get("rest_gui_scaling",50e-6))},
        show=False
    )
    fig.savefig(ica_dir/f"{sub}_postICA_raw_REST_continuum_preview.png",dpi=150,bbox_inches="tight")
    plt.close(fig)

    with open(Path(experiment_dir)/f"{sub}_pars.json","w") as json_file:
        json.dump(make_json_serializable(json_data),json_file,indent=4,sort_keys=True)

    print(f"✅ [{sub}] ICA continuo REST completata")
    print(f"   postICA_raw: {type(postICA_raw)}")
    print(f"   componenti totali: {json_data['ICA_components_tot']}")
    print(f"   componenti escluse: {json_data['ICA_excludedComponents']}")

    return postICA_raw,ica,json_data



def run_ica_continuum_pipeline(raw, events, json_data, experiment_dir, sub):
    from pathlib import Path
    import pickle

    # Set soglia dinamica
    do_run = json_data.get('do_ica_continuum', False)
    json_data['ica_continuum_tr'] = 30 if do_run else 10000
    ica_threshold_uv = json_data['ica_continuum_tr']

    if do_run:
        print(f"⚙️ Running ICA on continuous raw with threshold {ica_threshold_uv} µV...")
        raw_ica, ica_model = run_ica_artist_tms_events(
            raw.copy(),
            events,
            ext_threshold_uv=ica_threshold_uv,
            manualCheck=True,
            demean_between_events=True,
            json_data=json_data,
            experiment_dir=experiment_dir
        )

        # Save filtered raw and ICA model
        pkl_raw_path = Path(experiment_dir) / "7.pkls" / f"{sub}_raw_ICA_continuum.pkl"
        with open(pkl_raw_path, 'wb') as f:
            pickle.dump(raw_ica, f)

        pkl_ica_path = Path(experiment_dir) / "7.pkls" / f"{sub}_ica_model_continuum.pkl"
        with open(pkl_ica_path, 'wb') as f:
            pickle.dump(ica_model, f)

        # Optional epoching post-ICA for inspection
        epochs_cont = mne.Epochs(
            raw_ica,
            events,
            tmin=json_data['epochs_timewindow_min'],
            tmax=json_data['epochs_timewindow_max'],
            detrend=0,
            preload=True
        )
        epochs_cont = epochs_cont.pick('eeg').set_eeg_reference('average')

        # Save basic plots
        basicPlots(
            epochs_cont,
            json_data, experiment_dir, sub, 
            key=f'epochs_continumm_tr{ica_threshold_uv}',
            subPath='1.basic'
        )
        print("✅ ICA on continuum complete.")
        return raw_ica, ica_model

    else:
        print("⏭️ ICA on continuum skipped.")
        return raw, None
    
import matplotlib.pyplot as plt
from pathlib import Path

def remove_tms_artifact_and_plot_psd(raw, events, json_data, experiment_dir, sub, figsize=(10, 6), subPath='1.basic', do_plot=False, ica_continuum=False):
    from tmspath_utils_adj import tms_pulse_removal_init

    psd_dir = Path(experiment_dir) / subPath
    psd_dir.mkdir(parents=True, exist_ok=True)

    # === PSD PRE-removal ===
    print("📉 Plotting PSD before pulse removal...")
    fig = raw.plot_psd(
        fmin=json_data['l_freq'],
        fmax=json_data['broad_band_h_freq'],
        xscale='log',
        show=False
    )
    fig.set_size_inches(figsize)
    fig.savefig(psd_dir / '1.psdrawnoPulseRemovalnoBroadBandnoNotch.png')
    plt.close(fig)

    # === Optional: plot raw signal to inspect pulse
    if do_plot:
        raw.plot(n_channels=raw.info['nchan'])

    # === Pulse Artifact Removal ===
    print("⚡ Removing TMS pulse artifact...")
    raw = tms_pulse_removal_init(
        raw=raw,
        sfreq=raw.info['sfreq'],
        events_sample=events[:, 0],
        window=(json_data['pulse_artifact_rej_timewindow_min'],
                json_data['pulse_artifact_rej_timewindow_max']),
        smooth_window=(-json_data['pulse_artifact_rej_smoothingvalue'],
                       json_data['pulse_artifact_rej_smoothingvalue']),
        span=2
    )

    # === PSD POST-removal ===
    print("📉 Plotting PSD after pulse removal...")
    fig = raw.plot_psd(
        fmin=json_data['l_freq'],
        fmax=json_data['broad_band_h_freq'],
        xscale='log',
        show=False
    )
    fig.set_size_inches(figsize)
    fig.savefig(psd_dir / '2.psdrawwithPulseRemovalnoBroadBandnoNotch.png')
    plt.close(fig)

    # === Optional: plot raw after correction
    if do_plot:
        raw.plot(n_channels=raw.info['nchan'], scalings={'eeg': 50e-6})

    if ica_continuum:
        print('do ica continuum')
        raw, ica_model = run_ica_continuum_pipeline(
                                                raw=raw,
                                                events=events,
                                                json_data=json_data,
                                                experiment_dir=experiment_dir,
                                                sub=sub
                                                )

    return raw

import pickle
import matplotlib.pyplot as plt
from pathlib import Path

def filter_and_plot_raw(raw,json_data,experiment_dir,sub,figsize=(10,6),subPath="1.basic"):
    from pathlib import Path
    import pickle
    import matplotlib.pyplot as plt

    experiment_dir=str(Path(json_data.get("experiment_dir",experiment_dir)).expanduser().resolve())
    json_data["experiment_dir"]=experiment_dir
    paths=rest_paths(experiment_dir)

    out_dir=(Path(experiment_dir)/subPath).expanduser().resolve()
    out_dir.mkdir(parents=True,exist_ok=True)

    print(f"[DEBUG] filter_and_plot_raw out_dir: {out_dir}")
    print(f"[DEBUG] filter_and_plot_raw parent exists: {out_dir.exists()}")

    pkl_dir=paths["pkls"]
    pkl_dir.mkdir(parents=True,exist_ok=True)

    eeg_type=json_data.get("eeg_type",json_data.get("EEGTYPE","tms")).lower()

    print("DO-BROADBAND------------------------------------")

    raw=raw.filter(
        l_freq=json_data["l_freq"],
        h_freq=json_data["broad_band_h_freq"],
        method="iir",
        iir_params=dict(order=3,ftype="butter",phase="zero-double",btype="bandpass"),
        verbose=True
    )

    fig=raw.plot_psd(
        fmin=json_data["l_freq"],
        fmax=json_data["broad_band_h_freq"],
        xscale="log",
        show=False
    )
    fig.set_size_inches(figsize)
    fig.savefig(str(out_dir/"3.psdrawwithPulseRemovalwithBroadBand_noNotch.png"),dpi=300,bbox_inches="tight")
    plt.close(fig)

    print("DO-NOTCH------------------------------------")

    base_freq=float(json_data["powerline_freq"])
    h_freq=float(json_data.get("broad_band_h_freq",json_data.get("h_freq",raw.info["sfreq"]/2)))

    centers=[base_freq*i for i in range(1,6) if base_freq*i<h_freq]

    if len(centers)>0:
        print(f"NOTCH freqs: {centers}")
        raw=raw.notch_filter(freqs=centers)
    else:
        print("NOTCH skipped: no harmonics below high cutoff")

    fig=raw.plot_psd(
        fmin=json_data["l_freq"],
        fmax=json_data["broad_band_h_freq"],
        xscale="log",
        show=False
    )
    fig.set_size_inches(figsize)
    fig.savefig(str(out_dir/"4.psdrawwithPulseRemovalwithBroadBandwithNotch.png"),dpi=300,bbox_inches="tight") 
    plt.close(fig)
    
    raw_pkl_path=pkl_dir/f"{sub}_raw.pkl"

    with open(raw_pkl_path,"wb") as f:
        pickle.dump(raw,f)

    print(f"[INFO] Raw filtered and saved → {raw_pkl_path}")

    return raw

def clean_trials_channels(raw, events, json_data, experiment_dir, sub, seedChans=None):
    import mne, numpy as np, sys, io, json, re, ast
    from pathlib import Path

    def clean_bad_trials_list(x):
        if x is None:
            return []
        if isinstance(x, np.ndarray):
            x = x.tolist()
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return []
            try:
                parsed = ast.literal_eval(x)
                return clean_bad_trials_list(parsed)
            except Exception:
                nums = re.findall(r"\d+", x)
                return sorted(set([int(n) for n in nums]))
        if isinstance(x, (list, tuple, set)):
            out = []
            for item in x:
                if item is None:
                    continue
                if isinstance(item, (int, np.integer)):
                    out.append(int(item))
                elif isinstance(item, float) and item.is_integer():
                    out.append(int(item))
                elif isinstance(item, str):
                    item = item.strip()
                    if item == "" or item in ["[", "]", ","]:
                        continue
                    try:
                        parsed = ast.literal_eval(item)
                        if isinstance(parsed, (list, tuple, set, np.ndarray)):
                            out.extend(clean_bad_trials_list(parsed))
                        elif isinstance(parsed, (int, np.integer)):
                            out.append(int(parsed))
                        elif isinstance(parsed, float) and parsed.is_integer():
                            out.append(int(parsed))
                    except Exception:
                        nums = re.findall(r"\d+", item)
                        out.extend([int(n) for n in nums])
            return sorted(set(out))
        if isinstance(x, (int, np.integer)):
            return [int(x)]
        if isinstance(x, float) and x.is_integer():
            return [int(x)]
        return []

    def clean_bad_channels_list(x, valid_chans):
        if x is None:
            return []
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return []
            try:
                parsed = ast.literal_eval(x)
                return clean_bad_channels_list(parsed, valid_chans)
            except Exception:
                return [x] if x in valid_chans else []
        if isinstance(x, (list, tuple, set, np.ndarray)):
            return sorted(set([str(ch).strip() for ch in x if str(ch).strip() in valid_chans]))
        return []

    if seedChans is None:
        seedChans = json_data.get("seedChans", [])

    if "bad_trials" not in json_data or json_data["bad_trials"] is None:
        json_data["bad_trials"] = []

    if "bad_channels" not in json_data or json_data["bad_channels"] is None:
        json_data["bad_channels"] = []

    do_auto = json_data.get("do_chan_trials_selection_automatic", False)

    raw_copy = raw.copy()

    raw_copy.filter(
        l_freq=json_data["l_freq"],
        h_freq=json_data["h_freq"],
        method="iir",
        iir_params=dict(order=3, ftype="butter", phase="zero-double", btype="bandpass"),
        verbose=True
    )

    temp_epochs = mne.Epochs(
        raw_copy,
        events,
        tmin=-0.8,
        tmax=8,
        detrend=None,
        preload=True
    )

    temp_epochs = temp_epochs.pick("eeg")
    # temp_epochs = temp_epochs.set_eeg_reference("average")

    n_trials_before = len(temp_epochs)

    json_data["bad_channels"] = clean_bad_channels_list(
        json_data.get("bad_channels", []),
        temp_epochs.ch_names
    )

    json_data["bad_trials"] = clean_bad_trials_list(
        json_data.get("bad_trials", [])
    )

    data = temp_epochs.get_data()

    # here are pars for controlling automatic rejection of trials and chans
    # AUTOCHANTRIALSREJ
    chan_var = np.var(data, axis=(0, 2))
    thresh_low = np.percentile(chan_var, 5)
    thresh_high = np.percentile(chan_var, 95)
    auto_bad_channels = [
        ch for ch, var in zip(temp_epochs.ch_names, chan_var)
        if var < thresh_low or var > thresh_high
    ]
    auto_bad_channels = [ch for ch in auto_bad_channels if ch not in seedChans]
    trial_var = np.var(data, axis=(1, 2))
    t_low = np.percentile(trial_var, 5)
    t_high = np.percentile(trial_var, 95)

    auto_bad_trials = np.where((trial_var < t_low) | (trial_var > t_high))[0].tolist()

    if do_auto:
        print("🤖 Automatic artifact rejection")

        final_bad_channels = clean_bad_channels_list(auto_bad_channels, temp_epochs.ch_names)
        final_bad_trials = clean_bad_trials_list(auto_bad_trials)

        json_data["bad_channels"] = final_bad_channels
        json_data["bad_trials"] = final_bad_trials

        if len(final_bad_trials) > 0:
            keep_idx = np.where(~np.isin(temp_epochs.selection, final_bad_trials))[0]
            temp_epochs = temp_epochs[keep_idx]

        if len(final_bad_channels) > 0:
            print(f"🧹 Dropping automatic bad channels: {final_bad_channels}")
            temp_epochs.drop_channels(final_bad_channels)

    else:
        print("🖱️ Manual artifact rejection")

        user_bad_channels = clean_bad_channels_list(
            json_data.get("bad_channels", []),
            temp_epochs.ch_names
        )

        user_bad_trials = clean_bad_trials_list(
            json_data.get("bad_trials", [])
        )

        if len(user_bad_channels) > 0:
            print("📌 Using user-defined bad_channels before GUI")
            print(f"I am marking channels: {user_bad_channels}")
            temp_epochs.info["bads"] = user_bad_channels
        else:
            print("📌 No user-defined bad_channels: GUI starts with clean channels")
            temp_epochs.info["bads"] = []

        if len(user_bad_trials) > 0:
            print("📌 User-defined bad_trials present")
            print(f"These trials will be removed after GUI: {user_bad_trials}")
        else:
            print("📌 No user-defined bad_trials: all trials kept unless selected in GUI")

        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer

        temp_epochs.plot(
            butterfly=False,
            n_epochs=5,
            n_channels=len(temp_epochs.ch_names),
            block=True,
            use_opengl=True,
            scalings={"eog": 50e-6}
        )

        sys.stdout = old_stdout
        log_message = buffer.getvalue()

        save_bad_epochs_and_channels(log_message, experiment_dir, sub, json_data)
        print(log_message)

        gui_bad_channels = clean_bad_channels_list(
            temp_epochs.info["bads"],
            temp_epochs.ch_names
        )

        gui_bad_trials = clean_bad_trials_list(
            json_data.get("bad_trials", [])
        )

        final_bad_channels = sorted(set(user_bad_channels) | set(gui_bad_channels))
        final_bad_trials = sorted(set(user_bad_trials) | set(gui_bad_trials))

        json_data["bad_channels"] = final_bad_channels
        json_data["bad_trials"] = final_bad_trials

        if len(final_bad_trials) > 0:
            print(f"🧹 Dropping bad trials: {final_bad_trials}")
            keep_idx = np.where(~np.isin(temp_epochs.selection, final_bad_trials))[0]
            temp_epochs = temp_epochs[keep_idx]

        if len(final_bad_channels) > 0:
            print(f"🧹 Dropping bad channels: {final_bad_channels}")
            temp_epochs.drop_channels(final_bad_channels)

    json_data["bad_trials"] = clean_bad_trials_list(json_data.get("bad_trials", []))
    json_data["bad_channels"] = clean_bad_channels_list(
        json_data.get("bad_channels", []),
        raw.copy().pick("eeg").ch_names
    )

    json_data["trials_tot"] = int(n_trials_before)
    json_data["trials_selected"] = int(len(temp_epochs))
    json_data["channels_tot"] = int(len(raw.copy().pick("eeg").ch_names))
    json_data["channels_dropped"] = json_data["bad_channels"]
    json_data["channels_selected"] = int(len(temp_epochs.ch_names))
    json_data["ch_names_after_cleaning"] = list(temp_epochs.ch_names)

    temp_epochs = temp_epochs.resample(sfreq=json_data["r_sfreq"])
    # temp_epochs = temp_epochs.set_eeg_reference("average")

    with open(Path(experiment_dir) / f"{sub}_pars.json", "w") as json_file:
        json.dump(json_data, json_file, indent=4, sort_keys=True)

    print("✅ clean_trials_channels completed")
    print(f"   Trials kept: {json_data['trials_selected']} / {json_data['trials_tot']}")
    print(f"   Channels kept: {json_data['channels_selected']} / {json_data['channels_tot']}")
    print(f"   Dropped channels: {json_data['bad_channels']}")

    return temp_epochs, json_data


def annotate_rest_bad_segments_free(
    raw,
    json_data,
    experiment_dir,
    sub,
    h_freq_vis=60,
    scaling=50e-6,
    save=True,
    note="free_annotations"
):
    import json
    import pickle
    import mne
    import matplotlib.pyplot as plt
    from pathlib import Path

    paths=rest_paths(experiment_dir)

    out_dir=paths["trials"]/f"REST_{note}"
    out_dir.mkdir(parents=True,exist_ok=True)

    print(f"🖱️ [{sub}] Manual free annotation of REST bad segments")
    print("   Usa la GUI MNE per marcare porzioni BAD libere.")
    print("   Premi 'a' se la modalità annotation non è già attiva.")
    print("   Chiudi la finestra quando hai finito.")

    raw_in=raw.copy()

    preselect_edges=bool(
        json_data.get(
            "rest_preselect_edges_as_bad",
            True
        )
    )

    start_bad_sec=float(
        json_data.get(
            "rest_crop_start_sec",
            0.0
        )
    )

    end_bad_sec=float(
        json_data.get(
            "rest_crop_end_sec",
            0.0
        )
    )

    duration=float(raw_in.times[-1])

    if start_bad_sec<0 or end_bad_sec<0:
        raise ValueError(
            "rest_crop_start_sec e rest_crop_end_sec devono essere >= 0"
        )

    if (
        preselect_edges
        and start_bad_sec+end_bad_sec>=duration
    ):
        raise ValueError(
            f"Intervalli BAD non validi: "
            f"start={start_bad_sec}s, "
            f"end={end_bad_sec}s, "
            f"durata={duration:.3f}s"
        )

    if preselect_edges:
        annotations=raw_in.annotations.copy()

        existing_descriptions={
            str(description)
            for description in annotations.description
        }

        start_added=False
        end_added=False

        if (
            start_bad_sec>0
            and "BAD_REST_START" not in existing_descriptions
        ):
            annotations.append(
                onset=0.0,
                duration=start_bad_sec,
                description="BAD_REST_START"
            )
            start_added=True

        if (
            end_bad_sec>0
            and "BAD_REST_END" not in existing_descriptions
        ):
            annotations.append(
                onset=duration-end_bad_sec,
                duration=end_bad_sec,
                description="BAD_REST_END"
            )
            end_added=True

        raw_in.set_annotations(annotations)

        print(
            f"📌 BAD preselezionati: "
            f"inizio={start_bad_sec:.3f}s, "
            f"fine={end_bad_sec:.3f}s"
        )

        if not start_added and start_bad_sec>0:
            print("   BAD_REST_START era già presente.")

        if not end_added and end_bad_sec>0:
            print("   BAD_REST_END era già presente.")

    else:
        print("⏭️ Preselezione automatica dei bordi disattivata")

    raw_tmp=raw_in.copy().filter(
        l_freq=None,
        h_freq=h_freq_vis,
        method="iir",
        iir_params=dict(
            order=3,
            ftype="butter",
            phase="zero-double",
            btype="lowpass"
        ),
        verbose=True
    )

    raw_tmp.info["bads"]=list(
        raw_in.info.get("bads",[])
    )

    fig=raw_tmp.plot(
        n_channels=raw_tmp.info["nchan"],
        scalings={"eeg":scaling},
        block=False,
        show=True
    )

    try:
        fig.fake_keypress("a")
    except Exception:
        print("   Premi manualmente 'a' per attivare le annotazioni.")

    plt.show(block=True)
    plt.close("all")

    raw_out=raw_in.copy()

    raw_out.set_annotations(
        raw_tmp.annotations.copy()
    )

    raw_out.info["bads"]=list(
        raw_tmp.info.get("bads",[])
    )

    bad_annotations=[
        annotation
        for annotation in raw_out.annotations
        if str(annotation["description"]).startswith("BAD")
    ]

    bad_seconds=float(
        sum(
            float(annotation["duration"])
            for annotation in bad_annotations
        )
    )

    final_descriptions={
        str(description)
        for description in raw_out.annotations.description
    }

    json_data["rest_bad_segment_mode"]="free_annotations_like_PreprocessingRest"
    json_data["rest_free_annotations_tot"]=int(
        len(raw_out.annotations)
    )
    json_data["rest_free_bad_annotations_tot"]=int(
        len(bad_annotations)
    )
    json_data["rest_free_bad_seconds"]=bad_seconds
    json_data["rest_free_bad_channels"]=list(
        raw_out.info.get("bads",[])
    )
    json_data["rest_free_annotation_descriptions"]=[
        str(description)
        for description in raw_out.annotations.description
    ]

    json_data["rest_edges_preselected_as_bad"]=bool(
        preselect_edges
        and (
            "BAD_REST_START" in final_descriptions
            or "BAD_REST_END" in final_descriptions
        )
    )

    json_data["rest_bad_start_present"]=bool(
        "BAD_REST_START" in final_descriptions
    )

    json_data["rest_bad_end_present"]=bool(
        "BAD_REST_END" in final_descriptions
    )

    print(f"✅ [{sub}] Free annotations copied to raw")
    print(f"   Total annotations: {len(raw_out.annotations)}")
    print(f"   BAD annotations:   {len(bad_annotations)}")
    print(f"   BAD seconds:       {bad_seconds:.2f}")
    print(f"   Bad channels:      {raw_out.info['bads']}")
    print(
        "   BAD_REST_START:   "
        f"{'presente' if json_data['rest_bad_start_present'] else 'assente'}"
    )
    print(
        "   BAD_REST_END:     "
        f"{'presente' if json_data['rest_bad_end_present'] else 'assente'}"
    )

    if save:
        with open(
            out_dir/f"{sub}_raw_REST_freeAnnotated.pkl",
            "wb"
        ) as file:
            pickle.dump(raw_out,file)

        raw_out.save(
            out_dir/f"{sub}_raw_REST_freeAnnotated.fif",
            overwrite=True
        )

        annotations_df=raw_out.annotations.to_data_frame()

        annotations_df.to_csv(
            out_dir/f"{sub}_REST_free_annotations.csv",
            index=False
        )

        with open(
            Path(experiment_dir)/f"{sub}_pars.json",
            "w"
        ) as file:
            json.dump(
                make_json_serializable(json_data),
                file,
                indent=4,
                sort_keys=True
            )

    return raw_out,json_data

"""
changed on 30/06/2026
def clean_trials_channels(raw, events, json_data, experiment_dir, sub, seedChans=None):
    import mne, numpy as np, sys, io, json
    from pathlib import Path

    if seedChans is None:
        seedChans = json_data.get("seedChans", [])

    if "bad_trials" not in json_data or json_data["bad_trials"] is None:
        json_data["bad_trials"] = []

    if "bad_channels" not in json_data or json_data["bad_channels"] is None:
        json_data["bad_channels"] = []

    raw_copy = raw.copy()
    raw_copy.filter(
        l_freq=json_data['l_freq'],
        h_freq=json_data['h_freq'],
        method='iir',
        iir_params=dict(order=3, ftype='butter', phase='zero-double', btype='bandpass'),
        verbose=True
    )

    temp_epochs = mne.Epochs(
        raw_copy,
        events,
        # tmin=json_data['epochs_timewindow_min'], v2.2
        # tmax=json_data['epochs_timewindow_max'], v2.2
        tmin=-0.8, # different epoching for larger visualization v1
        tmax=8, # v1
        detrend=None,
        preload=True
    )
    temp_epochs = temp_epochs.pick('eeg')
    temp_epochs = temp_epochs.set_eeg_reference('average')

    n_trials_before = len(temp_epochs)

    data = temp_epochs.get_data()
    chan_var = np.var(data, axis=(0, 2))
    thresh_low = np.percentile(chan_var, 5)
    thresh_high = np.percentile(chan_var, 95)
    auto_bad_channels = [ch for ch, var in zip(temp_epochs.ch_names, chan_var) if var < thresh_low or var > thresh_high]
    auto_bad_channels = [ch for ch in auto_bad_channels if ch not in seedChans]

    trial_var = np.var(data, axis=(1, 2))
    t_low = np.percentile(trial_var, 5)
    t_high = np.percentile(trial_var, 95)
    auto_bad_trials = np.where((trial_var < t_low) | (trial_var > t_high))[0].tolist()

    if len(json_data["bad_channels"]) > 0:
        print("📌 Using bad_channels provided in json_data")
        print(f"I am taking off the channels {json_data['bad_channels']}")
        temp_epochs.info['bads'] = [ch for ch in json_data["bad_channels"] if ch in temp_epochs.ch_names]
    else:
        temp_epochs.info['bads'] = auto_bad_channels

    if len(json_data["bad_trials"]) > 0:
        print("📌 Using bad_trials provided in json_data")
        print(f"I am taking off the trials {json_data['bad_trials']}")
        mask = ~np.isin(temp_epochs.selection, json_data["bad_trials"])
        trialsToTake = temp_epochs.selection[mask]
        temp_epochs = temp_epochs[trialsToTake]

    elif not json_data['do_chan_trials_selection_automatic']:
    #if not json_data['do_chan_trials_selection_automatic']:
        print("🖱️ Manual artifact rejection")
        buffer = io.StringIO()
        sys.stdout = buffer
        temp_epochs.plot(
            butterfly=False,
            n_epochs=5, # 5 v1 # 20, v2.2
            n_channels=raw.info['nchan'],
            block=True,
            use_opengl=True,
            scalings={'eog': 50e-6}
        )
        sys.stdout = sys.__stdout__
        log_message = buffer.getvalue()
        save_bad_epochs_and_channels(log_message, experiment_dir, sub, json_data)
        print(log_message)

        if "bad_channels" in json_data and json_data["bad_channels"] is not None and len(json_data["bad_channels"]) > 0:
            temp_epochs.info['bads'] = [ch for ch in json_data["bad_channels"] if ch in temp_epochs.ch_names]

        if "bad_trials" in json_data and json_data["bad_trials"] is not None and len(json_data["bad_trials"]) > 0:
            #mask = ~np.isin(temp_epochs.selection, json_data["bad_trials"])
            #trialsToTake = temp_epochs.selection[mask]
            #temp_epochs = temp_epochs[trialsToTake]
            keep_idx = np.where(~np.isin(temp_epochs.selection, json_data["bad_trials"]))[0]
            temp_epochs = temp_epochs[keep_idx]
    else:
        print("🤖 Automatic artifact rejection")
        json_data['bad_channels'] = temp_epochs.info['bads']
        json_data['bad_trials'] = auto_bad_trials
        #mask = ~np.isin(temp_epochs.selection, json_data['bad_trials'])
        #trialsToTake = temp_epochs.selection[mask]
        #temp_epochs = temp_epochs[trialsToTake]
        keep_idx = np.where(~np.isin(temp_epochs.selection, json_data["bad_trials"]))[0]
        temp_epochs = temp_epochs[keep_idx]

    json_data['trials_tot'] = n_trials_before
    json_data['trials_selected'] = len(temp_epochs)
    json_data['bad_channels'] = temp_epochs.info['bads']

    temp_epochs = temp_epochs.resample(sfreq=json_data['r_sfreq'])
    temp_epochs = temp_epochs.set_eeg_reference('average')

    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
        json.dump(json_data, json_file, indent=4, sort_keys=True)

    return temp_epochs, json_data
"""    

def compute_rest_psd_bands(
    raw,
    json_data,
    experiment_dir,
    sub,
    note="preICA",
    fmin=0.5,
    fmax=40.0,
    psd_fmax_plot=60.0,
    n_per_seg=2000,
    n_overlap=1000,
    n_fft=2048,
    relative=True,
    remove_aperiodic=True,
    aperiodic_fit_fmin=1.0,
    aperiodic_fit_fmax=40.0,
    aperiodic_mode="fixed",
    save=True,
    show=False
):
    from pathlib import Path
    import json
    import pickle
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from fooof import FOOOF

    experiment_dir=Path(
        json_data.get(
            "experiment_dir",
            experiment_dir
        )
    ).expanduser().resolve()

    json_data["experiment_dir"]=str(
        experiment_dir
    )

    paths=rest_paths(
        experiment_dir
    )

    out_dir=(
        Path(paths["features"])
        /f"PSD{note}"
    ).expanduser().resolve()

    raw_dir=out_dir/"raw"
    corrected_dir=out_dir/"apec"
    fit_dir=out_dir/"aperiodic_fits"

    for directory in [
        out_dir,
        raw_dir,
        corrected_dir,
        fit_dir
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    print(
        "Saving PSD outputs to:",
        out_dir
    )

    bands={
        "delta":(0.5,4.0),
        "theta":(4.0,8.0),
        "alpha":(8.0,13.0),
        "beta":(13.0,30.0),
        "gamma":(30.0,40.0)
    }

    raw_in=raw.copy().pick("eeg")

    nyquist=float(
        raw_in.info["sfreq"]
    )/2.0

    fmax=min(
        float(fmax),
        nyquist
    )

    psd_fmax_plot=min(
        float(psd_fmax_plot),
        nyquist
    )

    aperiodic_fit_fmax=min(
        float(aperiodic_fit_fmax),
        fmax
    )

    if aperiodic_fit_fmin>=aperiodic_fit_fmax:
        raise ValueError(
            "Intervallo aperiodico non valido: "
            f"{aperiodic_fit_fmin}-"
            f"{aperiodic_fit_fmax} Hz"
        )

    psd=raw_in.compute_psd(
        method="welch",
        fmin=float(fmin),
        fmax=float(fmax),
        reject_by_annotation=True,
        n_per_seg=int(n_per_seg),
        n_overlap=int(n_overlap),
        n_fft=int(n_fft)
    )

    freqs=np.asarray(
        psd.freqs,
        dtype=float
    )

    psd_array=np.asarray(
        psd.get_data(
            exclude=[]
        ),
        dtype=float
    )

    psd_array[
        ~np.isfinite(psd_array)
    ]=np.nan

    positive_values=psd_array[
        np.isfinite(psd_array)
        &(psd_array>0)
    ]

    if positive_values.size==0:
        raise ValueError(
            "La PSD non contiene valori positivi validi."
        )

    epsilon=max(
        float(
            np.nanmin(
                positive_values
            )
        )*1e-12,
        np.finfo(float).tiny
    )

    psd_safe=np.maximum(
        psd_array,
        epsilon
    )

    def integrate(values,frequencies):
        values=np.asarray(
            values,
            dtype=float
        )

        frequencies=np.asarray(
            frequencies,
            dtype=float
        )

        finite=(
            np.isfinite(values)
            &np.isfinite(frequencies)
        )

        if np.sum(finite)<2:
            return np.nan

        if hasattr(np,"trapezoid"):
            return float(
                np.trapezoid(
                    values[finite],
                    frequencies[finite]
                )
            )

        return float(
            np.trapz(
                values[finite],
                frequencies[finite]
            )
        )

    def compute_band_dataframe(
        spectrum_array,
        spectrum_type,
        corrected=False
    ):
        rows=[]

        for ch_idx,ch_name in enumerate(
            psd.ch_names
        ):
            channel_spectrum=np.asarray(
                spectrum_array[ch_idx],
                dtype=float
            )

            total_power=integrate(
                channel_spectrum,
                freqs
            )

            row={
                "subject":sub,
                "note":note,
                "channel":ch_name,
                "channel_index":int(ch_idx),
                "spectrum_type":spectrum_type,
                "power_type":(
                    "relative"
                    if relative
                    else "absolute"
                ),
                "apec":bool(
                    corrected
                ),
                "fmin":float(fmin),
                "fmax":float(fmax),
                "total_power":float(
                    total_power
                ) if np.isfinite(total_power) else np.nan
            }

            for band_name,(bmin,bmax) in bands.items():
                mask=(
                    (freqs>=float(bmin))
                    &(freqs<float(bmax))
                )

                band_power=integrate(
                    channel_spectrum[mask],
                    freqs[mask]
                )

                row[
                    f"{band_name}_absolute"
                ]=(
                    float(band_power)
                    if np.isfinite(band_power)
                    else np.nan
                )

                if relative:
                    if (
                        np.isfinite(total_power)
                        and total_power>0
                        and np.isfinite(band_power)
                    ):
                        row[band_name]=float(
                            band_power/total_power
                        )
                    else:
                        row[band_name]=np.nan
                else:
                    row[band_name]=(
                        float(band_power)
                        if np.isfinite(band_power)
                        else np.nan
                    )

            rows.append(row)

        dataframe=pd.DataFrame(
            rows
        )

        return dataframe

    df_bands_raw=compute_band_dataframe(
        spectrum_array=psd_safe,
        spectrum_type="raw",
        corrected=False
    )

    periodic_linear=np.full_like(
        psd_safe,
        np.nan,
        dtype=float
    )

    aperiodic_linear=np.full_like(
        psd_safe,
        np.nan,
        dtype=float
    )

    periodic_log10=np.full_like(
        psd_safe,
        np.nan,
        dtype=float
    )

    aperiodic_log10=np.full_like(
        psd_safe,
        np.nan,
        dtype=float
    )

    aperiodic_rows=[]

    fit_mask=(
        (freqs>=float(aperiodic_fit_fmin))
        &(freqs<=float(aperiodic_fit_fmax))
    )

    fit_freqs=freqs[
        fit_mask
    ]

    if np.sum(fit_mask)<3:
        raise ValueError(
            "Numero insufficiente di frequenze "
            "per il fit aperiodico."
        )

    if remove_aperiodic:
        print(
            "🧮 Fit aperiodico per canale:",
            f"{aperiodic_fit_fmin}-"
            f"{aperiodic_fit_fmax} Hz,",
            f"mode={aperiodic_mode}"
        )

        for ch_idx,ch_name in enumerate(
            psd.ch_names
        ):
            channel_psd=psd_safe[
                ch_idx
            ]

            fit_power=channel_psd[
                fit_mask
            ]

            fm=FOOOF(
                aperiodic_mode=str(
                    aperiodic_mode
                ),
                verbose=False
            )

            fit_success=True
            fit_error_message=""

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore"
                    )

                    fm.fit(
                        fit_freqs,
                        fit_power,
                        [
                            float(
                                aperiodic_fit_fmin
                            ),
                            float(
                                aperiodic_fit_fmax
                            )
                        ]
                    )

                if not fm.has_model:
                    raise RuntimeError(
                        "FOOOF non ha prodotto "
                        "un modello valido."
                    )

                ap_params=np.asarray(
                    fm.aperiodic_params_,
                    dtype=float
                )

                if len(ap_params)==2:
                    offset=float(
                        ap_params[0]
                    )

                    exponent=float(
                        ap_params[1]
                    )

                    knee=np.nan

                    ap_fit_full=(
                        offset
                        -exponent
                        *np.log10(freqs)
                    )

                elif len(ap_params)==3:
                    offset=float(
                        ap_params[0]
                    )

                    knee=float(
                        ap_params[1]
                    )

                    exponent=float(
                        ap_params[2]
                    )

                    ap_fit_full=(
                        offset
                        -np.log10(
                            knee
                            +freqs**exponent
                        )
                    )

                else:
                    raise RuntimeError(
                        "Numero inatteso di parametri "
                        f"aperiodici: {len(ap_params)}"
                    )

                channel_log10=np.log10(
                    channel_psd
                )

                residual_log10=(
                    channel_log10
                    -ap_fit_full
                )

                periodic_ratio=(
                    10.0**residual_log10
                    -1.0
                )

                periodic_ratio=np.maximum(
                    periodic_ratio,
                    0.0
                )

                periodic_log10[
                    ch_idx
                ]=residual_log10

                periodic_linear[
                    ch_idx
                ]=periodic_ratio

                aperiodic_log10[
                    ch_idx
                ]=ap_fit_full

                aperiodic_linear[
                    ch_idx
                ]=10.0**ap_fit_full

                fit_error=float(
                    fm.error_
                )

                fit_r2=float(
                    fm.r_squared_
                )

            except Exception as error:
                fit_success=False
                fit_error_message=str(
                    error
                )

                offset=np.nan
                exponent=np.nan
                knee=np.nan
                fit_error=np.nan
                fit_r2=np.nan

                periodic_log10[
                    ch_idx
                ]=np.nan

                periodic_linear[
                    ch_idx
                ]=np.nan

                aperiodic_log10[
                    ch_idx
                ]=np.nan

                aperiodic_linear[
                    ch_idx
                ]=np.nan

                print(
                    f"⚠️ Fit aperiodico fallito "
                    f"per {ch_name}: {error}"
                )

            aperiodic_rows.append({
                "subject":sub,
                "note":note,
                "channel":ch_name,
                "channel_index":int(ch_idx),
                "aperiodic_mode":str(
                    aperiodic_mode
                ),
                "fit_fmin":float(
                    aperiodic_fit_fmin
                ),
                "fit_fmax":float(
                    aperiodic_fit_fmax
                ),
                "offset":offset,
                "knee":knee,
                "exponent":exponent,
                "fit_error":fit_error,
                "fit_r2":fit_r2,
                "fit_success":bool(
                    fit_success
                ),
                "fit_error_message":fit_error_message
            })

        df_aperiodic=pd.DataFrame(
            aperiodic_rows
        )

        df_bands_corrected=compute_band_dataframe(
            spectrum_array=periodic_linear,
            spectrum_type="apec",
            corrected=True
        )

    else:
        df_aperiodic=pd.DataFrame(
            columns=[
                "subject",
                "note",
                "channel",
                "channel_index",
                "aperiodic_mode",
                "fit_fmin",
                "fit_fmax",
                "offset",
                "knee",
                "exponent",
                "fit_error",
                "fit_r2",
                "fit_success",
                "fit_error_message"
            ]
        )

        df_bands_corrected=pd.DataFrame(
            columns=df_bands_raw.columns
        )

    band_columns=list(
        bands.keys()
    )

    id_columns=[
        "subject",
        "note",
        "channel",
        "channel_index",
        "spectrum_type",
        "power_type",
        "apec",
        "fmin",
        "fmax",
        "total_power"
    ]

    df_long_raw=df_bands_raw.melt(
        id_vars=id_columns,
        value_vars=band_columns,
        var_name="band",
        value_name="power"
    )

    if not df_bands_corrected.empty:
        df_long_corrected=(
            df_bands_corrected.melt(
                id_vars=id_columns,
                value_vars=band_columns,
                var_name="band",
                value_name="power"
            )
        )
    else:
        df_long_corrected=pd.DataFrame(
            columns=(
                id_columns
                +["band","power"]
            )
        )

    df_bands_all=pd.concat(
        [
            df_bands_raw,
            df_bands_corrected
        ],
        ignore_index=True
    )

    df_long_all=pd.concat(
        [
            df_long_raw,
            df_long_corrected
        ],
        ignore_index=True
    )

    fig=psd.plot(
        dB=False,
        xscale="linear",
        average=True,
        show=show
    )

    fig.savefig(
        raw_dir/f"{sub}_{note}_PSD_linear.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    fig=psd.plot(
        dB=True,
        xscale="log",
        average=True,
        show=show
    )

    fig.savefig(
        raw_dir/f"{sub}_{note}_PSD_log_dB.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    raw_band_means=(
        df_long_raw
        .groupby("band")["power"]
        .mean()
        .reindex(band_columns)
    )

    raw_band_stds=(
        df_long_raw
        .groupby("band")["power"]
        .std()
        .reindex(band_columns)
    )

    fig,ax=plt.subplots(
        figsize=(10,6)
    )

    ax.bar(
        raw_band_means.index,
        raw_band_means.values,
        yerr=raw_band_stds.values,
        capsize=4
    )

    ax.set_title(
        f"{sub} - {note} raw "
        f"{'relative' if relative else 'absolute'} "
        "band power"
    )

    ax.set_xlabel(
        "Band"
    )

    ax.set_ylabel(
        "Relative power"
        if relative
        else "Absolute power"
    )

    fig.tight_layout()

    fig.savefig(
        raw_dir/f"{sub}_{note}_bandpower_raw.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    if remove_aperiodic:
        valid_corrected=np.isfinite(
            periodic_linear
        ).any(axis=1)

        if np.any(valid_corrected):
            mean_raw=np.nanmean(
                psd_safe,
                axis=0
            )

            mean_aperiodic=np.nanmean(
                aperiodic_linear[
                    valid_corrected
                ],
                axis=0
            )

            mean_periodic=np.nanmean(
                periodic_linear[
                    valid_corrected
                ],
                axis=0
            )

            fig,ax=plt.subplots(
                figsize=(11,6)
            )

            ax.plot(
                freqs,
                10.0*np.log10(
                    np.maximum(
                        mean_raw,
                        epsilon
                    )
                ),
                label="Raw PSD"
            )

            ax.plot(
                freqs,
                10.0*np.log10(
                    np.maximum(
                        mean_aperiodic,
                        epsilon
                    )
                ),
                label="Aperiodic fit"
            )

            ax.set_xlim(
                float(fmin),
                float(
                    min(
                        psd_fmax_plot,
                        fmax
                    )
                )
            )

            ax.set_xscale(
                "log"
            )

            ax.set_xlabel(
                "Frequency [Hz]"
            )

            ax.set_ylabel(
                "PSD [dB]"
            )

            ax.set_title(
                f"{sub} - {note}: "
                "raw PSD and aperiodic fit"
            )

            ax.legend()
            ax.grid(
                True,
                alpha=0.2
            )

            fig.tight_layout()

            fig.savefig(
                corrected_dir
                /f"{sub}_{note}_raw_vs_aperiodic.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

            fig,ax=plt.subplots(
                figsize=(11,6)
            )

            ax.plot(
                freqs,
                mean_periodic
            )

            ax.axhline(
                0,
                linewidth=1
            )

            ax.set_xlim(
                float(fmin),
                float(
                    min(
                        psd_fmax_plot,
                        fmax
                    )
                )
            )

            ax.set_xlabel(
                "Frequency [Hz]"
            )

            ax.set_ylabel(
                "Power above aperiodic background"
            )

            ax.set_title(
                f"{sub} - {note}: "
                "aperiodic-corrected spectrum"
            )

            ax.grid(
                True,
                alpha=0.2
            )

            fig.tight_layout()

            fig.savefig(
                corrected_dir
                /f"{sub}_{note}_PSD_apec.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

        corrected_band_means=(
            df_long_corrected
            .groupby("band")["power"]
            .mean()
            .reindex(band_columns)
        )

        corrected_band_stds=(
            df_long_corrected
            .groupby("band")["power"]
            .std()
            .reindex(band_columns)
        )

        fig,ax=plt.subplots(
            figsize=(10,6)
        )

        ax.bar(
            corrected_band_means.index,
            corrected_band_means.values,
            yerr=corrected_band_stds.values,
            capsize=4
        )

        ax.set_title(
            f"{sub} - {note} aperiodic-corrected "
            f"{'relative' if relative else 'absolute'} "
            "band power"
        )

        ax.set_xlabel(
            "Band"
        )

        ax.set_ylabel(
            "Relative corrected power"
            if relative
            else "Corrected power"
        )

        fig.tight_layout()

        fig.savefig(
            corrected_dir
            /f"{sub}_{note}_bandpower_corrected.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        comparison=(
            df_long_all
            .groupby(
                [
                    "spectrum_type",
                    "band"
                ]
            )["power"]
            .agg(
                ["mean","std"]
            )
            .reset_index()
        )

        x=np.arange(
            len(band_columns)
        )

        width=0.35

        raw_summary=(
            comparison[
                comparison["spectrum_type"]=="raw"
            ]
            .set_index("band")
            .reindex(band_columns)
        )

        corrected_summary=(
            comparison[
                comparison[
                    "spectrum_type"
                ]=="apec"
            ]
            .set_index("band")
            .reindex(band_columns)
        )

        fig,ax=plt.subplots(
            figsize=(11,6)
        )

        ax.bar(
            x-width/2,
            raw_summary["mean"].values,
            width,
            yerr=raw_summary["std"].values,
            label="Raw",
            capsize=3
        )

        ax.bar(
            x+width/2,
            corrected_summary["mean"].values,
            width,
            yerr=corrected_summary["std"].values,
            label="Aperiodic corrected",
            capsize=3
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            band_columns
        )

        ax.set_ylabel(
            "Relative power"
            if relative
            else "Band power"
        )

        ax.set_title(
            f"{sub} - {note}: "
            "raw vs aperiodic-corrected band power"
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            out_dir
            /f"{sub}_{note}_bandpower_raw_vs_corrected.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        for ch_idx,ch_name in enumerate(
            psd.ch_names
        ):
            if not np.isfinite(
                periodic_linear[ch_idx]
            ).any():
                continue

            fig,ax=plt.subplots(
                figsize=(10,5)
            )

            ax.plot(
                freqs,
                10.0*np.log10(
                    np.maximum(
                        psd_safe[ch_idx],
                        epsilon
                    )
                ),
                label="Raw PSD"
            )

            ax.plot(
                freqs,
                10.0*np.log10(
                    np.maximum(
                        aperiodic_linear[ch_idx],
                        epsilon
                    )
                ),
                label="Aperiodic fit"
            )

            ax.set_xscale(
                "log"
            )

            ax.set_xlim(
                float(fmin),
                float(fmax)
            )

            ax.set_xlabel(
                "Frequency [Hz]"
            )

            ax.set_ylabel(
                "PSD [dB]"
            )

            ax.set_title(
                f"{sub} - {note} - {ch_name}"
            )

            ax.legend()
            ax.grid(
                True,
                alpha=0.2
            )

            fig.tight_layout()

            safe_channel=str(
                ch_name
            ).replace(
                "/",
                "_"
            ).replace(
                "\\",
                "_"
            )

            fig.savefig(
                fit_dir
                /f"{safe_channel}_aperiodic_fit.png",
                dpi=150,
                bbox_inches="tight"
            )

            plt.close(fig)

    if save:
        df_bands_raw.to_csv(
            raw_dir
            /f"{sub}_{note}_bandpower_raw_wide.csv",
            index=False
        )

        df_long_raw.to_csv(
            raw_dir
            /f"{sub}_{note}_bandpower_raw_long.csv",
            index=False
        )

        df_bands_corrected.to_csv(
            corrected_dir
            /f"{sub}_{note}_bandpower_corrected_wide.csv",
            index=False
        )

        df_long_corrected.to_csv(
            corrected_dir
            /f"{sub}_{note}_bandpower_corrected_long.csv",
            index=False
        )

        df_bands_all.to_csv(
            out_dir
            /f"{sub}_{note}_bandpower_all_wide.csv",
            index=False
        )

        df_long_all.to_csv(
            out_dir
            /f"{sub}_{note}_bandpower_all_long.csv",
            index=False
        )

        df_aperiodic.to_csv(
            out_dir
            /f"{sub}_{note}_aperiodic_parameters.csv",
            index=False
        )

        np.save(
            out_dir
            /f"{sub}_{note}_freqs.npy",
            freqs
        )

        np.save(
            raw_dir
            /f"{sub}_{note}_PSD_raw.npy",
            psd_safe
        )

        if remove_aperiodic:
            np.save(
                corrected_dir
                /f"{sub}_{note}_PSD_periodic_linear.npy",
                periodic_linear
            )

            np.save(
                corrected_dir
                /f"{sub}_{note}_PSD_periodic_log10.npy",
                periodic_log10
            )

            np.save(
                corrected_dir
                /f"{sub}_{note}_PSD_aperiodic_linear.npy",
                aperiodic_linear
            )

            np.save(
                corrected_dir
                /f"{sub}_{note}_PSD_aperiodic_log10.npy",
                aperiodic_log10
            )

        with open(
            out_dir
            /f"{sub}_{note}_psd.pkl",
            "wb"
        ) as file:
            pickle.dump(
                psd,
                file
            )

        try:
            psd.save(
                out_dir
                /f"{sub}_{note}_psd.hdf5",
                overwrite=True
            )
        except Exception as error:
            print(
                f"⚠️ PSD hdf5 non salvata: {error}"
            )

    json_data[
        f"PSD_{note}_dir"
    ]=str(out_dir)

    json_data[
        f"PSD_{note}_raw_bands_csv"
    ]=str(
        raw_dir
        /f"{sub}_{note}_bandpower_raw_wide.csv"
    )

    json_data[
        f"PSD_{note}_corrected_bands_csv"
    ]=str(
        corrected_dir
        /f"{sub}_{note}_bandpower_corrected_wide.csv"
    )

    json_data[
        f"PSD_{note}_aperiodic_parameters_csv"
    ]=str(
        out_dir
        /f"{sub}_{note}_aperiodic_parameters.csv"
    )

    json_data[
        f"PSD_{note}_relative"
    ]=bool(relative)

    json_data[
        f"PSD_{note}_aperiodic_removed"
    ]=bool(remove_aperiodic)

    json_data[
        f"PSD_{note}_aperiodic_mode"
    ]=str(aperiodic_mode)

    json_data[
        f"PSD_{note}_aperiodic_fit_fmin"
    ]=float(aperiodic_fit_fmin)

    json_data[
        f"PSD_{note}_aperiodic_fit_fmax"
    ]=float(aperiodic_fit_fmax)

    json_data[
        f"PSD_{note}_n_per_seg"
    ]=int(n_per_seg)

    json_data[
        f"PSD_{note}_n_overlap"
    ]=int(n_overlap)

    json_data[
        f"PSD_{note}_n_fft"
    ]=int(n_fft)

    with open(
        experiment_dir/f"{sub}_pars.json",
        "w"
    ) as file:
        json.dump(
            make_json_serializable(
                json_data
            ),
            file,
            indent=4,
            sort_keys=True
        )

    plt.close("all")

    print(
        f"✅ PSD raw + aperiodic-corrected "
        f"{note} salvate in: {out_dir}"
    )

    return (
        psd,
        df_bands_raw,
        df_long_raw,
        df_bands_corrected,
        df_long_corrected,
        df_aperiodic,
        json_data
    )

def compare_rest_bandpower_pre_post(
    df_pre,
    df_post,
    json_data,
    experiment_dir,
    sub,
    pre_label="preICA",
    post_label="postICA",
    save=True
):
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    experiment_dir=Path(
        json_data.get(
            "experiment_dir",
            experiment_dir
        )
    ).expanduser().resolve()

    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"experiment_dir non esiste: {experiment_dir}"
        )

    json_data["experiment_dir"]=str(experiment_dir)

    paths=rest_paths(experiment_dir)

    out_dir=(
        Path(paths["features"])
        /f"BP_{pre_label}_{post_label}"
    ).resolve()

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Saving bandpower comparison to:",out_dir)
    print("out_dir exists:",out_dir.exists())
    print("Path length:",len(str(out_dir)))

    bands=[
        "delta",
        "theta",
        "alpha",
        "beta",
        "gamma"
    ]

    required_columns={"channel",*bands}

    missing_pre=required_columns-set(df_pre.columns)
    missing_post=required_columns-set(df_post.columns)

    if missing_pre:
        raise ValueError(
            f"Colonne mancanti in df_pre: {sorted(missing_pre)}"
        )

    if missing_post:
        raise ValueError(
            f"Colonne mancanti in df_post: {sorted(missing_post)}"
        )

    pre=df_pre[
        ["channel"]+bands
    ].copy()

    post=df_post[
        ["channel"]+bands
    ].copy()

    pre=pre.drop_duplicates(
        subset="channel",
        keep="first"
    )

    post=post.drop_duplicates(
        subset="channel",
        keep="first"
    )

    merged=pre.merge(
        post,
        on="channel",
        how="inner",
        suffixes=(
            f"_{pre_label}",
            f"_{post_label}"
        ),
        validate="one_to_one"
    )

    if merged.empty:
        raise ValueError(
            "Nessun canale comune tra df_pre e df_post."
        )

    for band in bands:
        pre_col=f"{band}_{pre_label}"
        post_col=f"{band}_{post_label}"

        merged[f"{band}_delta"]=(
            merged[post_col]
            -merged[pre_col]
        )

        denominator=merged[pre_col].replace(
            0,
            np.nan
        )

        merged[f"{band}_ratio"]=(
            merged[post_col]
            /denominator
        )

        merged[f"{band}_pct"]=(
            100.0
            *merged[f"{band}_delta"]
            /denominator
        )

    rows=[]

    for band in bands:
        ratio_values=merged[
            f"{band}_ratio"
        ].replace(
            [np.inf,-np.inf],
            np.nan
        )

        percent_values=merged[
            f"{band}_pct"
        ].replace(
            [np.inf,-np.inf],
            np.nan
        )

        rows.append({
            "band":band,
            f"mean_{pre_label}":float(
                merged[
                    f"{band}_{pre_label}"
                ].mean()
            ),
            f"std_{pre_label}":float(
                merged[
                    f"{band}_{pre_label}"
                ].std()
            ),
            f"mean_{post_label}":float(
                merged[
                    f"{band}_{post_label}"
                ].mean()
            ),
            f"std_{post_label}":float(
                merged[
                    f"{band}_{post_label}"
                ].std()
            ),
            "mean_delta":float(
                merged[
                    f"{band}_delta"
                ].mean()
            ),
            "std_delta":float(
                merged[
                    f"{band}_delta"
                ].std()
            ),
            "mean_ratio":float(
                ratio_values.mean()
            ),
            "mean_percent_change":float(
                percent_values.mean()
            ),
            "n_channels":int(
                merged[
                    f"{band}_delta"
                ].notna().sum()
            )
        })

    df_summary=pd.DataFrame(rows)

    x=np.arange(len(bands))
    width=0.35

    fig,ax=plt.subplots(
        figsize=(10,6)
    )

    ax.bar(
        x-width/2,
        df_summary[f"mean_{pre_label}"],
        width,
        yerr=df_summary[f"std_{pre_label}"],
        label=pre_label,
        capsize=4
    )

    ax.bar(
        x+width/2,
        df_summary[f"mean_{post_label}"],
        width,
        yerr=df_summary[f"std_{post_label}"],
        label=post_label,
        capsize=4
    )

    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel("Relative band power")
    ax.set_title(
        f"{sub} REST bandpower "
        f"{pre_label} vs {post_label}"
    )
    ax.legend()
    fig.tight_layout()

    comparison_png=out_dir/"compare.png"

    fig.savefig(
        str(comparison_png),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    fig,ax=plt.subplots(
        figsize=(10,6)
    )

    ax.bar(
        df_summary["band"],
        df_summary["mean_delta"],
        yerr=df_summary["std_delta"],
        capsize=4
    )

    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_ylabel(
        f"{post_label} - {pre_label}"
    )

    ax.set_title(
        f"{sub} REST bandpower delta"
    )

    fig.tight_layout()

    delta_png=out_dir/"delta.png"

    fig.savefig(
        str(delta_png),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    fig,ax=plt.subplots(
        figsize=(10,6)
    )

    ax.bar(
        df_summary["band"],
        df_summary["mean_percent_change"]
    )

    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_ylabel(
        "Mean change [%]"
    )

    ax.set_title(
        f"{sub} REST bandpower percent change"
    )

    fig.tight_layout()

    percent_png=out_dir/"percent.png"

    fig.savefig(
        str(percent_png),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    channelwise_csv=out_dir/"channelwise.csv"
    summary_csv=out_dir/"summary.csv"

    if save:
        merged.to_csv(
            channelwise_csv,
            index=False
        )

        df_summary.to_csv(
            summary_csv,
            index=False
        )

    json_data[
        f"PSD_compare_{pre_label}_vs_{post_label}_dir"
    ]=str(out_dir)

    json_data[
        f"PSD_compare_{pre_label}_vs_{post_label}_channelwise"
    ]=str(channelwise_csv)

    json_data[
        f"PSD_compare_{pre_label}_vs_{post_label}_summary"
    ]=str(summary_csv)

    json_data[
        f"PSD_compare_{pre_label}_vs_{post_label}_png"
    ]=str(comparison_png)

    json_data[
        f"PSD_compare_{pre_label}_vs_{post_label}_n_channels"
    ]=int(len(merged))

    with open(
        experiment_dir/f"{sub}_pars.json",
        "w"
    ) as file:
        json.dump(
            make_json_serializable(json_data),
            file,
            indent=4,
            sort_keys=True
        )

    print(
        f"✅ Confronto PSD bands salvato in: {out_dir}"
    )

    return merged,df_summary,json_data

def clean_trials_channels_old_20260416(raw, events, json_data, experiment_dir, sub, seedChans=None):
    import mne, numpy as np, sys, io
    from pathlib import Path

    if seedChans is None:
        seedChans = json_data.get("seedChans", [])

    # === Filtering
    raw_copy = raw.copy()
    raw_copy.filter(
        l_freq=json_data['l_freq'],
        h_freq=json_data['h_freq'],
        method='iir',
        iir_params=dict(order=3, ftype='butter', phase='zero-double', btype='bandpass'),
        verbose=True
    )

    # === Epoching
    temp_epochs = mne.Epochs(
        raw_copy,
        events,
        tmin=json_data['epochs_timewindow_min'],
        tmax=json_data['epochs_timewindow_max'],
        detrend=None,
        preload=True
    )
    temp_epochs = temp_epochs.pick('eeg')
    temp_epochs = temp_epochs.set_eeg_reference('average')

    # === Automatic detection: bad channels
    data = temp_epochs.get_data()
    chan_var = np.var(data, axis=(0, 2))
    thresh_low = np.percentile(chan_var, 5)
    thresh_high = np.percentile(chan_var, 95)
    bad_channels = [ch for ch, var in zip(temp_epochs.ch_names, chan_var) if var < thresh_low or var > thresh_high]
    bad_channels_non_seed = [ch for ch in bad_channels if ch not in seedChans]

    # === Automatic detection: bad trials
    trial_var = np.var(data, axis=(1, 2))
    t_low = np.percentile(trial_var, 5)
    t_high = np.percentile(trial_var, 95)
    auto_bad_trials = np.where((trial_var < t_low) | (trial_var > t_high))[0]
    
    if "bad_trials" not in json_data or json_data["bad_trials"] is None:
        json_data["bad_trials"] = []
    
    if len(json_data["bad_trials"]) > 0:
        print("📌 Using bad_trials provided in json_data")
        print(f"I am taking off the trials {json_data['bad_trials']}")
        temp_epochs.info['bads'] = bad_channels_non_seed
        mask = ~np.isin(temp_epochs.selection, json_data["bad_trials"])
        trialsToTake = temp_epochs.selection[mask]
        temp_epochs = temp_epochs[trialsToTake]
    
    elif not json_data['do_chan_trials_selection_automatic']:
        print("🖱️ Manual artifact rejection")
        buffer = io.StringIO()
        sys.stdout = buffer
        fig = temp_epochs.plot(
            butterfly=False,
            n_epochs=20,
            n_channels=raw.info['nchan'],
            block=True,
            use_opengl=True,
            scalings={'eog': 50e-6}
        )
        sys.stdout = sys.__stdout__
        log_message = buffer.getvalue()
        save_bad_epochs_and_channels(log_message, experiment_dir, json_data)
        print(log_message)
    
    else:
        print("🤖 Automatic artifact rejection")
        temp_epochs.info['bads'] = bad_channels_non_seed
        json_data['bad_trials'] = auto_bad_trials.tolist()
        mask = ~np.isin(temp_epochs.selection, json_data['bad_trials'])
        trialsToTake = temp_epochs.selection[mask]
        temp_epochs = temp_epochs[trialsToTake]

    # === Aggiornamenti json
    json_data['trials_tot'] = len(temp_epochs.annotations)
    json_data['trials_selected'] = temp_epochs.events.shape[0]
    json_data['bad_channels'] = temp_epochs.info['bads']

    # === Final steps
    temp_epochs = temp_epochs.resample(sfreq=json_data['r_sfreq'])
    temp_epochs = temp_epochs.set_eeg_reference('average')

    # Salva parametri
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
            json.dump(json_data, json_file, indent=4, sort_keys=True)


    return temp_epochs, json_data



def clean_trials_channels_20260415(raw, events, json_data, experiment_dir, sub, seedChans=None):
    import mne, numpy as np, sys, io
    from pathlib import Path

    if seedChans is None:
        seedChans = json_data.get("seedChans", [])

    # === Filtering
    raw_copy = raw.copy()
    raw_copy.filter(
        l_freq=json_data['l_freq'],
        h_freq=json_data['h_freq'],
        method='iir',
        iir_params=dict(order=3, ftype='butter', phase='zero-double', btype='bandpass'),
        verbose=True
    )

    # === Epoching
    temp_epochs = mne.Epochs(
        raw_copy,
        events,
        tmin=json_data['epochs_timewindow_min'],
        tmax=json_data['epochs_timewindow_max'],
        detrend=None,
        preload=True
    )
    temp_epochs = temp_epochs.pick('eeg')
    temp_epochs = temp_epochs.set_eeg_reference('average')

    # === Automatic detection: bad channels
    data = temp_epochs.get_data()
    chan_var = np.var(data, axis=(0, 2))
    thresh_low = np.percentile(chan_var, 5)
    thresh_high = np.percentile(chan_var, 95)
    bad_channels = [ch for ch, var in zip(temp_epochs.ch_names, chan_var) if var < thresh_low or var > thresh_high]
    bad_channels_non_seed = [ch for ch in bad_channels if ch not in seedChans]

    # === Automatic detection: bad trials
    trial_var = np.var(data, axis=(1, 2))
    t_low = np.percentile(trial_var, 5)
    t_high = np.percentile(trial_var, 95)
    bad_trials = np.where((trial_var < t_low) | (trial_var > t_high))[0]

    if not json_data['do_chan_trials_selection_automatic']:
        print("🖱️ Manual artifact rejection")
        buffer = io.StringIO()
        sys.stdout = buffer
        fig = temp_epochs.plot(butterfly=False, n_epochs=20, n_channels=raw.info['nchan'], block=True, use_opengl=True, scalings={'eog': 50e-6})
        sys.stdout = sys.__stdout__
        log_message = buffer.getvalue()
        save_bad_epochs_and_channels(log_message, experiment_dir, json_data)
        print(log_message)
    else:
        print("🤖 Automatic artifact rejection")
        temp_epochs.info['bads'] = bad_channels_non_seed
        json_data['bad_trials'] = bad_trials.tolist()
        mask = ~np.isin(temp_epochs.selection, json_data['bad_trials'])
        trialsToTake = temp_epochs.selection[mask]
        temp_epochs = temp_epochs[trialsToTake]

    # === Aggiornamenti json
    json_data['trials_tot'] = len(temp_epochs.annotations)
    json_data['trials_selected'] = temp_epochs.events.shape[0]
    json_data['bad_channels'] = temp_epochs.info['bads']

    # === Final steps
    temp_epochs = temp_epochs.resample(sfreq=json_data['r_sfreq'])
    temp_epochs = temp_epochs.set_eeg_reference('average')

    # Salva parametri
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
            json.dump(json_data, json_file, indent=4, sort_keys=True)

    return temp_epochs, json_data

def apply_notch_to_offsetChans(epochs, json_data, centers=[50], apply_to_all=False):
    from mne.filter import notch_filter

    sfreq = epochs.info['sfreq']
    data = epochs.get_data()
    data_filtered = data.copy()

    if apply_to_all:
        print(f"🔧 Applico filtro Notch *aggressivo* a TUTTI i canali | Frequenze: {centers} Hz")
        chan_idx = range(len(epochs.ch_names))
        json_data['notch_applied_chans'] = epochs.ch_names  # salva tutti
    else:
        chans = json_data.get('offsetChans', [])
        if not chans:
            print("⚠️ Nessun canale in 'offsetChans'. Nessun notch applicato.")
            return epochs, json_data
        chan_idx = [epochs.ch_names.index(ch) for ch in chans if ch in epochs.ch_names]
        print(f"🔧 Applico filtro Notch *aggressivo* a {len(chan_idx)} canali: {chans} | Frequenze: {centers} Hz")
        json_data['notch_applied_chans'] = chans

    # Applica filtro notch aggressivo solo sui canali selezionati
    data_filtered[:, chan_idx, :] = notch_filter(
        data[:, chan_idx, :],
        Fs=sfreq,
        freqs=centers,
        method='iir',
        iir_params=dict(
            ftype='butter',  # filtro Butterworth
            gpass=0.5,       # ripple massimo nel pass-band (dB)
            gstop=40,        # attenuazione minima nello stop-band (dB)
            order=12          # ordine elevato
        )
    )

    epochs._data = data_filtered
    return epochs, json_data

import numpy as np
import pickle
import warnings
from pathlib import Path

def run_detrend_pipeline(epochs, json_data, sub, experiment_dir, do_plot_variability=True):

    # --- GESTIONE TRIAL-WISE VS MEAN-WISE ---
    is_trials_wise = json_data.get('trials_wise', True)
    
    if not is_trials_wise:
        print("--- Mode: MEAN-WISE (Averaging epochs before detrend) ---")
        avg_data = epochs.get_data().mean(axis=0, keepdims=True)
        epochs_to_process = epochs.copy()[:1]
        epochs_to_process._data = avg_data.copy()
    else:
        print("--- Mode: TRIALS-WISE (Processing all individual trials) ---")
        epochs_to_process = epochs.copy()
    
    print("epochs_to_process data shape:", epochs_to_process.get_data().shape)
    print("epochs_to_process times len:", len(epochs_to_process.times))
    assert epochs_to_process.get_data().shape[2] == len(epochs_to_process.times), \
        f"{epochs_to_process.get_data().shape[2]} vs {len(epochs_to_process.times)}"
    
    # === CASE 1: windowed detrend attivo ===
    if json_data['do_detrend']:
        print(f"I am doing {json_data['detrend_type']} detrend")
        print('###')
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)

            epochs_detrended, mse_detrend, max_order_pre_list = computeDetrend_v6(
                epochs_to_process,
                json_data, experiment_dir, sub,
                detrendMode=json_data['detrend_type'],
                fitConstraint=json_data['detrend_fitConstraint'],
                typeOffsetRise=json_data['detrend_typeOffsetRise'],
                typeOffsetDecay=json_data['detrend_typeOffsetDecay'],
                correctMode=json_data['detrend_offsetCorrectionType'],
                oddSamples=json_data['detrend_offsetOddSamples'],
                offsetChans=json_data['offsetChans'],
                lag_correction=json_data['detrend_lag_correction'],
                doDetrendOnlyOffsetChans=json_data['do_detrend_onlyOffsetChans']
            )

        json_data[f'detrend_{json_data["detrend_type"]}_pars'] = [
            json_data['detrend_typeOffsetRise'],
            json_data['detrend_typeOffsetDecay']
        ]
        json_data['detrend_MSE'] = mse_detrend

        df_slopes_detrended = computeSlopes_v4(epochs_detrended, json_data, experiment_dir, sub)
        computeSlopesPlot(
            df_slopes_detrended,
            json_data, experiment_dir, sub,
            saveNote=f'ALL-DET_fit{json_data["detrend_fitConstraint"]}',
            subPath='2.detrend',
            sharex=True
        )
        detrendedEpochs = epochs_detrended
        if json_data['sourceData']!='SIMS':
            detrendedEpochs, json_data = notch_filter_offset_chans(detrendedEpochs, json_data)
            post_label = f"fit{json_data['detrend_fitConstraint']}"
            basicPlots(detrendedEpochs, json_data, experiment_dir, sub, key=f'{post_label}', subPath='2.detrend', show=False)
        else:
            post_label = f"fit{json_data['detrend_fitConstraint']}"
            basicPlots(detrendedEpochs, json_data, experiment_dir, sub, key=f'{post_label}', subPath='2.detrend', show=False)

        
    # === CASE 2: detrend disattivato ===
    else:
        print('I am not doing windowed detrend')
        print('###')
        json_data['detrend_polOrder_preOffset'] = np.NaN
        #json_data['detrend_polOrder_Offset'] = np.NaN
        #json_data['detrend_polOrder_postOffset'] = np.NaN
        json_data['detrend_offsetCorrectionType'] = np.NaN
        json_data['detrend_offsetOddSamples'] = np.NaN

        detrendedEpochs = epochs

        if json_data['detrend_overall']:
            def nonlinear_detrend(signal, order=3):
                times = np.arange(len(signal))
                poly_coeffs = np.polyfit(times, signal, order)
                trend = np.polyval(poly_coeffs, times)
                return signal - trend

            order = json_data['detrend_noWindowedOrder']
            detrendedEpochs = detrendedEpochs.apply_function(lambda x: nonlinear_detrend(x, order=order))
            basicPlots(detrendedEpochs, json_data, experiment_dir, sub, key=f'overallPolyOrder{order}', subPath='2.detrend', show=False)

    # === Salvataggio ===
    pkl_path = Path(experiment_dir) / '7.pkls' / f'{sub}_detrendedEpochs.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(detrendedEpochs, f)

    json_data['cn_detrendedEpochs'] = compute_condition_number_epochs_average(detrendedEpochs)

    # === Opzionale: plottaggio variabilità TEP per canale ===
    if do_plot_variability:
        for name in detrendedEpochs.ch_names:
            plotTrialTepVariability(detrendedEpochs, json_data, experiment_dir, sub, chanNAME=name, operator=np.mean, save=True, parDir='postDetrend')

    return detrendedEpochs, json_data

def notch_filter_offset_chans(epochs, json_data):
    """
    Applica notch filter IIR ai canali presenti in json_data['offsetChans']
    per rimuovere armoniche della powerline.

    Parametri:
    - epochs: mne.Epochs object
    - json_data: dict contenente 'powerline_freq' e 'offsetChans'

    Output:
    - epochs modificato (in-place)
    """

    from mne.filter import notch_filter

    data = epochs.get_data()
    sfreq = epochs.info['sfreq']

    centers = [
        json_data['powerline_freq'],
        json_data['powerline_freq'] * 2,
        json_data['powerline_freq'] * 3,
        json_data['powerline_freq'] * 4,
        json_data['powerline_freq'] * 5
    ]

    chans = json_data.get('offsetChans', [])
    chan_idx = [epochs.ch_names.index(ch) for ch in chans if ch in epochs.ch_names]

    if chan_idx:
        print(f"🔧 Applico notch IIR a canali offset {chans} su frequenze: {centers}")
        for f in centers:
            """
            data[:, chan_idx, :] = notch_filter(
                data[:, chan_idx, :],
                Fs=sfreq,
                freqs=f,
                method='iir',
                iir_params=dict(
                    ftype='butter',
                    gpass=0.5,    # ripple massimo nel pass-band (dB)
                    gstop=60,   # attenuazione minima nello stop-band (dB)
                    order=8     # ordine del filtro
                )
            )
            """
            data[:, chan_idx, :] = notch_filter(
                                data[:, chan_idx, :],
                                Fs=sfreq,
                                freqs=centers,            # tutti i notch insieme
                                method='fir',
                                notch_widths=2,         # più largo (es. ±1.5 Hz)
                                trans_bandwidth=2.0,      # zona di transizione
                            )
        epochs._data = data
    else:
        print("⚠️ Nessun canale offset trovato, nessun notch applicato.")

    return epochs, json_data

from scipy import signal
import numpy as np

def apply_notch_filter(data, fs, notch_freqs, quality_factor=30):
    """
    Applica un filtro notch alle frequenze specificate.

    Parameters:
    - data: array 1D (n_samples) o 2D (n_channels x n_samples)
    - fs: frequenza di campionamento (Hz)
    - notch_freqs: lista delle frequenze notch (Hz) da eliminare
    - quality_factor: Q-factor del filtro notch (maggiore = più stretto, default=30)

    Returns:
    - filtered_data: array filtrato della stessa forma di 'data'
    """
    filtered_data = np.copy(data)

    for f0 in notch_freqs:
        b, a = signal.iirnotch(f0, quality_factor, fs)
        filtered_data = signal.filtfilt(b, a, filtered_data, axis=-1)

    return filtered_data

def prepare_epochs(raw, events, temp_epochs, json_data, experiment_dir, sub):
    from pathlib import Path
    import pickle
    import json
    import mne

    epochs = mne.Epochs(
        raw,
        events,
        tmin=json_data["epochs_timewindow_min"],
        tmax=json_data["epochs_timewindow_max"],
        detrend=None,
        preload=True
    )

    epochs = epochs[temp_epochs.selection]
    epochs = epochs.pick("eeg")

    bad_channels = json_data.get("bad_channels", [])
    bad_channels = [ch for ch in bad_channels if ch in epochs.ch_names]

    if len(bad_channels) > 0:
        print(f"🧹 Dropping bad channels also from final epochs: {bad_channels}")
        epochs.drop_channels(bad_channels)

    epochs = epochs.resample(json_data["r_sfreq"])
    # epochs = epochs.set_eeg_reference("average")

    json_data["epochs_final_trials"] = int(len(epochs))
    json_data["epochs_final_channels"] = int(len(epochs.ch_names))
    json_data["epochs_final_ch_names"] = list(epochs.ch_names)
    json_data["cn_epochs"] = compute_condition_number_epochs_average(epochs)

    basicPlots(
        epochs,
        json_data,
        experiment_dir,
        sub,
        key="epochs",
        subPath="1.basic"
    )

    pkl_path = Path(experiment_dir) / "7.pkls" / f"{sub}_epochs.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(epochs, f)

    with open(Path(experiment_dir) / f"{sub}_pars.json", "w") as json_file:
        json.dump(json_data, json_file, indent=4, sort_keys=True)

    with open(pkl_path, "rb") as f:
        epochs = pickle.load(f)

    print("✅ prepare_epochs completed")
    print(f"   Final trials: {json_data['epochs_final_trials']}")
    print(f"   Final channels: {json_data['epochs_final_channels']}")
    print(f"   Final channel names: {json_data['epochs_final_ch_names']}")

    return epochs, json_data

"""
until 01/07/2026
def prepare_epochs(raw, events, temp_epochs, json_data, experiment_dir, sub):
    from pathlib import Path
    import pickle
    import mne

    epochs = mne.Epochs(
        raw,
        events,
        tmin=json_data['epochs_timewindow_min'], 
        tmax=json_data['epochs_timewindow_max'], 
        detrend=0,
        preload=True
    )
    epochs = epochs[temp_epochs.selection]
    epochs.info['bads'] = temp_epochs.info['bads']
    epochs = epochs.resample(json_data['r_sfreq'])
    epochs = epochs.pick('eeg')
    epochs = epochs.set_eeg_reference('average')
    basicPlots(epochs, json_data, experiment_dir, sub, key='epochs', subPath='1.basic')

    pkl_path = Path(experiment_dir) / '7.pkls' / f'{sub}_epochs.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(epochs, f)

    json_data['cn_epochs'] = compute_condition_number_epochs_average(epochs)

    # Ricarica opzionale (sanity check)
    with open(pkl_path, 'rb') as f:
        epochs = pickle.load(f)

    return epochs, json_data
"""

def analyze_offset_times(epochs, json_data, experiment_dir, sub, do_plot_variability=True):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from pathlib import Path

    offsetTimes = []
    for chan in tqdm(epochs.ch_names, desc="Processing channels"):
        id_chan = epochs.ch_names.index(chan)
        for id_trial in range(len(epochs)):
            maskPreOffset, maskOffset, maskPostOffset = computeTimeMasks(
                epochs, id_chan, id_trial, json_data, offset=json_data['detrend_maxTimeWindowOffset']
            )
            offsetTimes.append([
                chan, id_trial,
                epochs.times[maskOffset].min(),
                epochs.times[maskOffset].max()
            ])
    
    df = pd.DataFrame(offsetTimes, columns=['chan', 'trial', 'toffsetmin', 'toffsetmax'])
    df.to_csv(Path(experiment_dir) / '2.detrend' / 'offsetTimes_df.csv', index=False)

    mean_offset = df['toffsetmax'].mean()
    std_offset = df['toffsetmax'].std()
    hist_values, bin_edges = np.histogram(df['toffsetmax'], bins=15)
    mode_offset = bin_edges[np.argmax(hist_values)] + np.diff(bin_edges)[0]/2

    json_data['detrend_modeTimeWindowOffset'] = round(mode_offset, 3)
    json_data['detrend_meanTimeWindowOffset'] = round(mean_offset, 4)
    json_data['detrend_stdTimeWindowOffset'] = round(std_offset, 4)

    # Salva parametri
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
            json.dump(json_data, json_file, indent=4)


    if do_plot_variability:
        for name in tqdm(epochs.ch_names):
            plotTrialTepVariability(epochs, json_data, experiment_dir, sub, chanNAME=name, operator=np.mean, save=True, parDir='preDetrend')
    
    # see results in 3.trials
    return json_data, df

from pathlib import Path
import numpy as np

def check_detrend_need(epochs, json_data, experiment_dir, sub):

    print(f"\n🔍 [{sub}] Step 1: analisi delle latenze offset")
    json_data, df = analyze_offset_times(
        epochs,
        json_data,
        experiment_dir,
        sub,
        do_plot_variability=True
    )

    print(f"📉 [{sub}] Step 2: Calcolo delle pendenze (computeSlopes_v4)")
    df_slopes = computeSlopes_v4(epochs, json_data, experiment_dir, sub)

    print(f"📊 [{sub}] Step 3: Plot delle pendenze normalizzate (Zslope)")
    computeSlopesPlot(df_slopes, json_data, experiment_dir, sub, saveNote='ALL-TRIALS_preDetrend', zvalue=True)

    print(f"📈 [{sub}] Step 4: Calcolo media Zslope per canale e finestra")
    threshold = json_data['detrend_slopeThr']
    mean_df = df_slopes.groupby(['id_twindow', 'chan'], as_index=False)['Zslope'].mean()
    
    outliers_df = mean_df[np.abs(mean_df['Zslope']) >= threshold]
    found_outliers = not outliers_df.empty

    json_data['offsetChans'] = outliers_df['chan'].unique().tolist()
    json_data['do_detrend'] = found_outliers if json_data.get('do_detrend', True) else False

    # Plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 17))
    for ax, title, mask in zip(
        [ax1, ax2],
        ['All Channels', f'offsetChans Channels: {json_data["offsetChans"]}'],
        [df['chan'].isin(df['chan']), df['chan'].isin(json_data['offsetChans'])]
    ):
        hist_values, bin_edges = np.histogram(df[mask]['toffsetmax'], bins=15)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, hist_values, width=np.diff(bin_edges), alpha=0.3, edgecolor='black')
        ax.axvline(json_data['detrend_modeTimeWindowOffset'], color='purple', linestyle='-.', linewidth=3, label=f'Mode {json_data['detrend_modeTimeWindowOffset']:.4f}')
        ax.axvline(json_data['detrend_minTimeWindowOffset'], color='red', linestyle='--', linewidth=3, label='Min Detrend')
        ax.axvline(json_data['detrend_maxTimeWindowOffset'], color='red', linestyle='--', linewidth=3, label='Max Detrend')
        ax.axvspan(json_data['pulse_artifact_rej_timewindow_min'],
                   json_data['pulse_artifact_rej_timewindow_max'],
                   color='k', alpha=0.3, label='Pulse Artifact Window')
        ax.set_title(f'Histogram of toffsetmax — {title}', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency')
        #ax.set_xlim(json_data['detrend_minTimeWindowOffset'], json_data['detrend_maxTimeWindowOffset'])
        ax.grid(True)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.tight_layout()
    plot_path = Path(experiment_dir) / '2.detrend' / 'histogram_toffsetmax_subplots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n📌 [{sub}] Risultati:")
    print(f"   • Zslope threshold = {threshold}")
    print(f"   • Canali oltre soglia ({len(json_data['offsetChans'])}): {json_data['offsetChans']}")
    print(f"   • do_detrend = {json_data['do_detrend']}\n")

    # Salva parametri
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
            json.dump(json_data, json_file, indent=4, sort_keys=True)
        
    # === Salvataggio CSV dei risultati ===
    detrend_dir = Path(experiment_dir) / '2.detrend'
    detrend_dir.mkdir(parents=True, exist_ok=True)

    mean_df.to_csv(detrend_dir / 'mean_Zslope_per_chan_twindow.csv', index=False)
    outliers_df.to_csv(detrend_dir / 'outlier_Zslope.csv', index=False)

    return json_data

def add_TEP_to_json(json_file, postICA_final):
    """
    Aggiunge le versioni TEP_3d, TEP_2d, TEP_1d al dizionario json_file esistente.

    Parametri:
        - json_file: dict già esistente con altri metadati
        - postICA_final: oggetto mne.Epochs
    """
    data_3d = postICA_final.get_data()  # (epochs, chans, times)

    json_file['TEP_3d'] = np.transpose(data_3d, (1, 2, 0)).tolist()     # (chan, time, trial)
    json_file['TEP_2d'] = data_3d.mean(axis=0).tolist()                 # (chan, time)
    json_file['TEP_1d'] = data_3d.mean(axis=0).mean(axis=0).tolist()   # (time,)

    print("[INFO] Aggiunte TEP_3d, TEP_2d, TEP_1d a json_file")
    return json_file

def ICAprocessing(file,
                  json_data,
                  experiment_dir,
                  sub,
                  autoReject=True,
                  manualCheck=True,
                  computeFOOOF=False,
                  raw_to_apply=None):

    from pathlib import Path
    from datetime import datetime
    import os
    import json
    import pickle
    import matplotlib.pyplot as plt
    
    experiment_dir=str(Path(json_data.get("experiment_dir",experiment_dir)).expanduser().resolve())
    json_data["experiment_dir"]=experiment_dir
    sub=json_data.get("subject",sub)
    
    paths=rest_paths(experiment_dir)

    pkl_dir=paths["pkls"]
    pkl_dir.mkdir(parents=True,exist_ok=True)

    eeg_type=json_data.get("eeg_type","tms").lower()

    label_prob_threshold=json_data.get("do_label_prob_threshold",0.8)
    threshold_percentile=json_data.get("do_ica_eigThresh",0)

    timestamp=datetime.now().strftime("%Y%m%d%H%M%S")

    postica_dir=paths["postICA"]/timestamp
    postica_dir.mkdir(parents=True,exist_ok=True)

    json_data["ICA_timestamp"]=timestamp
    json_data["postICA_dir"]=str(postica_dir)

    if isinstance(file,str) and file.endswith(".pkl"):
        with open(file,"rb") as f:
            epochs_ica=pickle.load(f)
    else:
        epochs_ica=file

    print(f"⚙️ [{sub}] ICA fit su epochs")

    postICA_epochs,ica_model=run_ica_filtering_v3(
        epochs_ica,
        json_data,
        postica_dir,
        sub,
        autoReject=autoReject,
        manualCheck=manualCheck,
        label_prob_threshold=label_prob_threshold,
        threshold_percentile=threshold_percentile
    )

    postICA_raw_continuum=None

    if raw_to_apply is not None:

        print(f"🧼 [{sub}] Applicazione ICA al continuo")

        raw_apply=raw_to_apply.copy().pick("eeg")
        # raw_apply.set_eeg_reference("average")

        postICA_raw_continuum=ica_model.apply(raw_apply)

        with open(
            pkl_dir/f"{timestamp}_{sub}_postICA_raw_continuum.pkl",
            "wb"
        ) as f:
            pickle.dump(postICA_raw_continuum,f)

        fig=postICA_raw_continuum.compute_psd(
            fmin=json_data.get("l_freq",0.5),
            fmax=json_data.get("h_freq",45)
        ).plot(
            show=False
        )

        fig.savefig(
            postica_dir/f"{sub}_postICA_raw_continuum_PSD.png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

    json_data["ICA_components_tot"]=int(ica_model.n_components_)
    json_data["ICA_excludedComponents"]=[
        int(x) for x in ica_model.exclude
    ]

    json_data["ICA_includedComponents_tot"]=int(
        ica_model.n_components_-len(ica_model.exclude)
    )

    with open(
        pkl_dir/f"{timestamp}_{sub}_postICA_epochs_5s.pkl",
        "wb"
    ) as f:
        pickle.dump(postICA_epochs,f)


    with open(
        pkl_dir/f"{timestamp}_{sub}_ica_model.pkl",
        "wb"
    ) as f:
        pickle.dump(ica_model,f)


    if postICA_raw_continuum is not None:

        with open(
            pkl_dir/f"{timestamp}_{sub}_postICA_raw_continuum.pkl",
            "wb"
        ) as f:
            pickle.dump(postICA_raw_continuum,f)


    postICA_final,json_data=postICAsteps(
        postICA_raw_continuum,
        json_data,
        experiment_dir,
        sub
    )

    if computeFOOOF:

        extract_psd_features(
            postICA_epochs,
            "postICA_epochs_5s",
            experiment_dir,
            json_data
        )


    json_data_clean=make_json_serializable(json_data)

    with open(
        Path(experiment_dir)/f"{sub}_pars.json",
        "w"
    ) as f:
        json.dump(
            json_data_clean,
            f,
            indent=4,
            sort_keys=True
        )


    print(f"✅ [{sub}] ICA REST completata")
    print(f"   epochs ICA: {type(postICA_epochs)}")
    print(f"   raw continuum: {type(postICA_raw_continuum)}")
    print(f"   excluded IC: {ica_model.exclude}")


    return (
        postICA_epochs,
        postICA_raw_continuum,
        ica_model,
        json_data
    )


def computeFeatExtraction(postICA_final, json_data, experiment_dir, sub):
    # 1
    times = postICA_final.times
    seed_indices = [postICA_final.ch_names.index(chan) for chan in json_data["seedChans"] if chan in postICA_final.ch_names]
    json_data['feats_step'] = np.mean(postICA_final.average().get_data()[seed_indices, :], axis=0)
    reduced_data = json_data['feats_step']
    reduced_data, times_filtered = reduced_data[np.where(times > 0)[0]], times[np.where(times > 0)[0]]
    data = postICA_final.get_data()
    tep_integral = scipy.integrate.trapezoid(np.abs(reduced_data), times_filtered)
    tep_energy =  np.mean(reduced_data ** 2)
    # complexity metrics
    #sampen = ant.sample_entropy(reduced_data)
    #perm_entropy = ant.perm_entropy(reduced_data)
    # save feats
    #json_data['feat_step_sampleEntropy'] = sampen
    #json_data['feat_step_permEntropy'] = perm_entropy
    json_data['feat_step_energy'] = tep_energy
    json_data['feat_step_integral'] = tep_integral
    # Salva parametri
    with open(Path(experiment_dir) / f'{sub}_parsjsontxt', 'w') as txt_file:
        for key, value in sorted(json_data.items()):
            txt_file.write(f'{key}: {value}\n')

    # 2
    corr_matrix = np.corrcoef(postICA_final.average().get_data())
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)  
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", #annot=False, fmt=".2f",
                yticklabels=postICA_final.ch_names, 
                xticklabels=postICA_final.ch_names,
                vmin=-1, vmax=1, 
                linewidths=0.5, cbar=True)
    plt.title("Lower Triangular Correlation Matrix (Without Diagonal)")
    plt.savefig(f'{experiment_dir}\\5.final\\FE\\{sub}_FE_corrMatrix.png')
    plt.close()
    
    # 3    
    corr_matrix = np.corrcoef(postICA_final.average().get_data())
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)  
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix[seed_indices,:], 
                 #mask=mask, 
                cmap="coolwarm", #annot=False, fmt=".2f",
                yticklabels=postICA_final.ch_names, 
                xticklabels=postICA_final.ch_names,
                vmin=-1, vmax=1, 
                linewidths=0.5, cbar=True)
    plt.title("Lower Triangular Correlation Matrix (Without Diagonal)")
    plt.savefig(f'{experiment_dir}\\5.final\\FE\\{sub}_FE_corrMatrix_seed.png')
    plt.close()

    # 4
    plt.figure(figsize=(FIGSIZE))
    plt.plot(postICA_final.times, postICA_final.average().get_data()[seed_indices, :].T , label=f'seed chans {json_data['seedChans']}', c='r')
    plt.plot(postICA_final.times, json_data['feats_step'], label='average TEP', linewidth=10)
    plt.xlabel("Time (ms)")
    plt.ylabel('Amplitude (µV) Seed TEP')
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.title('Average Seed TEP')
    plt.tight_layout()
    plt.savefig(f'{experiment_dir}\\5.final\\FE\\{sub}_FE_STEP.png')
    plt.close()

    seed_indices = [postICA_final.ch_names.index(chan) for chan in json_data["seedChans"] if chan in postICA_final.ch_names]
    signals = np.mean(postICA_final.average().get_data()[seed_indices, :], axis=0)
    len(signals.shape)

    # 5
    do_run=True
    if do_run:
        selected_peaks = selectTEPfeat(postICA_final,  json_data, experiment_dir, sub, seed=json_data['seedChans'])
        selected_peaks
    
    json_data_clean = make_json_serializable(json_data)
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
            json.dump(json_data, json_file, indent=4, sort_keys=True)
        
    return json_data

def computeFeatExtraction_v2(postICA_final, json_data, experiment_dir, sub):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.integrate
    import antropy as ant
    from fooof import FOOOF
    from pathlib import Path

    # 1 - Seed TEP features
    times = postICA_final.times
    seed_indices = [postICA_final.ch_names.index(chan) for chan in json_data["seedChans"] if chan in postICA_final.ch_names]
    json_data['feats_step'] = np.mean(postICA_final.average().get_data()[seed_indices, :], axis=0)
    reduced_data = json_data['feats_step']
    reduced_data, times_filtered = reduced_data[np.where(times > 0)[0]], times[np.where(times > 0)[0]]
    data = postICA_final.get_data()
    tep_integral = scipy.integrate.trapezoid(np.abs(reduced_data), times_filtered)
    tep_energy = np.mean(reduced_data ** 2)

    # Complexity metrics
    sampen = ant.sample_entropy(reduced_data)
    perm_entropy = ant.perm_entropy(reduced_data)

    # FOOOF metrics
    fs = 1.0 / (times[1] - times[0])
    from scipy.signal import welch
    f, psd = welch(reduced_data, fs=fs, nperseg=128)
    fm = FOOOF()
    fm.fit(f, psd)
    fooof_offset, fooof_exponent = fm.get_params('aperiodic_params')

    # Save features
    json_data['feat_step_energy'] = tep_energy
    json_data['feat_step_integral'] = tep_integral
    json_data['feat_step_sampleEntropy'] = sampen
    json_data['feat_step_permEntropy'] = perm_entropy
    json_data['feat_step_fooofOffset'] = fooof_offset
    json_data['feat_step_fooofExponent'] = fooof_exponent

    # Salva parametri
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as txt_file:
        for key, value in sorted(json_data.items()):
            txt_file.write(f'{key}: {value}\n')

    # 2 - Correlation matrix full
    corr_matrix = np.corrcoef(postICA_final.average().get_data())
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm",
                yticklabels=postICA_final.ch_names,
                xticklabels=postICA_final.ch_names,
                vmin=-1, vmax=1,
                linewidths=0.5, cbar=True)
    plt.title("Lower Triangular Correlation Matrix (Without Diagonal)")
    plt.savefig(f'{experiment_dir}/5.Extra/FE/{sub}_FE_corrMatrix.png')
    plt.close()

    # 3 - Correlation matrix seed vs all
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix[seed_indices, :],
                cmap="coolwarm",
                yticklabels=np.array(postICA_final.ch_names)[seed_indices],
                xticklabels=postICA_final.ch_names,
                vmin=-1, vmax=1,
                linewidths=0.5, cbar=True)
    plt.title("Seed vs All Correlation Matrix")
    plt.savefig(f'{experiment_dir}/5.Extra/FE/{sub}_FE_corrMatrix_seed.png')
    plt.close()

    # 4 - Plot TEP signal
    plt.figure(figsize=(10, 5))
    plt.plot(postICA_final.times, postICA_final.average().get_data()[seed_indices, :].T, label=f"seed chans {json_data['seedChans']}", c='r')
    plt.plot(postICA_final.times, json_data['feats_step'], label='average TEP', linewidth=3)
    plt.xlabel("Time (ms)")
    plt.ylabel('Amplitude (µV) Seed TEP')
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.title('Average Seed TEP')
    plt.tight_layout()
    plt.savefig(f'{experiment_dir}/5.Extra/FE/{sub}_FE_STEP.png')
    plt.close()

    # 5 - Peak selection
    do_run = True
    if do_run:
        selected_peaks = selectTEPfeat(postICA_final, json_data, experiment_dir, sub, seed=json_data['seedChans'])

    from scipy.signal import hilbert
    # 6 - Phase Locking Value (PLV)
    def compute_PLV(data, seed_indices):
        analytic_signal = hilbert(data, axis=-1)
        phase = np.angle(analytic_signal)
    
        seed_phase = np.mean(phase[seed_indices, :], axis=0)
        plv_values = []
    
        for i in range(data.shape[0]):
            phase_diff = seed_phase - phase[i, :]
            plv = np.abs(np.sum(np.exp(1j * phase_diff))) / len(phase_diff)
            plv_values.append(plv)
    
        return np.array(plv_values)
    
    # Compute PLV on average TEP
    avg_data = postICA_final.average().get_data()
    plv_seed_all = compute_PLV(avg_data, seed_indices)
    json_data['feat_step_meanPLV_seed'] = float(np.mean(plv_seed_all))
    json_data['feat_step_maxPLV_seed'] = float(np.max(plv_seed_all))

    json_data_clean = make_json_serializable(json_data)
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
            json.dump(json_data_clean, json_file, indent=4)

    return json_data


import matplotlib.pyplot as plt
import numpy as np
import os

def plot_TEP_1d_with_shading(epochs, output_path):
    """
    Plot della TEP media su canali e trial, con shading ± std, e salvataggio PNG.

    Parametri:
        - epochs: oggetto mne.Epochs (es. postICA_final)
        - output_path: percorso completo al file PNG da salvare
    """
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    times_ms = epochs.times * 1e3
    data_uV = data * 1e6  # conversione in microvolt

    mean_1d = data_uV.mean(axis=(0, 1))  # media su epoche e canali
    std_1d = data_uV.std(axis=(0, 1))    # std su epoche e canali

    plt.figure(figsize=(10, 5))
    plt.plot(times_ms, mean_1d, color='black', linewidth=1.5, label='Mean TEP')
    plt.fill_between(times_ms, mean_1d - std_1d, mean_1d + std_1d, 
                     color='gray', alpha=0.3, label='±1 STD')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (µV)')
    plt.xlim(-100, 400)
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Plot TEP_1d salvato in: {output_path}")



def saveLoadTestFinal(postICA_final,json_data,experiment_dir,sub,start_time):

    pkl_dir=Path(experiment_dir)/"7.pkls"
    pkl_dir.mkdir(parents=True,exist_ok=True)

    filePKL=pkl_dir/f"{sub}_notebookState.pkl"

    with open(filePKL,"wb") as f:
        pickle.dump({},f)

    if postICA_final is not None:
        with open(pkl_dir/f"{sub}_postICA_final.pkl","wb") as f:
            pickle.dump(postICA_final,f)

    json_data_clean=make_json_serializable(json_data)

    with open(Path(experiment_dir)/f"{sub}_pars.json","w") as f:
        json.dump(json_data_clean,f,indent=4)

    return json_data


def selectTEPfeat(EPOCH,  json_data, experiment_dir, sub, seed=['Cz', 'Fz']):
    
    times = EPOCH.times * 1000

    if seed==None:
        signals = EPOCH.average().get_data()
    else:
        seed_indices = [EPOCH.ch_names.index(chan) for chan in seed if chan in EPOCH.ch_names]
        signals = np.mean(EPOCH.average().get_data()[seed_indices, :], axis=0)

    
    ch_names = EPOCH.ch_names
    selected_peaks = []
    colors = itertools.cycle(plt.cm.get_cmap("tab10").colors)
    
    def find_nearest_channel(y_value):
        distances = [np.abs(y_value - np.mean(signals[i] * 1e6)) for i in range(len(ch_names))]
        min_distance = np.min(distances)
        if min_distance > np.std(signals) * 3:
            return np.nan
        return ch_names[np.argmin(distances)]

    def onclick(event):
        if event.xdata is None or event.ydata is None or event.dblclick or event.button != 1:
            return  # Ignora zoom, pan, doppi clic e tasti diversi dal sinistro
        
        root = tk.Tk()
        root.withdraw()
        peak_name = simpledialog.askstring("Peak Name", "Enter peak name (e.g., N15, P30):")
        if peak_name:
            nearest_channel = find_nearest_channel(event.ydata)
            peak_data = {
                "name": peak_name,
                "latency_ms": round(event.xdata, 2),
                "amplitude_uv": round(event.ydata, 2),
                "channel": nearest_channel if not pd.isna(nearest_channel) else "NaN"
            }
            selected_peaks.append(peak_data)
            ax.scatter(event.xdata, event.ydata, color='red', s=100, marker='o')
            fig.canvas.draw()

    root = tk.Tk()
    root.withdraw()

    fig, ax = plt.subplots(figsize=(10, 5))
    if len(signals.shape)!=1:
        for signal, ch_name, color in zip(signals, ch_names, colors):
            ax.plot(times, signal * 1e6, label=ch_name, color=color, alpha=0.8, linewidth=5)
    else:
        ax.plot(times, signals * 1e6, label=f'average seed of {seed}', color='k', alpha=0.8, linewidth=5)

    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title("Click to Select Peaks")
    ax.legend()
    fig.canvas.mpl_connect("button_press_event", onclick)

    #save_button = tk.Button(root, text="Save Peaks", command=save_to_json)
    #save_button.pack()
    
    #plt.show()
    plt.close()

    json_data['feat_tep_manual'] = selected_peaks
    json_data_clean = make_json_serializable(json_data)
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
            json.dump(json_data_clean, json_file, indent=4, sort_keys=True)

def set_plot_params(fontsize=16):
    plt.rcParams.update({
        'font.size': fontsize,            # Dimensione generale del font
        'axes.labelsize': fontsize,       # Dimensione dei label degli assi
        'axes.titlesize': fontsize,       # Dimensione del titolo degli assi
        'xtick.labelsize': fontsize,      # Dimensione dei tick dell'asse x
        'ytick.labelsize': fontsize,      # Dimensione dei tick dell'asse y
        'legend.fontsize': fontsize*0.75,      # Dimensione della legenda
        'figure.titlesize': fontsize      # Dimensione del titolo della figura
    })

FIGSIZE=19, 11
set_plot_params(fontsize=22)


import numpy as np

def shift_signal_by_mask(signal, mask):
    """
    Shift a signal forward in time by the length of a boolean mask.
    Pads zeros at the beginning and trims the end to maintain the same length.

    Parameters:
    - signal: 1D np.array
    - mask: boolean np.array

    Returns:
    - shifted_signal: 1D np.array, same shape as signal
    - n_shift: number of samples shifted
    """
    n_shift = np.sum(mask)
    shifted_signal = np.pad(signal, (n_shift, 0), mode='constant')[:len(signal)]
    return shifted_signal, n_shift

def polyfit_constrained_start(x, y, order, x0, y0):
    """
    Fit polinomiale di grado `order`, vincolato a passare da (x0, y0).
    Ritorna: trend_line, coeffs (inclusi i coef. vincolati, cioè con il termine costante = y0)
    """
    # Shift dei dati rispetto al vincolo
    x_shifted = x - x0
    y_shifted = y - y0

    # Costruzione della matrice di Vandermonde senza termine costante (che è y0)
    X = np.vander(x_shifted, N=order+1)[:, :-1]  # esclude il termine x^0

    # Fit dei coefficienti rimanenti
    coeffs_reduced = np.linalg.lstsq(X, y_shifted, rcond=None)[0]

    # Ricostruzione del polinomio completo: y0 + a₁(x−x₀) + a₂(x−x₀)² + ...
    trend_line = y0 + np.polyval(np.append(coeffs_reduced, 0), x_shifted)

    # Coefficienti del polinomio globale (per compatibilità)
    # NB: per avere anche il termine costante effettivo, ricostruiamo il polinomio globale
    full_poly = np.poly1d(np.append(coeffs_reduced, 0))  # append 0 per x^0
    poly_coeffs = full_poly.integ()(x - x0) * 0 + y0 + np.polyval(full_poly, x - x0)

    return trend_line, coeffs_reduced


import numpy as np

def compute_condition_number_epochs_average(epochs):
    """
    Calcola il numero di condizionamento della matrice dei dati EEG
    dell'oggetto `epochs.average()`, ovvero la media degli Epochs.

    Restituisce:
    - Il numero di condizionamento della matrice EEG media.
    """
    evoked_data = epochs.average().data  # Matrice (n_channels, n_times)

    # Calcolo del numero di condizionamento usando la SVD
    condition_number = np.linalg.cond(evoked_data)

    return condition_number

def run_ica_filtering_v3(EPOCHS, json_data, experiment_dir, sub,
                         n_components=None, manualCheck=True,
                         autoReject=True, label_prob_threshold=0,
                         threshold_percentile=75):

    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from mne.preprocessing import ICA
    from mne_icalabel import label_components
    import json
    
    save_dir=Path(experiment_dir).expanduser().resolve()
    save_dir.mkdir(parents=True,exist_ok=True)
    print(f"[DEBUG] run_ica_filtering_v3 save_dir: {save_dir}")
    print(f"[DEBUG] run_ica_filtering_v3 save_dir exists: {save_dir.exists()}")

    ica = ICA(n_components=n_components, method='fastica', random_state=42)
    ica.fit(EPOCHS)

    ic_labels = label_components(EPOCHS, ica, method='iclabel')
    labels = ic_labels['labels']

    artifact_tags = [
        'eye blink',
        'muscle artifact',
        'heart beat',
        'line noise',
        'channel noise',
        'other'
    ]

    auto_excluded = []
    low_eigen_excluded = []

    mixing_matrix = ica.mixing_matrix_
    eigenvalues = np.linalg.svd(mixing_matrix, compute_uv=False) ** 2
    threshold = np.percentile(eigenvalues, threshold_percentile)

    if autoReject:
        for i, label in enumerate(labels):
            probs = np.array(ic_labels['y_pred_proba'][i], ndmin=1)
            max_prob = probs.max()

            if label in artifact_tags and max_prob >= label_prob_threshold:
                print(f"❌ IC {i}: {label} (prob: {max_prob:.2f}) – escluso automaticamente")
                auto_excluded.append(i)
            else:
                print(f"✅ IC {i}: {label} (prob: {max_prob:.2f}) – mantenuto automaticamente")

        low_eigen_excluded = np.where(eigenvalues <= threshold)[0].tolist()

        print(f"[Auto-tagging] Componenti escluse per ICLabel: {auto_excluded}")
        print(f"[Autovalori] Componenti escluse eigenvalue <= {threshold:.4f}: {low_eigen_excluded}")

    else:
        print("🚫 Esclusione automatica disattivata.")

    initial_excluded = sorted(set(auto_excluded + low_eigen_excluded)) if autoReject else []
    ica.exclude = initial_excluded.copy()

    json_data["ICA_autoExcludedComponents"] = [int(x) for x in auto_excluded]
    json_data["ICA_lowEigenExcludedComponents"] = [int(x) for x in low_eigen_excluded]
    json_data["ICA_initialExcludedComponents"] = [int(x) for x in initial_excluded]

    all_components = set(np.arange(ica.n_components_))
    initial_remaining = sorted(list(all_components - set(initial_excluded)))

    if initial_excluded:
        fig1 = ica.plot_components(picks=initial_excluded, show_names=False, show=False)
        fig1.savefig(save_dir / f'{sub}_AUTO_excluded_ICAs.png')
        plt.close(fig1)

    if initial_remaining:
        fig2 = ica.plot_components(picks=initial_remaining, show_names=False, show=False)
        fig2.savefig(save_dir / f'{sub}_AUTO_included_ICAs.png')
        plt.close(fig2)

    fig, ax = plt.subplots(figsize=(10, 5))
    above_threshold = np.where(eigenvalues >= threshold)[0]
    below_threshold = np.where(eigenvalues < threshold)[0]

    ax.plot(
        below_threshold,
        eigenvalues[below_threshold],
        marker='o',
        linestyle='-',
        color='black',
        label='Eigenvalues'
    )

    ax.scatter(
        above_threshold,
        eigenvalues[above_threshold],
        color='red',
        label='Above threshold',
        zorder=3
    )

    ax.axhline(
        threshold,
        color='r',
        linestyle='--',
        label=f'Threshold ({threshold_percentile}° percentile)'
    )

    ax.set_xlabel("ICA Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Eigenvalues of ICA Components")
    ax.legend()
    fig.savefig(save_dir / f'{sub}_eigenvalueDist.png')
    plt.close(fig)

    components_dir = save_dir / 'components'
    components_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(ica.n_components_):
        tag = labels[idx] if labels is not None else 'Unknown'
        tag_clean = tag.replace('/', '_').replace(' ', '')

        fig = ica.plot_components(picks=idx, show=False)

        if isinstance(fig, list):
            for i, f in enumerate(fig):
                fname = components_dir / f"component_{idx}_{tag_clean}_view{i}.png"
                f.savefig(fname, dpi=150)
                plt.close(f)
        else:
            fname = components_dir / f"component_{idx}_{tag_clean}.png"
            fig.savefig(fname, dpi=150)
            plt.close(fig)

    if manualCheck:
        try:
            import tmspath_utils_adj

            print(f"🖱️ Manual ICA check. Componenti pre-marcate: {ica.exclude}")
#            ica = tmspath_utils_adj.ICApp(ica, EPOCHS.copy(), apply_baseline=True, json_data=json_data)  
            ica=tmspath_utils_adj.ICApp(
                    ica,
                    EPOCHS.copy(),
                    apply_baseline=True,
                    overview_avg_xlim=(None,None),
                    json_data=json_data
                )

        except ImportError:
            print("⚠️ tmspath_utils_adj non disponibile. Salto ispezione manuale.")

    final_excluded = sorted(set([int(x) for x in ica.exclude]))
    ica.exclude = final_excluded

    json_data["ICA_finalExcludedComponents"] = final_excluded
    json_data["ICA_manualAddedComponents"] = sorted(list(set(final_excluded) - set(initial_excluded)))
    json_data["ICA_manualRecoveredComponents"] = sorted(list(set(initial_excluded) - set(final_excluded)))

    print(f"📌 ICA initial excluded: {initial_excluded}")
    print(f"📌 ICA final excluded:   {final_excluded}")
    print(f"📌 ICA manual added:     {json_data['ICA_manualAddedComponents']}")
    print(f"📌 ICA manual recovered: {json_data['ICA_manualRecoveredComponents']}")

    postICA_clean = ica.apply(EPOCHS.copy())

    final_remaining = sorted(list(all_components - set(final_excluded)))

    if final_excluded:
        fig3 = ica.plot_components(picks=final_excluded, show_names=False, show=False)
        fig3.savefig(save_dir / f'{sub}_FINAL_excluded_ICAs.png')
        plt.close(fig3)

    if final_remaining:
        fig4 = ica.plot_components(picks=final_remaining, show_names=False, show=False)
        fig4.savefig(save_dir / f'{sub}_FINAL_included_ICAs.png')
        plt.close(fig4)

    with open(save_dir / f'{sub}_ICA_selection_summary.json', 'w') as f:
        json.dump({
            "auto_excluded": json_data["ICA_autoExcludedComponents"],
            "low_eigen_excluded": json_data["ICA_lowEigenExcludedComponents"],
            "initial_excluded": json_data["ICA_initialExcludedComponents"],
            "final_excluded": json_data["ICA_finalExcludedComponents"],
            "manual_added": json_data["ICA_manualAddedComponents"],
            "manual_recovered": json_data["ICA_manualRecoveredComponents"]
        }, f, indent=4)

    return postICA_clean, ica


"""
until 01/07/2026
def run_ica_filtering_v3(EPOCHS, json_data, experiment_dir, sub,
                         n_components=None, manualCheck=True,
                         autoReject=True, label_prob_threshold=0,
                         threshold_percentile=75):

    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from mne.preprocessing import ICA
    from mne_icalabel import label_components

    save_dir = Path(experiment_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ica = ICA(n_components=n_components, method='fastica', random_state=42)
    ica.fit(EPOCHS)

    ic_labels = label_components(EPOCHS, ica, method='iclabel')
    labels = ic_labels['labels']

    artifact_tags = ['eye blink', 'muscle artifact', 'heart beat', 'line noise', 'channel noise', 'other']
    auto_excluded = []
    low_eigen_excluded = []

    if autoReject:
        for i, label in enumerate(labels):
            probs = np.array(ic_labels['y_pred_proba'][i], ndmin=1)
            max_prob = probs.max()
            if label in artifact_tags and max_prob >= label_prob_threshold:
                print(f"❌ IC {i}: {label} (prob: {max_prob:.2f}) – escluso")
                auto_excluded.append(i)
            else:
                print(f"✅ IC {i}: {label} (prob: {max_prob:.2f}) – mantenuto")

        print(f"[Auto-tagging] Componenti escluse per label ICLabel: {auto_excluded}")

        mixing_matrix = ica.mixing_matrix_
        eigenvalues = np.linalg.svd(mixing_matrix, compute_uv=False) ** 2
        threshold = np.percentile(eigenvalues, threshold_percentile)
        low_eigen_excluded = np.where(eigenvalues <= threshold)[0].tolist()
        print(f"[Autovalori] Componenti escluse (eigenvalue < {threshold:.4f}): {low_eigen_excluded}")
    else:
        print("🚫 Esclusione automatica disattivata.")
        mixing_matrix = ica.mixing_matrix_
        eigenvalues = np.linalg.svd(mixing_matrix, compute_uv=False) ** 2
        threshold = np.percentile(eigenvalues, threshold_percentile)

    excluded_components = sorted(set(auto_excluded + low_eigen_excluded)) if autoReject else []
    ica.exclude = excluded_components

    postICA_raw = ica.apply(EPOCHS.copy())

    all_components = set(np.arange(ica.get_components().shape[1]))
    remaining_components = list(all_components - set(excluded_components))

    if excluded_components:
        fig1 = ica.plot_components(picks=excluded_components, show_names=False, show=False)
        fig1.savefig(save_dir / f'{sub}_excluded_ICAs.png')
        plt.close(fig1)

    if remaining_components:
        fig2 = ica.plot_components(picks=remaining_components, show_names=False, show=False)
        fig2.savefig(save_dir / f'{sub}_included_ICAs.png')
        plt.close(fig2)

    fig, ax = plt.subplots(figsize=(10, 5))
    above_threshold = np.where(eigenvalues >= threshold)[0]
    below_threshold = np.where(eigenvalues < threshold)[0]
    ax.plot(below_threshold, eigenvalues[below_threshold], marker='o', linestyle='-', color='black', label='Eigenvalues')
    ax.scatter(above_threshold, eigenvalues[above_threshold], color='red', label='Above Threshold', zorder=3)
    ax.axhline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold_percentile}° percentile)')
    ax.set_xlabel("ICA Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Eigenvalues of ICA Components (Above Threshold: {len(above_threshold)})")
    ax.legend()
    fig.savefig(save_dir / f'{sub}_eigenvalueDist.png')
    plt.close(fig)

    components_dir = save_dir / 'components'
    components_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Salvataggio componenti ICA in: {components_dir}")

    for idx in range(ica.n_components_):
        tag = labels[idx] if labels is not None else 'Unknown'
        tag_clean = tag.replace('/', '_').replace(' ', '')
        fig = ica.plot_components(picks=idx, show=False)
        if isinstance(fig, list):
            for i, f in enumerate(fig):
                fname = components_dir / f"component_{idx}_{tag_clean}_view{i}.png"
                f.savefig(fname, dpi=150)
                plt.close(f)
        else:
            fname = components_dir / f"component_{idx}_{tag_clean}.png"
            fig.savefig(fname, dpi=150)
            plt.close(fig)

    if manualCheck:
        try:
            import tmspath_utils_adj
            ica = tmspath_utils_adj.ICApp(ica, postICA_raw)
            postICA_clean = ica.apply(postICA_raw.copy())
        except ImportError:
            print("⚠️ tmspath_utils_adj non disponibile. Salto ispezione manuale.")
            postICA_clean = postICA_raw
    else:
        postICA_clean = postICA_raw

    return postICA_clean, ica
"""

def run_ica_filtering_v3_old_20260416(EPOCHS, json_data, experiment_dir, sub,
                         n_components=None, manualCheck=True,
                          autoReject=True, label_prob_threshold=0,
                          threshold_percentile=75,
                          subPath='4.postICA',
                          ):

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from mne.preprocessing import ICA
    from mne_icalabel import label_components

    # ICA decomposition
    ica = ICA(n_components=n_components, method='fastica', random_state=42)
    ica.fit(EPOCHS)

    # 1. Auto-tagging delle componenti
    ic_labels = label_components(EPOCHS, ica, method='iclabel')
    labels = ic_labels['labels']

    artifact_tags = ['eye blink', 'muscle artifact', 'heart beat', 'line noise', 'channel noise', 'other']
    auto_excluded = []
    low_eigen_excluded = []

    # 2. Esclusione automatica (opzionale)
    if autoReject:
        for i, label in enumerate(labels):
            probs = np.array(ic_labels['y_pred_proba'][i], ndmin=1)
            max_prob = probs.max()
            if label in artifact_tags and max_prob >= label_prob_threshold:
                print(f"❌ IC {i}: {label} (prob: {max_prob:.2f}) – escluso")
                auto_excluded.append(i)
            else:
                print(f"✅ IC {i}: {label} (prob: {max_prob:.2f}) – mantenuto")

        print(f"[Auto-tagging] Componenti escluse per label ICLabel: {auto_excluded}")

        # 3. Filtro per soglia su autovalori
        mixing_matrix = ica.mixing_matrix_
        eigenvalues = np.linalg.svd(mixing_matrix, compute_uv=False) ** 2
        threshold = np.percentile(eigenvalues, threshold_percentile)
        low_eigen_excluded = np.where(eigenvalues <= threshold)[0].tolist()
        print(f"[Autovalori] Componenti escluse (eigenvalue < {threshold:.4f}): {low_eigen_excluded}")
    else:
        print("🚫 Esclusione automatica disattivata.")
        mixing_matrix = ica.mixing_matrix_
        eigenvalues = np.linalg.svd(mixing_matrix, compute_uv=False) ** 2
        threshold = np.percentile(eigenvalues, threshold_percentile)

    excluded_components = sorted(set(auto_excluded + low_eigen_excluded)) if autoReject else []
    ica.exclude = excluded_components

    # 4. Applica ICA per rimuovere componenti escluse
    postICA_raw = ica.apply(EPOCHS.copy())

    # 5. Salvataggio grafici riassuntivi
    all_components = set(np.arange(ica.get_components().shape[1]))
    remaining_components = list(all_components - set(excluded_components))

    if excluded_components:
        fig1 = ica.plot_components(picks=excluded_components, show_names=False, show=False)
        fig1.savefig(os.path.join(experiment_dir, subPath, f'{sub}_excluded_ICAs.png'))
        plt.close(fig1)

    if remaining_components:
        fig2 = ica.plot_components(picks=remaining_components, show_names=False, show=False)
        fig2.savefig(os.path.join(experiment_dir, subPath, f'{sub}_included_ICAs.png'))
        plt.close(fig2)

    fig, ax = plt.subplots(figsize=(10, 5))
    above_threshold = np.where(eigenvalues >= threshold)[0]
    below_threshold = np.where(eigenvalues < threshold)[0]
    ax.plot(below_threshold, eigenvalues[below_threshold], marker='o', linestyle='-', color='black', label='Eigenvalues')
    ax.scatter(above_threshold, eigenvalues[above_threshold], color='red', label='Above Threshold', zorder=3)
    ax.axhline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold_percentile}° percentile)')
    ax.set_xlabel("ICA Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Eigenvalues of ICA Components (Above Threshold: {len(above_threshold)})")
    ax.legend()
    fig.savefig(os.path.join(experiment_dir, subPath, f'{sub}_eigenvalueDist.png'))
    plt.close(fig)

    # === SALVA COMPONENTI ICA SINGOLE CON TAG ===
    components_dir = os.path.join(experiment_dir, subPath, 'components')
    os.makedirs(components_dir, exist_ok=True)

    print(f"[INFO] Salvataggio componenti ICA in: {components_dir}")

    for idx in range(ica.n_components_):
        tag = labels[idx] if labels is not None else 'Unknown'
        tag_clean = tag.replace('/', '_').replace(' ', '')
        fig = ica.plot_components(picks=idx, show=False)
        if isinstance(fig, list):
            for i, f in enumerate(fig):
                fname = os.path.join(components_dir, f"component_{idx}_{tag_clean}_view{i}.png")
                f.savefig(fname, dpi=150)
                plt.close(f)
        else:
            fname = os.path.join(components_dir, f"component_{idx}_{tag_clean}.png")
            fig.savefig(fname, dpi=150)
            plt.close(fig)

    # 6. Controllo manuale finale (opzionale)
    if manualCheck:
        try:
            import tmspath_utils_adj
            ica = tmspath_utils_adj.ICApp(ica, postICA_raw)
            postICA_clean = ica.apply(postICA_raw.copy())
        except ImportError:
            print("⚠️ tmspath_utils_adj non disponibile. Salto ispezione manuale.")
            postICA_clean = postICA_raw
    else:
        postICA_clean = postICA_raw

    return postICA_clean, ica

def postICAsteps(postICA_raw,json_data,experiment_dir,sub):
    from pathlib import Path
    import pickle
    import json

    paths=rest_paths(experiment_dir)

    postICA_final=postICA_raw.copy().filter(
        l_freq=json_data["l_freq"],
        h_freq=json_data["h_freq"],
        method="fir",
        verbose=True
    )

    newrate=json_data["sfreq"]
    postICA_final=postICA_final.resample(sfreq=newrate)

    bad_channels=json_data.get("bad_channels",[])
    bad_channels=[ch for ch in bad_channels if ch in postICA_final.ch_names]

    if len(bad_channels)>0:
        postICA_final.info["bads"]=bad_channels
        postICA_final.interpolate_bads(reset_bads=True)

    postICA_final=postICA_final.pick("eeg")

    json_data["postICA_final_type"]="Raw_REST_postICA_filtered_interpolated"
    json_data["postICA_final_l_freq"]=float(json_data["l_freq"])
    json_data["postICA_final_h_freq"]=float(json_data["h_freq"])
    json_data["channel_interpolation_timing"]="after_ica_postICAsteps"
    json_data["channels_interpolated_after_ica"]=bad_channels

    with open(paths["pkls"]/f"{sub}_postICA_final.pkl","wb") as f:
        pickle.dump(postICA_final,f)

    postICA_final.save(
        paths["pkls"]/f"{sub}_postICA_final.fif",
        overwrite=True
    )

    json_data_clean=make_json_serializable(json_data)

    with open(Path(experiment_dir)/f"{sub}_pars.json","w") as json_file:
        json.dump(json_data_clean,json_file,indent=4,sort_keys=True)

    return postICA_final,json_data


def plot_ersp(postICA_final, channel=['Cz', 'Fx'], subDir='4.postICA', saveNote='postICA'):
    do_run=1
    if do_run:
        # plot of event-related spectral perturbations 
        # for a single channel specified by the user
        cols = 8 if len(raw.ch_names)==31 else 6
        fig, ax = plt.subplots(4, cols, figsize=(13, 11))
        ax = ax.flatten()
        fig.tight_layout()
        plt.subplots_adjust(wspace=1.25, hspace=1)
        for i, FEAT in enumerate(raw.ch_names):
            print(FEAT)
            ersp, freqs = tmspath_utils_adj.plot_ersp(postICA_final, FEAT, n_cycle=2, show=False, ax=ax[i])
        fig.savefig(f'{experiment_dir}\\{subDir}\\{sub}_ERSP_{saveNote}.png')

def plot_topomap(postICA_final, 
                 json_data, experiment_dir, sub,
                 subDir='4.postICA',saveNote='postICA'):
    do_run=1
    if do_run:
        # Concatenare gli intervalli
        times = [postICA_final.times.min(), 0, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070, 0.080, 0.090, 0.100, 0.200, 0.299, postICA_final.times.max()]
        fig=postICA_final.average().plot_topomap(times, show=False,
                             ch_type='eeg', 
                         #mask=mask, mask_params=mask_params, 
                         #image_interp="linear", 
                         contours=10)
                
        from pathlib import Path
        out_dir=Path(experiment_dir)/subDir
        out_dir.mkdir(parents=True,exist_ok=True)
        fig.savefig(out_dir/f"{sub}_scalpmaptime_{saveNote}.png")       
        plt.close()

        times = np.linspace(postICA_final.times.min(), postICA_final.times.max())
        evoked = postICA_final.average()
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))  # Assicuriamoci che ci sia un solo asse
        fig.suptitle("Scalp Topomap over Time")
        def update(frame):
            ax.clear()  # Puliamo l'asse per il nuovo tempo
            evoked.plot_topomap(times=[frame], ch_type='eeg', contours=10, axes=ax, show=False, colorbar=False)
            ax.set_title(f"Time: {frame:.3f} s")
        ani = FuncAnimation(fig, update, frames=times, repeat=False)
        output_path = os.path.join(experiment_dir, f"{subDir}/{sub}_scalpmaptime_{saveNote}.gif")
        ani.save(output_path, writer='ffmpeg', fps=2)  # Cambia fps per regolare la velocità
        print(f"Video salvato in: {output_path}")
        plt.close()

def plot_gmfp(postICA_final, json_data, experiment_dir, sub, FEAT=['Cz', 'Fx']):
    do_run=1
    if do_run:
        # plot of either 
        # GMFP time course (if channels='all') or 
        # LMFP (if channels = ['Fz', 'Cz',...]
        #FEAT = json_data['seedChans'] #'all'
        g_l_mfp = tmspath_utils_adj.plot_gmfp(postICA_final, channels=FEAT, show=False)
        json_data['feats_smfp'] = g_l_mfp
        # add saving
    return g_l_mfp


def find_outlier_channels_by_twindow_v2(df_slopes, threshold=3):
    """
    Identifica i canali che hanno almeno un trial con |Zslope| > threshold
    in ogni finestra temporale (id_twindow).
    
    Parametri:
    - df_slopes: DataFrame con colonne 'id_twindow', 'chan', 'Zslope'.
    - threshold: soglia in unità Z-score (default=3).
    
    Ritorna:
    - outlier_channels: dict con time window come chiave e lista di canali outlier come valore.
    - found_outlier: True se almeno un outlier è stato trovato.
    """
    outlier_channels = {}
    found_outlier = False

    for time_window in df_slopes['id_twindow'].unique():
        df_time_window = df_slopes[df_slopes['id_twindow'] == time_window]
        outliers = []

        for chan in df_time_window['chan'].unique():
            zvals = df_time_window[df_time_window['chan'] == chan]['Zslope']
            if any(abs(zvals) > threshold):
                outliers.append(chan)

        if outliers:
            outlier_channels[time_window] = outliers
            found_outlier = True

    return outlier_channels, found_outlier


def find_outlier_channels_by_twindow(df_slopes, threshold=3):
    """
    Identifica i canali che hanno una media di ZSlope superiore a 3 sigma in ogni finestra temporale.
    
    Parametri:
    - df_slopes: DataFrame contenente le colonne 'id_twindow', 'chan' e 'Zslope'.
    - threshold: Soglia in deviazioni standard per considerare un canale anomalo (default=3).
    
    Ritorna:
    - Dizionario con finestre temporali come chiavi e liste di canali outlier come valori.
    - Booleano True se almeno una finestra temporale contiene outlier, altrimenti False.
    """
    # Dizionario per contenere i canali outlier per ogni finestra temporale
    outlier_channels = {}
    found_outlier = False

    # Loop su ogni finestra temporale unica
    for time_window in df_slopes['id_twindow'].unique():
        # Filtra i dati per la finestra temporale corrente
        df_time_window = df_slopes[df_slopes['id_twindow'] == time_window]
        
        # Calcola la media e la deviazione standard di ZSlope per tutti i canali
        mean_zslope = df_time_window.groupby('chan')['Zslope'].mean()
        std_zslope = df_time_window.groupby('chan')['Zslope'].std()

        # Identifica i canali con valore medio di ZSlope superiore alla soglia di 3 sigma
        outliers = mean_zslope[mean_zslope > threshold * std_zslope].index.tolist()

        # Salva gli outlier se presenti
        if outliers:
            outlier_channels[time_window] = outliers
            found_outlier = True  # Almeno un outlier trovato

    return outlier_channels, found_outlier



def find_outlier_channels_by_twindow_v3(df_slopes, threshold=3):
    """
    Identifica i canali che hanno una media di ZSlope superiore alla media globale ± threshold * std 
    in ogni finestra temporale, e stampa lo Z-score medio per ogni canale outlier.
    
    Parametri:
    - df_slopes: DataFrame con colonne 'id_twindow', 'chan' e 'Zslope'.
    - threshold: Soglia in deviazioni standard per considerare un canale anomalo (default=3).
    
    Ritorna:
    - Dizionario con finestre temporali come chiavi e liste di canali outlier come valori.
    - Booleano True se almeno una finestra temporale contiene outlier, altrimenti False.
    """
    outlier_channels = {}
    found_outlier = False

    for time_window in df_slopes['id_twindow'].unique():
        df_time_window = df_slopes[df_slopes['id_twindow'] == time_window]

        # Calcola la media Zslope per ciascun canale
        mean_zslope_per_chan = df_time_window.groupby('chan')['Zslope'].mean()

        # Calcola media e std globale tra canali
        global_mean = mean_zslope_per_chan.mean()
        global_std = mean_zslope_per_chan.std()

        # Calcola Z-score normalizzato per ciascun canale
        zscore = (mean_zslope_per_chan - global_mean) / global_std

        # Identifica outlier sopra o sotto soglia
        outliers = zscore[abs(zscore) > threshold]

        if not outliers.empty:
            outlier_channels[time_window] = outliers.index.tolist()
            found_outlier = True

            print(f"\n[Time window: {time_window}] Canali outlier (Z-score > ±{threshold}):")
            for chan in outliers.index:
                print(f" - Canale: {chan}, Z-score medio: {zscore[chan]:.2f}")

    return outlier_channels, found_outlier


def run_ica_filtering(EPOCHS, n_components=None, manualCheck=True, threshold_percentile=75, subPath='4.postICA', saveNote='postICA'):
    
    #ic_labels, ica = runICA(EPOCHS)    
    ica = mne.preprocessing.ICA(n_components=n_components, method='fastica', random_state=42)
    ica.fit(EPOCHS)

    mixing_matrix = ica.mixing_matrix_
    eigenvalues = np.linalg.svd(mixing_matrix, compute_uv=False) ** 2  # Autovalori
    threshold = np.percentile(eigenvalues, threshold_percentile)
    excluded_components = np.where(eigenvalues <= threshold)[0].tolist()
    print(f"Componenti escluse (eigenvalue < {threshold:.4f}): {excluded_components}")
    ica.exclude = excluded_components
    postICA_raw = ica.apply(EPOCHS.copy())
    all_components = set(np.arange(ica.get_components().shape[1]))
    remaining_components = list(all_components - set(excluded_components))
    fig1 = ica.plot_components(picks=excluded_components, show_names=False, show=False)
    fig1.savefig(f'{experiment_dir}\\{subPath}\\{sub}_excluded_ICAs_{saveNote}.png')
    plt.close()
    fig2 = ica.plot_components(picks=remaining_components, show_names=False, show=False)
    fig2.savefig(f'{experiment_dir}\\{subPath}\\{sub}_included_ICAs_{saveNote}.png')
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 5))
    above_threshold = np.where(eigenvalues >= threshold)[0]  # Indici sopra soglia
    below_threshold = np.where(eigenvalues < threshold)[0]   # Indici sotto soglia
    ax.plot(below_threshold, eigenvalues[below_threshold], marker='o', linestyle='-', color='black', label='Eigenvalues')
    ax.scatter(above_threshold, eigenvalues[above_threshold], color='red', label='Above Threshold', zorder=3)
    ax.axhline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold_percentile}° percentile)')
    ax.set_xlabel("ICA Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Eigenvalues of ICA Components (Above Threshold: {len(above_threshold)})")
    ax.legend()
    fig.savefig(f'{experiment_dir}\\{subPath}\\{sub}_eigenvalueDist_{saveNote}.png')
    plt.close()
    if manualCheck: 
        ica = tmspath_utils_adj.ICApp(ica, postICA_raw)
        postICA_raw_bis = ica.apply(postICA_raw.copy())  
        postICA_clean = postICA_raw_bis
        ica.exclude = excluded_components

    if not manualCheck: 
        postICA_clean = postICA_raw

    return postICA_clean, ica


def run_ica_filtering_v2(EPOCHS, n_components=None, manualCheck=True, threshold_percentile=75,
                          subPath='4.postICA', saveNote='postICA', experiment_dir='.', sub='subject'):

    from mne.preprocessing import ICA
    from mne_icalabel import label_components
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    os.makedirs(os.path.join(experiment_dir, subPath), exist_ok=True)

    # ICA decomposition
    ica = ICA(n_components=n_components, method='fastica', random_state=42)
    ica.fit(EPOCHS)

    # 1. Auto-tagging delle componenti
    ic_labels = label_components(EPOCHS, ica, method='iclabel')
    labels = ic_labels['labels']

    artifact_tags = ['eye blink', 'muscle artifact', 'heart beat', 'line noise', 'channel noise', 'other']
    auto_excluded = [i for i, label in enumerate(labels) if label in artifact_tags]

    print(f"[Auto-tagging] Componenti escluse per label ICLabel: {auto_excluded}")

    # 2. Filtro per soglia su autovalori
    mixing_matrix = ica.mixing_matrix_
    eigenvalues = np.linalg.svd(mixing_matrix, compute_uv=False) ** 2
    threshold = np.percentile(eigenvalues, threshold_percentile)
    low_eigen_excluded = np.where(eigenvalues <= threshold)[0].tolist()

    print(f"[Autovalori] Componenti escluse (eigenvalue < {threshold:.4f}): {low_eigen_excluded}")

    # 3. Unione esclusioni
    excluded_components = sorted(set(auto_excluded + low_eigen_excluded))
    ica.exclude = excluded_components

    # 4. Applica ICA per rimuovere componenti escluse
    postICA_raw = ica.apply(EPOCHS.copy())

    # 5. Salvataggio grafici
    all_components = set(np.arange(ica.get_components().shape[1]))
    remaining_components = list(all_components - set(excluded_components))

    fig1 = ica.plot_components(picks=excluded_components, show_names=False, show=False)
    fig1.savefig(os.path.join(experiment_dir, subPath, f'{sub}_excluded_ICAs_{saveNote}.png'))
    plt.close()

    fig2 = ica.plot_components(picks=remaining_components, show_names=False, show=False)
    fig2.savefig(os.path.join(experiment_dir, subPath, f'{sub}_included_ICAs_{saveNote}.png'))
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    above_threshold = np.where(eigenvalues >= threshold)[0]
    below_threshold = np.where(eigenvalues < threshold)[0]
    ax.plot(below_threshold, eigenvalues[below_threshold], marker='o', linestyle='-', color='black', label='Eigenvalues')
    ax.scatter(above_threshold, eigenvalues[above_threshold], color='red', label='Above Threshold', zorder=3)
    ax.axhline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold_percentile}° percentile)')
    ax.set_xlabel("ICA Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Eigenvalues of ICA Components (Above Threshold: {len(above_threshold)})")
    ax.legend()
    fig.savefig(os.path.join(experiment_dir, subPath, f'{sub}_eigenvalueDist_{saveNote}.png'))
    plt.close()

    # 6. Controllo manuale finale
    if manualCheck:
        try:
            import tmspath_utils_adj
            ica = tmspath_utils_adj.ICApp(ica, postICA_raw)
            postICA_clean = ica.apply(postICA_raw.copy())
        except ImportError:
            print("tmspath_utils_adj non disponibile. Salto ispezione manuale.")
            postICA_clean = postICA_raw
    else:
        postICA_clean = postICA_raw

    return postICA_clean, ica



def run_ica_artist_ext_only(EPOCHS, n_components=None, ext_threshold_uv=30, manualCheck=True, subPath='4.postICA', saveNote='postICA'):
    import os
    os.makedirs(f'{experiment_dir}\\{subPath}', exist_ok=True)

    # ICA
    ica = mne.preprocessing.ICA(n_components=n_components,
                                method='fastica', 
                                random_state=42)
    ica.fit(EPOCHS)

    # Estrai le sorgenti ICA
    
    sources = ica.get_sources(EPOCHS).get_data() * 1e6  # μV
    
    if sources.ndim == 3:
        # EPOCHS: (n_components, n_epochs, n_times)
        max_abs = np.max(np.abs(sources), axis=(1, 2))
    elif sources.ndim == 2:
        # RAW: (n_components, n_times)
        max_abs = np.max(np.abs(sources), axis=1)
    else:
        raise ValueError("Formato sorgenti ICA non riconosciuto.")
    
    excluded_components = np.where(max_abs > ext_threshold_uv)[0].tolist()


    print(f"Componenti escluse (criterio EXT: |amp| > {ext_threshold_uv}μV in qualsiasi punto): {excluded_components}")
    ica.exclude = excluded_components

    # Applica ICA
    postICA_raw = ica.apply(EPOCHS.copy())

    # Salva i plot
    all_components = set(np.arange(ica.get_components().shape[1]))
    remaining_components = list(all_components - set(excluded_components))

    if manualCheck:
        ica = tmspath_utils_adj.ICApp(ica, postICA_raw)
        postICA_clean = ica.apply(postICA_raw.copy())
    else:
        postICA_clean = postICA_raw

    return postICA_clean, ica


def run_ica_artist_tms_events(raw, events, n_components=None, 
                              ext_threshold_uv=30, 
                              window_ms=50, 
                              manualCheck=True,
                              subPath='4.postICA', 
                              saveNote='postICA',
                              experiment_dir='./',
                              json_data=None,
                              demean_between_events=False):
    import os
    import numpy as np
    import mne

    os.makedirs(f'{experiment_dir}\\{subPath}', exist_ok=True)

    sfreq = raw.info['sfreq']
    n_samples_window = int(window_ms / 1000 * sfreq)
    lag_ms = json_data['baseline_cor_tmin']*-1000
    lag_samples = int(lag_ms / 1000 * sfreq)

    # ----------- Rimozione media opzionale su intervalli ritardati -----------
    if demean_between_events:
        eeg_data = raw.get_data(picks='eeg') * 1e6  # μV
        eeg_picks = mne.pick_types(raw.info, eeg=True)

        for i in range(len(events) - 1):
            start = events[i, 0] - lag_samples
            end = events[i + 1, 0] - lag_samples
            if start < 0 or end > eeg_data.shape[1] or start >= end:
                continue
            segment = eeg_data[:, start:end]
            segment_mean = np.mean(segment, axis=1, keepdims=True)
            eeg_data[:, start:end] -= segment_mean

        raw._data[eeg_picks] = eeg_data / 1e6  # torna in Volt

    # ----------- ICA -----------
    ica = mne.preprocessing.ICA(n_components=n_components, method='fastica', random_state=42)
    ica.fit(raw)

    sources = ica.get_sources(raw).get_data() * 1e6  # sorgenti in μV

    n_components = sources.shape[0]
    excluded_components = []

    for ic in range(n_components):
        ic_signal = sources[ic]
        for ev in events:
            onset = ev[0]
            end = onset + n_samples_window
            if end >= len(ic_signal):
                continue
            seg = ic_signal[onset:end]
            if np.any(np.abs(seg) > ext_threshold_uv):
                excluded_components.append(ic)
                break

    excluded_components = sorted(set(excluded_components))
    print(f"Componenti escluse (criterio EXT: ±{ext_threshold_uv}μV entro {window_ms}ms da eventi): {excluded_components}")
    ica.exclude = excluded_components

    postICA_raw = ica.apply(raw.copy())

    all_components = set(np.arange(n_components))
    remaining_components = list(all_components - set(excluded_components))
    postICA_clean = postICA_raw
    
    # if manualCheck:
    #    ica = tmspath_utils_adj.ICApp(ica, postICA_raw, apply_baseline = False)
    #    postICA_clean = ica.apply(postICA_raw.copy())
    # else:
    #    postICA_clean = postICA_raw

    return postICA_clean, ica


def plot_customTEP(EPOCHS, subDir, key, FIGSIZE):
    # Calcolo e configurazione dei dati
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)  # Grafico più largo per una migliore visualizzazione

    mat = np.mean(EPOCHS.get_data(), axis=0).T
    times = EPOCHS.times
    
    # Traccia la linea del segnale
    b = ax.plot(
        times, 
        mat, 
        c='blue', 
        #label='REF', 
        alpha=0.7, 
        linewidth=1.5  # Linea più visibile
    )
    
    # Linee verticali
    ax.axvline(
        x=json_data['pulse_artifact_rej_timewindow_min'], 
        linewidth=2.5, 
        c='black', 
        alpha=0.8, 
        linestyle='--', 
        label='Pulse Artifact Start'
    )
    ax.axvline(
        x=json_data['pulse_artifact_rej_timewindow_max'], 
        linewidth=2.5, 
        c='black', 
        alpha=0.8, 
        linestyle='--', 
        label='Pulse Artifact End'
    )
    ax.axvline(
        x=0,
        linewidth=2.5, 
        c='red', 
        alpha=0.8, 
        linestyle='-.', 
        label='Offset Detrend Start'
    )
    ax.axvline(
        x=json_data['detrend_modeTimeWindowOffset']  + (json_data['detrend_offsetOddSamples'] * 1e-3), 
        linewidth=2.5, 
        c='red', 
        alpha=0.8, 
        linestyle='-.', 
        label='Offset Detrend End'
    )

    # Ombreggiatura per la finestra del Offset
    ax.fill_between(
        times, 
        mat.min(),  # Valore minimo del segnale
        mat.max(),  # Valore massimo del segnale
        where=(
            (times >= 0) & 
            (times <= json_data['detrend_modeTimeWindowOffset']  + (json_data['detrend_offsetOddSamples'] * 1e-3))
        ),
        color='red', 
        alpha=0.3, 
        label='Offset Detrend Window'
    )
    
    # Ombreggiatura per la finestra del pulse artifact
    ax.fill_between(
        times, 
        mat.min(),  # Valore minimo del segnale
        mat.max(),  # Valore massimo del segnale
        where=(
            (times >= json_data['pulse_artifact_rej_timewindow_min']) & 
            (times <= json_data['pulse_artifact_rej_timewindow_max'])
        ),
        color='gray', 
        alpha=0.3, 
        label='Pulse Artifact Rejection Window'
    )

    ax.set_xlim(-0.1, 0.4)
    ax.set_ylim(mat.min() + 0 * abs(mat.min()), mat.max() + 0 * abs(mat.max()))  # Margini extra per evitare tagli
    ax.set_xlabel("Time (s)", fontweight='bold')
    ax.set_ylabel("Amplitude (μV)", fontweight='bold')
    ax.tick_params(axis='both', which='major')
    ax.set_title(f'{sub} - {key}')
    ax.legend(loc='upper right', frameon=True, shadow=True)
    #fig.tight_layout()
    fig.savefig(f'{experiment_dir}\\{subPath}\\butterflyPaper_{key}.png')
    plt.close(fig)

    """
    fig, ax = plt.subplots(1,1, figsize=(6, 3))

    mat = np.mean(EPOCHS.get_data(), axis=0).T
    for i in range(evoked_com.get_data().shape[0]):
        b=ax.plot(EPOCHS.times, EPOCHS.get_data()[i, :], c='b', alpha=0.5)
    ax.axvline(0, linewidth=10, c='k', alpha=0.5)
    ax.set_title(sub)
    
    fig.savefig(f'{experiment_dir}{sub}_butterfly_asPaper_{key}.png', dpi=300)
    plt.close(fig)
    """

def basicPlots(EPOCHS,json_data,experiment_dir,sub,
               key="epochs",
               subPath="1.basic",
               figsize=FIGSIZE,
               show=False,
               do_psdtopomap=False):

    from pathlib import Path
    import matplotlib.pyplot as plt

    out_dir=Path(experiment_dir)/Path(subPath)
    out_dir.mkdir(parents=True,exist_ok=True)

    fig=EPOCHS.average().plot(show=show,spatial_colors=True)
    fig.set_size_inches(figsize[0],figsize[1])
    fig.savefig(out_dir/f"tep_{key}.png")
    plt.close(fig)

    fig=EPOCHS.average().plot_topo(show=show)
    fig.set_size_inches(figsize[0],figsize[1])
    fig.savefig(out_dir/f"topo_{key}.png")
    plt.close(fig)

    fig=EPOCHS.plot_psd(
        method="welch",
        fmin=EPOCHS.info["highpass"],
        fmax=EPOCHS.info["lowpass"],
        xscale="log",
        show=show
    )
    fig.set_size_inches(figsize[0],figsize[1])
    fig.savefig(out_dir/f"PSD_{key}.png")
    plt.close(fig)

    if do_psdtopomap:
        fig=EPOCHS.plot_psd_topomap(
            method="welch",
            cmap="turbo",
            fmin=EPOCHS.info["highpass"],
            fmax=EPOCHS.info["lowpass"],
            show=show,
            normalize=True
        )
        fig.set_size_inches(figsize[0],figsize[1])
        fig.savefig(out_dir/f"PSD_topomap_{key}.png")
        plt.close(fig)

def runICA(detrendedEpochs):

    # find the maximum number of independent components
    # as the number of good channels - 1 because of average referencing
    
    n_components = len(detrendedEpochs.ch_names) - len(detrendedEpochs.info['bads']) - 1
    print(n_components)
    json_data['n_components'] = n_components
    # Salva parametri
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as txt_file:
        for key, value in sorted(json_data.items()):
            txt_file.write(f'{key}: {value}\n')
    
    # set ICA parameters
    ica=mne.preprocessing.ICA(n_components=n_components,
        max_iter="auto",
        method="infomax",
        random_state=220986,
        fit_params=dict(extended=True),
    )
    
    # perform ICA decomposition
    ica=ica.fit(detrendedEpochs)
    # tagging components
    ic_labels = label_components(detrendedEpochs, ica, method="iclabel")
    """
    brain
    muscle artifact
    eye blink
    heart beat
    line noise
    channel noise
    other
    """
    for idx, i  in enumerate(ic_labels['labels']):
        print(idx+1, i)
    
    return ic_labels, ica

import os
from pathlib import Path
import numpy as np
import pandas as pd
from fooof import FOOOF
from scipy.signal import welch
import matplotlib.pyplot as plt
import seaborn as sns
from mne.filter import filter_data

def extract_psd_features(epochs, note, experiment_dir, json_data):
    paths = rest_paths(experiment_dir)
    save_dir = paths["fooof"] / note
    save_dir.mkdir(parents=True, exist_ok=True)

    sfreq = epochs.info['sfreq']
    data = epochs.get_data()
    n_trials, n_channels, n_times = data.shape

    # Concatenazione trial → (n_channels, n_times * n_trials)
    data_concat = data.transpose(1, 0, 2).reshape(n_channels, n_times * n_trials)

    freqs, psds = welch(data_concat, fs=sfreq, average='mean', nperseg=1024)

    results = []

    # === Freq range usato per il fit
    freq_range = (json_data['l_freq'], json_data['h_freq']*2)

    for ch_idx, ch_name in enumerate(epochs.ch_names):
        fm = FOOOF(aperiodic_mode='fixed')
        fm.fit(freqs, psds[ch_idx])#, freq_range=freq_range)

        # Plot limitato al freq_range
        figp = fm.plot(freq_range=freq_range, 
                save_fig=True,
                file_name=f'chan_{ch_name}_{note}',
                file_path=save_dir)
        plt.close()

        intercept, slope = fm.aperiodic_params_
        n_resonances = len(fm.peak_params_)
        fit_error = fm.error_
        fit_r2 = fm.r_squared_

        freq_pos = freqs[freqs > 0]
        aperiodic_psd = intercept + np.log10(1 / freq_pos**slope)
        area_psd = np.trapz(aperiodic_psd, dx=freqs[1] - freqs[0])

        snr_index_broad = (area_psd / slope) / area_psd if slope != 0 and area_psd != 0 else np.nan
        snr_index_eq = 1 / slope if slope != 0 else np.nan

        freq_mask_band = (freq_pos >= json_data['l_freq']) & (freq_pos <= json_data['h_freq'])
        area_psd_narrowband = np.trapz(aperiodic_psd[freq_mask_band], dx=freqs[1] - freqs[0])
        snr_index_narrow = (area_psd_narrowband / slope) / area_psd if slope != 0 and area_psd != 0 else np.nan

        results.append({
            'channel': ch_name,
            'channel_index': ch_idx,
            'intercept': intercept,
            'slope': slope,
            'fiterror': fit_error,
            'r2': fit_r2,
            'n_resonances': n_resonances,
            'area_psd': area_psd,
            'area_psd_narrow': area_psd_narrowband,
            'snr_index_broad': snr_index_broad,
            'snr_index_eq': snr_index_eq,
            'snr_index_narrow': snr_index_narrow,
        })

    # === Salva risultati
    df = pd.DataFrame(results)
    df.to_csv(save_dir / f"{note}.csv", index=False)
    df.to_pickle(save_dir / f"{note}.pkl")

    # === Plot riepilogativo
    fig, axs = plt.subplots(2, 2, figsize=(16, 21))
    fig.suptitle(note)

    sns.barplot(y="channel", x="slope", data=df, ax=axs[0, 0], color="k")
    axs[0, 0].set_title("Aperiodic Slope per Channel")
    axs[0, 0].set_xlabel("Slope (log)")
    axs[0, 0].set_ylabel("Channel")
    axs[0, 0].set_xlim(0.01, 10)
    axs[0, 0].set_xscale("log")
    for x in [0.5, 1, 2, 3, 4, 5]:
        axs[0, 0].axvline(x=x, color='red', linestyle='-', linewidth=2)

    sns.barplot(y="channel", x="n_resonances", data=df, ax=axs[0, 1], color="k")
    axs[0, 1].set_title("Number of Resonances per Channel")
    axs[0, 1].set_xlabel("n Peaks (log)")
    axs[0, 1].set_ylabel("")
    axs[0, 1].set_xlim(0.01, 100)
    axs[0, 1].set_xscale("log")

    sns.barplot(y="channel", x="snr_index_narrow", data=df, ax=axs[1, 0], color="k")
    axs[1, 0].set_title("SNR Narrowband per Channel")
    axs[1, 0].set_xlabel("SNR")
    axs[1, 0].set_ylabel("Channel")
    axs[1, 0].set_xlim(0, 0.5)

    sns.barplot(y="channel", x="fiterror", data=df, ax=axs[1, 1], color="k")
    axs[1, 1].set_title("Fit Error per Channel")
    axs[1, 1].set_xlabel("Error")
    axs[1, 1].set_ylabel("")
    axs[1, 1].set_xlim(0, 0.5)

    plt.tight_layout()
    plt.savefig(save_dir / "summary_plots_hbar.png", dpi=150)
    plt.close()

    print(f"✅ FOOOF PSD features salvate e plottate in: {save_dir}")
    return df
    
def save_bad_epochs_and_channels(info_string, experiment_dir, sub, json_data):
    marker_epochs = "The following epochs were marked as bad and are dropped:"
    bad_epochs = ""
    
    if marker_epochs in info_string:
        print('i', info_string.find(marker_epochs))
        start = info_string.find(marker_epochs) + len(marker_epochs) + 1
        bad_epochs = info_string[start:].split("\n")[0].strip()
        print('f', info_string[start:].split("\n")[0].strip())
    
    json_data['bad_trials']=bad_epochs
    
    # Salva parametri
    with open(Path(experiment_dir) / f'{sub}_pars.json', 'w') as json_file:
            json.dump(json_data, json_file, indent=4, sort_keys=True)

def plotTrialTepVariability(epochs,json_data,experiment_dir,sub,
                            chanNAME="AF3",
                            operator=np.mean,
                            save=False,
                            figsize=FIGSIZE,
                            parDir="2.trials"):

    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np

    fig,ax=plt.subplots(1,1,figsize=figsize,sharey=False,sharex=True)

    i=np.where(np.array(epochs.ch_names)==chanNAME)[0][0]

    ax.plot(
        epochs.times,
        epochs.get_data()[:,i,:].T,
        c="b",
        linewidth=1,
        alpha=0.5
    )

    ax.plot(
        epochs.times,
        operator(epochs.get_data()[:,i,:],axis=0),
        c="r",
        label=str(operator),
        linewidth=5
    )

    ax.set_title(f"{epochs.ch_names[i]}")
    ax.legend()
    ax.set_xlabel("Times [s]")
    ax.set_ylabel("Amplitude [V]")
    plt.tight_layout()

    if save:
        saveNote=f"tepVar_{chanNAME}"

        if parDir==".":
            out_dir=Path(experiment_dir)/"2.trials"
        else:
            out_dir=Path(experiment_dir)/parDir

        out_dir.mkdir(parents=True,exist_ok=True)
        fig.savefig(out_dir/f"{sub}_{saveNote}.png")

    plt.close(fig)

def computeTimeMasks(epochs, chan, trial, json_data, do_plot=False, offset=0.20, plot_path=None, plot_title=None):
    import numpy as np
    t=epochs.times

    n=t.size
    data=epochs.get_data()
    sigAll=data[trial,chan,:] if data.ndim==3 else data[chan,:]

    #print("len(t) =", len(t))
    #print("data.shape =", data.shape)
    #print("len(sigAll) =", len(sigAll))
    
    def range_to_mask_by_t(t_start,t_end):
        i_start=int(np.searchsorted(t,t_start,side='left'))
        i_end=int(np.searchsorted(t,t_end,side='right'))-1
        i_start=max(0,min(i_start,n-1))
        i_end=max(0,min(i_end,n-1))
        if i_end<i_start:i_end=i_start
        m=np.zeros(n,dtype=bool)
        m[i_start:i_end+1]=True
        return m
    t_min_off=float(json_data['detrend_minTimeWindowOffset'])
    t_max_off=float(offset)
    maskPreOffset=range_to_mask_by_t(t.min(),t_min_off)
    maskTempOffset=range_to_mask_by_t(t_min_off,t_max_off)
    idx_temp=np.flatnonzero(maskTempOffset)
    if idx_temp.size==0:
        i0=int(np.searchsorted(t,t_min_off,side='left'))
        i0=max(0,min(i0,n-1))
        maskTempOffset=np.zeros(n,dtype=bool)
        maskTempOffset[i0]=True
        idx_temp=np.array([i0],dtype=int)
    sig=sigAll[idx_temp]
    extreme=json_data.get('detrendExtremeTechinque','max')
    if extreme=='derivative' and sig.size>=3:
        diff=np.diff(sig)
        zc=np.where((diff[:-1]<0)&(diff[1:]>0))[0]
        k=(zc[0]+1) if zc.size>0 else int(np.argmax(np.abs(sig)))
    else:
        k=int(np.argmax(np.abs(sig)))
    if k<0:k=0
    if k>=sig.size:k=sig.size-1
    i_peak=idx_temp[k]
    t_peak=t[i_peak]
    maskOffset=np.zeros(n,dtype=bool)
    maskOffset[np.flatnonzero(range_to_mask_by_t(t_min_off,t_peak))]=True
    maskPostOffset=np.zeros(n,dtype=bool)
    i_post_start=min(i_peak+1,n-1)
    maskPostOffset[i_post_start:]=True
    if do_plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(11,7))
        plt.plot(t,sigAll,label='signal (all)')
        plt.plot(t[idx_temp],sig,label='signal (temp window)')
        plt.scatter(t_peak,sigAll[i_peak],s=60,label=f'peak @ {t_peak:.4f}s')
        plt.axvline(x=t_min_off,linestyle='--',label=f"minTimeWindowOffset={t_min_off}")
        plt.axvline(x=t_max_off,linestyle='--',label=f"maxTimeWindowOffset={t_max_off}")
        plt.title(plot_title or f"chan={chan}, trial={trial}")
        plt.legend(loc='best')
        if plot_path:
            plt.savefig(plot_path,dpi=150,bbox_inches='tight');plt.close()
        else:
            plt.show()

    assert len(maskPreOffset) == len(t), f"maskPreOffset={len(maskPreOffset)} t={len(t)}"
    assert len(maskOffset) == len(t), f"maskOffset={len(maskOffset)} t={len(t)}"
    assert len(maskPostOffset) == len(t), f"maskPostOffset={len(maskPostOffset)} t={len(t)}"
    
    return maskPreOffset,maskOffset,maskPostOffset


def computeTimeMasks_old16102025(epochs, chan, trial, json_data, do_plot=False, offset=0.20, plot_path=None, plot_title=None):
    # dimensioni e sampling
    data = epochs.get_data()
    n = data.shape[-1]
    t = epochs.times
    sfreq = epochs.info['sfreq']

    # helper per convertire range (s) -> maschera booleana di lunghezza n
    def range_to_mask(t_start, t_end):
        s = int(round(t_start * sfreq))
        e = int(round(t_end   * sfreq))
        s = max(0, min(s, n-1))
        e = max(0, min(e, n-1))
        if e < s: e = s
        idx = np.arange(n)
        return (idx >= s) & (idx <= e)

    t_min_off = float(json_data['detrend_minTimeWindowOffset'])
    t_max_off = float(offset)  # qui usi l'argomento della funzione

    # maschere a indici (coerenti con n)
    maskPreOffset  = range_to_mask(t.min(), t_min_off)
    maskTempOffset = range_to_mask(t_min_off, t_max_off)

    # estrai il segnale corretto (2D o 3D)
    if data.ndim == 2:
        sigAll = data[chan, :]
    else:
        sigAll = data[trial, chan, :]

    sig = sigAll[maskTempOffset]
    if sig.size == 0:
        # fallback: se la finestra è vuota, prendi almeno 1 campione al limite
        # e rendi maskTempOffset con un singolo True in prossimità di t_min_off
        i0 = int(round(t_min_off * sfreq))
        i0 = max(0, min(i0, n-1))
        maskTempOffset = np.zeros(n, dtype=bool)
        maskTempOffset[i0] = True
        sig = sigAll[maskTempOffset]

    # stima del punto "estremo" nell'offset
    extreme = json_data.get('detrendExtremeTechinque', 'max')
    if extreme == 'derivative' and sig.size >= 3:
        diff = np.diff(sig)
        zc = np.where((diff[:-1] < 0) & (diff[1:] > 0))[0]
        tMaxOffsetIndex = (zc[0] + 1) if zc.size > 0 else int(np.argmax(np.abs(sig)))
    else:
        tMaxOffsetIndex = int(np.argmax(np.abs(sig)))

    # bound-check su sig (non su mask)
    if tMaxOffsetIndex <= 0:
        tMaxOffsetIndex = 1 if sig.size > 1 else 0
    if tMaxOffsetIndex >= sig.size:
        tMaxOffsetIndex = sig.size - 1

    # tempo del picco relativo alla finestra temp
    t_peak = t[maskTempOffset][tMaxOffsetIndex]

    # maschere finali: pre, offset (fino al picco), post
    maskOffset     = range_to_mask(t_min_off, t_peak)
    maskPostOffset = range_to_mask(t_peak, t.max())

    # opzionale: plot diagnostico, senza dipendenze esterne
    if do_plot:
        plt.figure(figsize=(11, 7))
        plt.plot(t, sigAll, label='signal (all)')
        plt.plot(t[maskTempOffset], sig, label='signal (temp window)')
        plt.scatter(t_peak, sig[tMaxOffsetIndex], s=60, label=f'peak @ {t_peak:.4f}s')
        plt.axvline(x=t_min_off, linestyle='--', label=f"minTimeWindowOffset={t_min_off}")
        plt.axvline(x=t_max_off, linestyle='--', label=f"maxTimeWindowOffset={t_max_off}")
        plt.title(plot_title or f"chan={chan}, trial={trial}")
        plt.legend(loc='best')
        if plot_path:
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    return maskPreOffset, maskOffset, maskPostOffset

def computeTimeMasks_old15102025(epochs, chan, trial, json_data, do_plot=False, offset=0.20):
    
    par = offset
    #print(json_data['pulse_artifact_rej_timewindow_max']*par)

    maskPreOffset = np.logical_and(epochs.times>=epochs.times.min(), 
                                   epochs.times<json_data['detrend_minTimeWindowOffset'])
    
    tempMaskOffset = np.logical_and(epochs.times>=json_data['detrend_minTimeWindowOffset'], 
                                    epochs.times<=par)

    
    if len(epochs.get_data().shape)==2: 
        sig = epochs.get_data()[chan, :][tempMaskOffset]
        sigAll = epochs.get_data()[chan, :]
        
    if len(epochs.get_data().shape)>2: 
        sig = epochs.get_data()[trial, chan, :][tempMaskOffset]
        sigAll = epochs.get_data()[trial, chan, :]

    extreme=json_data['detrendExtremeTechinque']
    if extreme=='max':
        tMaxOffsetIndex=np.argmax(abs(sig))
    if extreme=='derivative':
        diff = np.diff(sig)
        # Trova transizione da negativo a positivo (zero-crossing della derivata)
        zero_crossings = np.where((diff[:-1] < 0) & (diff[1:] > 0))[0]
        if len(zero_crossings) > 0:
            tMaxOffsetIndex = zero_crossings[0] + 1  # +1 perché diff è più corto di 1
        else:
            # fallback: se non trovi un punto di inversione, usa massimo assoluto
            tMaxOffsetIndex = np.argmax(np.abs(sig))

    if tMaxOffsetIndex==0: tMaxOffsetIndex=1
    if tMaxOffsetIndex==len(tempMaskOffset): tMaxOffsetIndex=len(tempMaskOffset)-1
   
    if do_plot:
        plt.figure(figsize=(13, 11))
        plt.plot(epochs.times[tempMaskOffset], sig)
        plt.plot(epochs.times, sigAll)
        plt.title(f'{sub}-{epochs.ch_names[chan]}-trial{trial}')
        plt.scatter(epochs.times[tempMaskOffset][tMaxOffsetIndex], sig[tMaxOffsetIndex], c='r', 
                    label=f'peak point at {epochs.times[tempMaskOffset][tMaxOffsetIndex]}')
        plt.axvline(x=offset, label=f'maxTimeWindowOffset={json_data['detrend_maxTimeWindowOffset']}')
        plt.legend(loc='upper right')
        plt.savefig(f'{experiment_dir}/2.detrend/test_maskTest_{sub}_{chan}_{trial}.png')
        plt.close()
        
    maskOffset = np.logical_and(epochs.times>=json_data['detrend_minTimeWindowOffset'], 
                                epochs.times<=epochs.times[tempMaskOffset][tMaxOffsetIndex])   
    
    maskPostOffset = np.logical_and(epochs.times>epochs.times[tempMaskOffset][tMaxOffsetIndex], 
                                    epochs.times<=epochs.times.max())

    return maskPreOffset, maskOffset, maskPostOffset

def computeTimeMasks_old(epochs, chan, trial, do_plot=False, offset=0.20):
    
    par = offset
    #print(json_data['pulse_artifact_rej_timewindow_max']*par)

    maskPreOffset = np.logical_and(epochs.times>=epochs.times.min(), 
                                   epochs.times<json_data['detrend_minTimeWindowOffset'])
    
    tempMaskOffset = np.logical_and(epochs.times>=json_data['detrend_minTimeWindowOffset'], 
                                    epochs.times<=par)

    
    if len(epochs.get_data().shape)==2: 
        sig = epochs.get_data()[chan, :][tempMaskOffset]
        sigAll = epochs.get_data()[chan, :]
        
    if len(epochs.get_data().shape)>2: 
        sig = epochs.get_data()[trial, chan, :][tempMaskOffset]
        sigAll = epochs.get_data()[trial, chan, :]
        

    tMaxOffsetIndex=np.argmax(abs(sig))
    
    if tMaxOffsetIndex==0: tMaxOffsetIndex=1
    if tMaxOffsetIndex==len(tempMaskOffset): tMaxOffsetIndex=len(tempMaskOffset)-1

    #print(tMaxOffsetIndex)
    
    if do_plot:
        plt.figure(figsize=(13, 11))
        plt.plot(epochs.times[tempMaskOffset], sig)
        plt.plot(epochs.times, sigAll)
        plt.title(f'{sub}-{epochs.ch_names[chan]}-trial{trial}')
        plt.scatter(epochs.times[tempMaskOffset][tMaxOffsetIndex], sig[tMaxOffsetIndex], c='r', 
                    label=f'peak point at {epochs.times[tempMaskOffset][tMaxOffsetIndex]}')
        plt.axvline(x=offset, label=f'maxTimeWindowOffset={json_data['detrend_maxTimeWindowOffset']}')
        plt.legend(loc='upper right')
        plt.savefig(f'{experiment_dir}/2.detrend/test/maskTest_{sub}_{chan}_{trial}.png')
        plt.close()
        
    maskOffset = np.logical_and(epochs.times>=json_data['detrend_minTimeWindowOffset'], 
                                epochs.times<=epochs.times[tempMaskOffset][tMaxOffsetIndex])   
    
    maskPostOffset = np.logical_and(epochs.times>epochs.times[tempMaskOffset][tMaxOffsetIndex], 
                                    epochs.times<=epochs.times.max())

    return maskPreOffset, maskOffset, maskPostOffset

def computeSlopes_v4(epochs, json_data, experiment_dir, sub, saveNote=f'plotTrialTepVariability'):
    """
    Compute slopes of linear regressions for EEG data across time windows and trials.
    
    Parameters:
        epochs: MNE Epochs object
            EEG data segmented into epochs.
        normalized_distances: array
            Array of normalized distances for each channel.
        channel_names: list
            List of channel names corresponding to the distances.
        saveNote: str
            A string used for saving plots or notes.

    Returns:
        df_slopes: DataFrame
            Dataframe containing slopes, intercepts, and distance information.
    """
    # compute distance
    evoked = epochs
    positions = np.array([ch['loc'][:3] for ch in evoked.info['chs']])  # Estrai solo le coordinate (x, y, z)
    seed_channels = json_data['seedChans']
    seed_indices = [evoked.info['ch_names'].index(ch_name) for ch_name in seed_channels]
    seed_positions = positions[seed_indices]
    mean_seed_position = np.mean(seed_positions, axis=0)
    distances_from_seed = np.zeros(len(positions))
    for i in range(len(positions)):
        distances_from_seed[i] = euclidean(mean_seed_position, positions[i])
    min_distance = np.min(distances_from_seed)
    normalized_distances = (distances_from_seed - min_distance) / (np.max(distances_from_seed) - min_distance)
    # Create a mapping of channel names to distances
    channel_distance_mapping = dict(zip(evoked.info['ch_names'], normalized_distances))
    
    timeMaskLabels = ['preOffset', 'offset', 'postOffset']
    slopes = []

    for chan in tqdm(epochs.ch_names):
        id_chan = np.where(np.array(epochs.ch_names) == chan)[0][0]

        if len(epochs.get_data().shape) > 2:
            for id_trial in range(epochs.get_data().shape[0]):
                timeMask = computeTimeMasks(epochs, id_chan, id_trial, json_data, offset=json_data['detrend_maxTimeWindowOffset'])
                for t_label, id_t, t_mask in zip(timeMaskLabels, [0, 1, 2], timeMask):
                    y = epochs.get_data()[id_trial, id_chan, t_mask]
                    slope, intercept, _, _, _ = linregress(epochs.times[t_mask], y)
                    slopes.append([t_label, id_trial, chan, intercept, slope])

        elif len(epochs.get_data().shape) == 2:
            id_trial = 0
            timeMask = computeTimeMasks(epochs, id_chan, id_trial, json_data, offset=json_data['detrend_maxTimeWindowOffset'])
            for t_label, id_t, t_mask in zip(timeMaskLabels, [0, 1, 2], timeMask):
                y = epochs.get_data()[id_chan, t_mask]
                slope, intercept, _, _, _ = linregress(epochs.times[t_mask], y)
                slopes.append([t_label, id_trial, chan, intercept, slope])

    # Convert slopes list to DataFrame
    df_slopes = pd.DataFrame(data=slopes, 
                             columns=['id_twindow', 'id_trial', 'chan', 'intercept', 'slope'])

    # Add the 'distanceFromSeed' column using the channel distance mapping
    df_slopes['distanceFromSeed'] = df_slopes['chan'].map(channel_distance_mapping)

    # Compute Z-scores for slopes within each time window
    df_slopes['Zslope'] = np.nan
    for time_label in timeMaskLabels:
        mask = df_slopes['id_twindow'] == time_label
        vals = df_slopes.loc[mask, 'slope'].values.astype(float)
        std = np.nanstd(vals)
        mean = np.nanmean(vals)
        if np.isnan(std) or std == 0:
            df_slopes.loc[mask, 'Zslope'] = 0.0
        else:
            df_slopes.loc[mask, 'Zslope'] = (vals - mean) / std

    # add plot distance from seed vs slope

    return df_slopes

    
def computeSlopes_v4_old(epochs, json_data, experiment_dir, sub, saveNote=f'plotTrialTepVariability'):
    """
    Compute slopes of linear regressions for EEG data across time windows and trials.
    
    Parameters:
        epochs: MNE Epochs object
            EEG data segmented into epochs.
        normalized_distances: array
            Array of normalized distances for each channel.
        channel_names: list
            List of channel names corresponding to the distances.
        saveNote: str
            A string used for saving plots or notes.

    Returns:
        df_slopes: DataFrame
            Dataframe containing slopes, intercepts, and distance information.
    """
    # compute distance
    evoked = epochs
    positions = np.array([ch['loc'][:3] for ch in evoked.info['chs']])  # Estrai solo le coordinate (x, y, z)
    seed_channels = json_data['seedChans']
    seed_indices = [evoked.info['ch_names'].index(ch_name) for ch_name in seed_channels]
    seed_positions = positions[seed_indices]
    mean_seed_position = np.mean(seed_positions, axis=0)
    distances_from_seed = np.zeros(len(positions))
    for i in range(len(positions)):
        distances_from_seed[i] = euclidean(mean_seed_position, positions[i])
    min_distance = np.min(distances_from_seed)
    normalized_distances = (distances_from_seed - min_distance) / (np.max(distances_from_seed) - min_distance)
    # Create a mapping of channel names to distances
    channel_distance_mapping = dict(zip(evoked.info['ch_names'], normalized_distances))
    
    timeMaskLabels = ['preOffset', 'offset', 'postOffset']
    slopes = []

    for chan in tqdm(epochs.ch_names):
        id_chan = np.where(np.array(epochs.ch_names) == chan)[0][0]

        if len(epochs.get_data().shape) > 2:
            for id_trial in range(epochs.get_data().shape[0]):
                timeMask = computeTimeMasks(epochs, id_chan, id_trial, json_data, offset=json_data['detrend_maxTimeWindowOffset'])
                for t_label, id_t, t_mask in zip(timeMaskLabels, [0, 1, 2], timeMask):
                    y = epochs.get_data()[id_trial, id_chan, t_mask]
                    slope, intercept, _, _, _ = linregress(epochs.times[t_mask], y)
                    slopes.append([t_label, id_trial, chan, intercept, slope])

        elif len(epochs.get_data().shape) == 2:
            id_trial = 0
            timeMask = computeTimeMasks(epochs, id_chan, id_trial, json_data, offset=json_data['detrend_maxTimeWindowOffset'])
            for t_label, id_t, t_mask in zip(timeMaskLabels, [0, 1, 2], timeMask):
                y = epochs.get_data()[id_chan, t_mask]
                slope, intercept, _, _, _ = linregress(epochs.times[t_mask], y)
                slopes.append([t_label, id_trial, chan, intercept, slope])

    # Convert slopes list to DataFrame
    df_slopes = pd.DataFrame(data=slopes, 
                             columns=['id_twindow', 'id_trial', 'chan', 'intercept', 'slope'])

    # Add the 'distanceFromSeed' column using the channel distance mapping
    df_slopes['distanceFromSeed'] = df_slopes['chan'].map(channel_distance_mapping)

    # Compute Z-scores for slopes within each time window
    df_slopes['Zslope'] = np.zeros(df_slopes['slope'].shape[0])
    for time_label in timeMaskLabels:
        mask = df_slopes['id_twindow'] == time_label
        df_slopes.loc[mask, 'Zslope'] = scipy.stats.zscore(df_slopes.loc[mask, 'slope'].values)

    # add plot distance from seed vs slope

    return df_slopes

import os
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def computeSlopesPlot_v3(df_slopes, sub,
                         saveNote='ALL-TRIALS', sharex=True, subPath='2.detrend',
                      zvalue=True, json_data=None, experiment_dir='.'):
    
    VAR = 'Zslope' if zvalue else 'slope'
    ntrial = df_slopes['id_trial'].nunique()
    timeMaskLabels = ['preOffset', 'offset', 'postOffset']
    
    # Dizionario e flag
    outlier_channels_by_twindow = {}
    found_outlier = False
    df_outliers_list = []

    # ANOVA per ciascuna finestra
    for time_window in timeMaskLabels:
        df_time_window = df_slopes[df_slopes["id_twindow"] == time_window]
        channel_groups = [df_time_window[df_time_window["chan"] == chan][VAR] for chan in df_time_window["chan"].unique()]
        anova_stat, anova_p = scipy.stats.f_oneway(*channel_groups)

    # Plot swarm/pointplot
    fig, ax = plt.subplots(1, 3, figsize=(15, 19), sharex=sharex, sharey=True)
    fig.suptitle(f'{saveNote} - N° Trials: {ntrial}')

    for idx, label in enumerate(timeMaskLabels):
        data_subset = df_slopes[df_slopes['id_twindow'] == label]
        mean_val = data_subset[VAR].mean()
        std_val = data_subset[VAR].std()
        has_outliers = any(abs(data_subset[VAR] - mean_val) > 3 * std_val)
        title_color = "red" if has_outliers else "black"

        ax[idx].axvline(0, linewidth=10, alpha=0.25, c='g')
        ax[idx].set_xlim(-7, 7) if zvalue else ax[idx].set_xlim(data_subset[VAR].min(), data_subset[VAR].max())

        if json_data is not None:
            thr = json_data.get('detrend_slopeThr', 3)
            ax[idx].axvline(x=-thr, alpha=0.25, linewidth=10, c='r')
            ax[idx].axvline(x=thr, alpha=0.25, linewidth=10, c='r')

        sns.swarmplot(data=data_subset, x=VAR, y='chan', ax=ax[idx], color='black', alpha=0.1)
        sns.pointplot(data=data_subset, estimator=np.mean, x=VAR, y='chan', ax=ax[idx], color='r', alpha=1)
        ax[idx].set_title(f"{label}", color=title_color)
        ax[idx].set_xlabel(VAR)
        ax[idx].set_ylabel('Channels')

        # --- Calcolo media e outlier per canale ---
        mean_per_chan = data_subset.groupby("chan")[VAR].mean()
        global_mean = mean_per_chan.mean()
        global_std = mean_per_chan.std()
        z_scores = (mean_per_chan - global_mean) / global_std
        outlier_mask = abs(z_scores) > 3

        df_summary = pd.DataFrame({
            'id_twindow': label,
            'chan': mean_per_chan.index,
            'mean_' + VAR: mean_per_chan.values,
            'Zscore': z_scores.values,
        })
        df_summary['is_outlier'] = abs(df_summary['Zscore']) > 3
        
        df_outliers_list.append(df_summary)
        
        if df_summary['is_outlier'].any():
            outlier_channels_by_twindow[label] = df_summary[df_summary['is_outlier']]['chan'].tolist()
            found_outlier = True


    # Salva plot swarm/pointplot
    out_path = os.path.join(experiment_dir, subPath, f'{VAR}_{saveNote}.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    # KDE plot per canali seed
    df_slopes_seed = df_slopes[df_slopes["chan"].isin(json_data['seedChans'])] if json_data else df_slopes
    timeMaskLabels = df_slopes['id_twindow'].unique()
    groups = [df_slopes_seed[df_slopes_seed['id_twindow'] == label][VAR] for label in timeMaskLabels]
    anova_stat, p_value = scipy.stats.f_oneway(*groups)

    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df_slopes_seed, x=VAR, hue='id_twindow', cumulative=False)
    plt.xlim(-7, 7) if zvalue else plt.xlim(df_slopes_seed[VAR].min(), df_slopes_seed[VAR].max())
    plt.ylim(0, 0.75)

    seed_list = json_data["seedChans"] if json_data and "seedChans" in json_data else "ALL"
    plt.title(f'{saveNote} - N° Trials: {ntrial} \n ANOVA: F={anova_stat:.3f}, p={p_value:.3f} \n seedChans={seed_list}')
    histo_path = os.path.join(experiment_dir, subPath, f'histo_{VAR}_{saveNote}.png')
    plt.savefig(histo_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Unisci risultati per finestra in unico DataFrame
    df_outliers_all = pd.concat(df_outliers_list, ignore_index=True)

    return outlier_channels_by_twindow, found_outlier, df_outliers_all




import os
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def computeSlopesPlot_v2(df_slopes,
                         sub,
                         saveNote='ALL-TRIALS', sharex=True, subPath='2.detrend', zvalue=True, json_data=None, experiment_dir='.'):
    VAR = 'Zslope' if zvalue else 'slope'
    ntrial = df_slopes['id_trial'].nunique()
    timeMaskLabels = ['preOffset', 'offset', 'postOffset']
    outlier_channels = {}

    # Calcolo ANOVA per ogni time window
    anova_results = []
    for time_window in timeMaskLabels:
        df_time_window = df_slopes[df_slopes["id_twindow"] == time_window]
        channel_groups = [df_time_window[df_time_window["chan"] == chan][VAR] for chan in df_time_window["chan"].unique()]
        anova_stat, anova_p = scipy.stats.f_oneway(*channel_groups)
        anova_results.append({"Time Window": time_window, "Statistic": anova_stat, "p-value": anova_p})

    df_anova_results = pd.DataFrame(anova_results)

    # Plot a 3 pannelli
    fig, ax = plt.subplots(1, 3, figsize=(15, 19), sharex=sharex, sharey=True)
    fig.suptitle(f'{saveNote} - N° Trials: {ntrial}')

    for idx, label in enumerate(timeMaskLabels):
        data_subset = df_slopes[df_slopes['id_twindow'] == label]
        mean_val = data_subset[VAR].mean()
        std_val = data_subset[VAR].std()
        has_outliers = any(abs(data_subset[VAR] - mean_val) > 3 * std_val)
        title_color = "red" if has_outliers else "black"

        ax[idx].axvline(0, linewidth=10, alpha=0.25, c='g')
        ax[idx].set_xlim(-7, 7) if zvalue else ax[idx].set_xlim(data_subset[VAR].min(), data_subset[VAR].max())

        if json_data is not None:
            thr = json_data.get('detrend_slopeThr', 3)
            ax[idx].axvline(x=-thr, alpha=0.25, linewidth=10, c='r')
            ax[idx].axvline(x=thr, alpha=0.25, linewidth=10, c='r')

        sns.swarmplot(data=data_subset, x=VAR, y='chan', ax=ax[idx], color='black', alpha=0.1)
        sns.pointplot(data=data_subset, estimator=np.mean, x=VAR, y='chan', ax=ax[idx], color='r', alpha=1)

        ax[idx].set_title(f"{label}", color=title_color)
        ax[idx].set_xlabel(VAR)
        ax[idx].set_ylabel('Channels')

        # --- Calcolo outlier medi per canale in questa finestra ---
        mean_per_chan = data_subset.groupby("chan")[VAR].mean()
        global_mean = mean_per_chan.mean()
        global_std = mean_per_chan.std()
        z_scores = (mean_per_chan - global_mean) / global_std
        outliers = z_scores[abs(z_scores) > 3]

        if not outliers.empty:
            outlier_channels[label] = [(chan, round(z_scores[chan], 2)) for chan in outliers.index]

    # Salvataggio del plot swarm+pointplot
    out_path = os.path.join(experiment_dir, subPath, f'{VAR}_{saveNote}.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    # KDE plot sui canali seed (o tutti se seedChans mancante)
    df_slopes_seed = df_slopes[df_slopes["chan"].isin(json_data['seedChans'])] if json_data else df_slopes
    timeMaskLabels = df_slopes['id_twindow'].unique()
    groups = [df_slopes_seed[df_slopes_seed['id_twindow'] == label][VAR] for label in timeMaskLabels]
    anova_stat, p_value = scipy.stats.f_oneway(*groups)

    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df_slopes_seed, x=VAR, hue='id_twindow', cumulative=False)
    plt.xlim(-7, 7) if zvalue else plt.xlim(df_slopes_seed[VAR].min(), df_slopes_seed[VAR].max())
    plt.ylim(0, 0.75)

    p_text = "p<0.05" if p_value < 0.05 else "p=ns"
    seed_list = json_data["seedChans"] if json_data and "seedChans" in json_data else "ALL"
    plt.title(f'{saveNote} - N° Trials: {ntrial} \n ANOVA: F={anova_stat:.3f}, p={p_value:.3f} \n seedChans={seed_list}')

    histo_path = os.path.join(experiment_dir, subPath, f'histo_{VAR}_{saveNote}.png')
    plt.savefig(histo_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {'p_value': p_value, 'F': anova_stat, 'outlier_channels': outlier_channels}


def computeSlopesPlot(df_slopes, 
                      json_data, experiment_dir, sub,
                      saveNote='ALL-TRIALS', sharex=True, subPath='2.detrend', zvalue=True):
    
    VAR = 'Zslope' if zvalue else 'slope'
    ntrial = df_slopes['id_trial'].nunique()
    timeMaskLabels = ['preOffset', 'offset', 'postOffset']

    # Calcolo ANOVA per ogni time window
    anova_results = []
    for time_window in timeMaskLabels:
        df_time_window = df_slopes[df_slopes["id_twindow"] == time_window]
        channel_groups = [df_time_window[df_time_window["chan"] == chan][VAR] for chan in df_time_window["chan"].unique()]
        anova_stat, anova_p = scipy.stats.f_oneway(*channel_groups)
        anova_results.append({"Time Window": time_window, "Statistic": anova_stat, "p-value": anova_p})

    # Convertire in DataFrame
    df_anova_results = pd.DataFrame(anova_results)

    # Plot dei pointplot per ogni timeMaskLabel con ANOVA p-value nei titoli
    fig, ax = plt.subplots(1, 3, figsize=(15, 19), sharex=sharex, sharey=True)
    fig.suptitle(f'{saveNote} - N° Trials: {ntrial}')

    for idx, label in enumerate(timeMaskLabels):
        data_subset = df_slopes[df_slopes['id_twindow'] == label]
    
        # Calcola media e deviazione standard
        mean_val = data_subset[VAR].mean()
        std_val = data_subset[VAR].std()
        
        # Determina se ci sono valori oltre 3 sigma
        #has_outliers = any(abs(data_subset[VAR] - mean_val) > threshold * std_val)
        title_color = "black" #red" if has_outliers else "black"
        ax[idx].axvline(0, linewidth=10, alpha=0.25, c='g')
        ax[idx].set_xlim(-3.5*2, 3.5*2)
        ax[idx].axvline(x=-json_data['detrend_slopeThr'], alpha=0.25, linewidth=10, c='r')
        ax[idx].axvline(x=json_data['detrend_slopeThr'], alpha=0.25, linewidth=10, c='r')
        
        # Crea il pointplot
        sns.swarmplot(
            data=data_subset,
            x=VAR, y='chan', ax=ax[idx], color='black', alpha=0.1, #errorbar='se'
        )

        # Crea il pointplot
        sns.pointplot(
            data=data_subset,
            estimator=np.mean,
            x=VAR, y='chan', ax=ax[idx], color='r', alpha=1, #errorbar='se'
        )

        ax[idx].set_title(f"{label}", color=title_color)
        ax[idx].set_xlabel(VAR)
        ax[idx].set_ylabel('Channels')
        out_path = os.path.join(experiment_dir, subPath, f'{VAR}_{saveNote}_{label}.csv')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        data_subset.to_csv(out_path, index=False)


    # Salvataggio del primo plot
    fig.savefig(f'{experiment_dir}\\{subPath}\\{VAR}_{saveNote}.png')
    plt.close(fig)

    # Seleziona la variabile di interesse
    VAR = 'Zslope' if zvalue else 'slope'
    timeMaskLabels = df_slopes['id_twindow'].unique()
    plt.figure(figsize=(8, 5))
    df_slopes_seed = df_slopes[df_slopes["chan"].isin(json_data['seedChans'])]
    groups = [df_slopes_seed[df_slopes_seed['id_twindow'] == label][VAR] for label in timeMaskLabels]
    anova_stat, p_value = scipy.stats.f_oneway(*groups)
    #sns.kdeplot(data=df_slopes_seed, x='Zslope', hue='id_twindow', cumulative=False)
    sns.kdeplot(data=df_slopes_seed, x=VAR, hue='id_twindow', cumulative=False)
    plt.xlim(-7, 7) if zvalue else plt.xlim(df_slopes_seed[VAR].min(), df_slopes_seed[VAR].max())
    plt.ylim(0, 0.75)
    #plt.title(f"ANOVA: F={anova_stat:.3f}, p={p_value:.3e} \n {json_data['seedChans']}") #\n {multimodal_text}")
    p_text = "p<0.05" if p_value < 0.05 else "p=ns"
    plt.title(f'{saveNote} - N° Trials: {ntrial} \n ANOVA: F={anova_stat:.3f}, p={p_value:.3f} \n seedChans={json_data['seedChans']}')
    plt.savefig(f'{experiment_dir}\\{subPath}\\histo_{VAR}_{saveNote}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print('p_value', p_value, 'F', anova_stat)

    return df_anova_results

def generate_noise_from_distribution(time_series, model='Gaussian', n_samples=1000):
    x=np.asarray(time_series).ravel()
    x=x[~np.isnan(x)]
    if x.size==0: return np.zeros(n_samples,dtype=float)
    mean=float(np.mean(x)); std=float(np.std(x)*0.5); med=float(np.median(x))
    mn=float(np.min(x)); mx=float(np.max(x))
    eps=1e-8
    std=max(std,eps)
    if not np.isfinite(mn) or not np.isfinite(mx): mn=mean-std; mx=mean+std
    if mn==mx: mn-=std; mx+=std
    if mn>mx: mn,mx=mx,mn
    m=model.lower()
    if m=='gaussian': return np.random.normal(mean,std,n_samples)
    if m=='exponential': return np.random.exponential(scale=max(std,eps),size=n_samples)
    if m=='laplace': return np.random.laplace(loc=med,scale=std,size=n_samples)
    if m=='poisson': return np.random.poisson(lam=max(mean,0.0),size=n_samples)
    if m=='rayleigh': return np.random.rayleigh(scale=std,size=n_samples)
    if m=='gamma':
        shape=(mean**2)/(std**2) if std>eps else 1.0
        shape=max(shape,eps)
        scale=(std**2)/max(mean,eps)
        return np.random.gamma(shape,scale,size=n_samples)
    if m in ('studentt','student_t','t'): return np.random.standard_t(2.0,size=n_samples)*std+mean
    if m=='uniform': return np.random.uniform(low=mn,high=mx,size=n_samples)
    raise ValueError("Unsupported model")


def generate_noise_from_distribution_old151012025(time_series, model='Gaussian', n_samples=1000):
    """
    Generate noise samples from a specified distribution based on a time series.

    Parameters:
    - time_series: The input time series (numpy array or pandas Series).
    - model: The distribution to sample from ('Gaussian', 'Exponential', 'Laplace', 
             'Poisson', 'Rayleigh', 'Gamma', 'StudentT', 'Uniform').
    - n_samples: The number of noise samples to generate.

    Returns:
    - noise_samples: The generated noise samples.
    """
    
    # Compute parameters based on the time series
    if isinstance(time_series, pd.Series):
        time_series = time_series.values  # Convert pandas Series to numpy array for easier handling
    
    # Common parameters
    mean = np.mean(time_series)
    std = np.std(time_series)*0.5
    min_val = np.min(time_series)
    max_val = np.max(time_series)
    median = np.median(time_series)
    
    # Now, depending on the chosen model, we compute the parameters and generate samples
    if model == 'Gaussian':
        # Gaussian distribution (Normal distribution)
        noise_samples = np.random.normal(loc=mean, scale=std, size=n_samples)
    
    elif model == 'Exponential':
        # Exponential distribution (mean = 1/lambda, use std for scale)
        lambda_param = 1 / std  # Use standard deviation as a proxy for rate
        noise_samples = np.random.exponential(scale=1/lambda_param, size=n_samples)
    
    elif model == 'Laplace':
        # Laplace distribution (mean = median, scale = std)
        noise_samples = np.random.laplace(loc=median, scale=std, size=n_samples)
    
    elif model == 'Poisson':
        # Poisson distribution (lambda = mean of the time series)
        lambda_param = mean  # Average rate of occurrence
        noise_samples = np.random.poisson(lam=lambda_param, size=n_samples)
    
    elif model == 'Rayleigh':
        # Rayleigh distribution (scale = std)
        noise_samples = np.random.rayleigh(scale=std, size=n_samples)
    
    elif model == 'Gamma':
        # Gamma distribution (shape = (mean^2)/(std^2), scale = (std^2)/mean)
        shape = (mean**2) / (std**2)
        scale = (std**2) / mean
        noise_samples = np.random.gamma(shape, scale, size=n_samples)
    
    elif model == 'StudentT':
        # Student's t-distribution (df = 2 for heavy tails, scale = std)
        df = 2  # degrees of freedom (use a small df for heavy tails)
        noise_samples = np.random.standard_t(df, size=n_samples) * std + mean
    
    elif model == 'Uniform':
        # Uniform distribution (min = min_val, max = max_val)
        noise_samples = np.random.uniform(low=min_val, high=max_val, size=n_samples)
    
    else:
        raise ValueError(f"Model '{model}' is not supported. Choose from ['Gaussian', 'Exponential', 'Laplace', 'Poisson', 'Rayleigh', 'Gamma', 'StudentT', 'Uniform']")

    return noise_samples


# no Exponential, Poisson, Rayleigh, 'Gamma', 'StudentT', 
supported_models = ['Exponential', 'Gaussian', 'Laplace', 'Uniform']

def computeDetrend_v6(EPOCHS, 
                        json_data, experiment_dir, sub,
                        typeOffsetRise,
                        typeOffsetDecay,
                        fitConstraint=True,
                        correctMode='resample', 
                        oddSamples=5, 
                        offsetChans=['Cz', 'Fz'], 
                        lag_correction=True,
                        detrendMode='single',
                        doDetrendOnlyOffsetChans=True,
                     ):
    # CHECK CONSTRAINT START OFFSET 0 or MIN OFFSET TIMEMASK
    # CHECK OTPIMIZATION ACROSS METHODS POLY
    
    from scipy.optimize import curve_fit
    import numpy as np
    
    # POLY
    def poly_func(x, *coeffs):
        return sum(c * x**i for i, c in enumerate(coeffs))
    def fit_polynomial_curvefit(x, y, max_order=5):
        best_order = 1
        best_mse = np.inf
        best_coeffs = None
        for order in range(1, max_order + 1):
            p0 = [0.0] * (order + 1)
            try:
                coeffs, _ = curve_fit(lambda x, *c: poly_func(x, *c), x, y, p0=p0)
                y_fit = poly_func(x, *coeffs)
                mse = np.mean((y - y_fit)**2)
                if mse < best_mse:
                    best_mse = mse
                    best_order = order
                    best_coeffs = coeffs
            except Exception as e:
                continue
        return best_order, best_coeffs

    # POWER LAW
    def power_law_decay(x, a, b, c):
        return a * np.power(x, -b) + c
        
    def power_law_decay_constrained(x, b, c, x0, y0):
        a = (y0 - c) * np.power(x0, b)
        return a * np.power(x, -b) + c

    # === MODELLI BASE con parametri diretti (tau_rise, tau_decay) ===
    def alpha_func(x, A, tau, c):
        return A * x * np.exp(-x / tau) + c
    def exp_func_single_rise(x, a, tau, c):
        return a * (1 - np.exp(-x / tau)) + c
    def exp_func_single_decay(x, a, tau, c):
        return a * np.exp(-x / tau) + c
    def exp_func_double_decay(x, a1, tau1, a2, tau2, c):
        return a1 * np.exp(-x / tau1) + a2 * np.exp(-x / tau2) + c
    def exp_func_biexp(x, A, tau_rise, tau_decay, c):
        return A * (1 - np.exp(-x / tau_rise)) * np.exp(-x / tau_decay) + c
    def exp_func_biexp_double_decay(x, A, tau_rise, tau_decay1, tau_decay2, B, c):
        rise = 1 - np.exp(-x / tau_rise)
        decay = B * np.exp(-x / tau_decay1) + (1 - B) * np.exp(-x / tau_decay2)
        return A * rise * decay + c
    def exp_func_triexp_decay(x, A, tau_rise, tau_decay1, tau_decay2, tau_decay3, B1, B2, c):
        rise = 1 - np.exp(-x / tau_rise)
        B3 = 1.0 - B1 - B2
        decay = B1 * np.exp(-x / tau_decay1) + B2 * np.exp(-x / tau_decay2) + B3 * np.exp(-x / tau_decay3)
        return A * rise * decay + c
    def exp_func_biexp_double_rise_decay(x, A, tau_rise1, tau_rise2, B_r, tau_decay1, tau_decay2, B_d, c):
        rise = B_r * (1 - np.exp(-x / tau_rise1)) + (1 - B_r) * (1 - np.exp(-x / tau_rise2))
        decay = B_d * np.exp(-x / tau_decay1) + (1 - B_d) * np.exp(-x / tau_decay2)
        return A * rise * decay + c
    
    # === MODELLI VINCOLATI ===
    # constrained to first element
    def alpha_func_constrained(x, tau, x0, y0):
        A = (y0 / x0) * np.exp(x0 / tau)
        return A * x * np.exp(-x / tau)
    def exp_func_single_rise_constrained(x, tau, c, x0, y0):
        a = (y0 - c) / (1 - np.exp(-x0 / tau))
        return a * (1 - np.exp(-x / tau)) + c
    def exp_func_single_decay_constrained(x, tau, c, x0, y0):
        a = (y0 - c) / np.exp(-x0 / tau)
        return a * np.exp(-x / tau) + c
    def exp_func_double_decay_constrained(x, tau1, tau2, a2, c, x0, y0):
        a1 = y0 - (a2 * np.exp(-x0 / tau2) + c)
        return a1 * np.exp(-x / tau1) + a2 * np.exp(-x / tau2) + c
  
    def exp_func_biexp_constrained(x, A, tau_rise, tau_decay, x0, y0):
        f0 = A * (1 - np.exp(-x0 / tau_rise)) * np.exp(-x0 / tau_decay)
        correction = y0 - f0
        return A * (1 - np.exp(-x / tau_rise)) * np.exp(-x / tau_decay) + correction
    
    def exp_func_biexp_double_decay_constrained(x, A, tau_rise, tau_decay1, tau_decay2, B, x0, y0):
        rise0 = 1 - np.exp(-x0 / tau_rise)
        decay0 = B * np.exp(-x0 / tau_decay1) + (1 - B) * np.exp(-x0 / tau_decay2)
        f0 = A * rise0 * decay0
        rise = 1 - np.exp(-x / tau_rise)
        decay = B * np.exp(-x / tau_decay1) + (1 - B) * np.exp(-x / tau_decay2)
        return A * rise * decay + (y0 - f0)
    
    def exp_func_triexp_decay_constrained(x, A, tau_rise, tau_decay1, tau_decay2, tau_decay3, B1, B2, x0, y0):
        B3 = 1.0 - B1 - B2
        rise0 = 1 - np.exp(-x0 / tau_rise)
        decay0 = B1 * np.exp(-x0 / tau_decay1) + B2 * np.exp(-x0 / tau_decay2) + B3 * np.exp(-x0 / tau_decay3)
        f0 = A * rise0 * decay0
        rise = 1 - np.exp(-x / tau_rise)
        decay = B1 * np.exp(-x / tau_decay1) + B2 * np.exp(-x / tau_decay2) + B3 * np.exp(-x / tau_decay3)
        return A * rise * decay + (y0 - f0)
    
    def exp_func_biexp_double_rise_decay_constrained(x, A, tau_rise1, tau_rise2, B_r, tau_decay1, tau_decay2, B_d, x0, y0):
        rise0 = B_r * (1 - np.exp(-x0 / tau_rise1)) + (1 - B_r) * (1 - np.exp(-x0 / tau_rise2))
        decay0 = B_d * np.exp(-x0 / tau_decay1) + (1 - B_d) * np.exp(-x0 / tau_decay2)
        f0 = A * rise0 * decay0
        rise = B_r * (1 - np.exp(-x / tau_rise1)) + (1 - B_r) * (1 - np.exp(-x / tau_rise2))
        decay = B_d * np.exp(-x / tau_decay1) + (1 - B_d) * np.exp(-x / tau_decay2)
        return A * rise * decay + (y0 - f0)

    # constrained to 0
    def alpha_func_constrained_to_zero(x, A, tau):
        return A * x * np.exp(-x / tau)
    def exp_func_biexp_constrained_to_zero(x, A, tau_rise, tau_decay, c=0):
        return A * (1 - np.exp(-x / tau_rise)) * np.exp(-x / tau_decay) + c
    def exp_func_biexp_double_decay_constrained_to_zero(x, A, tau_rise, tau_decay1, tau_decay2, B, c=0):
        rise = 1 - np.exp(-x / tau_rise)
        decay = B * np.exp(-x / tau_decay1) + (1 - B) * np.exp(-x / tau_decay2)
        return A * rise * decay + c
    def exp_func_triexp_decay_constrained_to_zero(x, A, tau_rise, tau_decay1, tau_decay2, tau_decay3, B1, B2, c=0):
        rise = 1 - np.exp(-x / tau_rise)
        B3 = 1.0 - B1 - B2
        decay = B1 * np.exp(-x / tau_decay1) + B2 * np.exp(-x / tau_decay2) + B3 * np.exp(-x / tau_decay3)
        return A * rise * decay + c
    def exp_func_biexp_double_rise_decay_constrained_to_zero(x, A, tau_rise1, tau_rise2, B_r, tau_decay1, tau_decay2, B_d, c=0):
        rise = B_r * (1 - np.exp(-x / tau_rise1)) + (1 - B_r) * (1 - np.exp(-x / tau_rise2))
        decay = B_d * np.exp(-x / tau_decay1) + (1 - B_d) * np.exp(-x / tau_decay2)
        return A * rise * decay + c
    
    # === FIT EXP GENERICO CON OPZIONE VINCOLO ===
    def fit_exp_model(x, y, model='singlerise', constrain_start=None):
        x = np.array(x)
        y = np.array(y)
        if len(x) < 3:
            return np.zeros_like(x), [np.nan] * (5 if model == 'doubledecay' else 3)

        elif model == 'wind_alpha':
            if constrain_start is not None:
                x0, y0 = constrain_start
                def func(x, tau):
                    return alpha_func_constrained(x, tau, x0, y0)
                p0 = [0.01]
                bounds = ([1e-6], [np.inf])
                popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=5000)
                yfit = func(x, *popt)
                return yfit, [*popt, x0, y0]  # se vuoi salvare anche x0, y0
            else:
                # fallback: libera
                p0 = [0, 0.01, np.min(y)]
                bounds = ([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
                popt, _ = curve_fit(alpha_func, x, y, p0=p0, bounds=bounds, maxfev=5000)
                yfit = alpha_func(x, *popt)
                return yfit, popt

        if model == 'wind_singlerise':
            if constrain_start is not None:
                x0, y0 = constrain_start
                def func(x, tau, c):
                    return exp_func_single_rise_constrained(x, tau, c, x0, y0)
                p0 = [0.01, y0]
                popt, _ = curve_fit(func, x, y, p0=p0, maxfev=5000)
                return func(x, *popt), [*popt]
            else:
                p0 = [np.max(y), 0.01, np.min(y)]
                bounds = ([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
                popt, _ = curve_fit(exp_func_single_rise, x, y, p0=p0, bounds=bounds, maxfev=5000)
                yfit = exp_func_single_rise(x, *popt)
                return yfit, popt

        elif model == 'wind_powerlaw':
            if constrain_start is not None:
                x0, y0 = constrain_start
                def func(x, b, c):
                    return power_law_decay_constrained(x, b, c, x0, y0)
                p0 = [1.0, 0.0]
                popt, _ = curve_fit(func, x, y, p0=p0, maxfev=5000)
                return func(x, *popt), [*popt]
            else:
                p0 = [1.0, 0.0, np.min(y)]
                bounds = ([0.001, 0.0, -np.inf], [5.0, 10.0, np.inf])
                popt, _ = curve_fit(power_law_decay, x, y, p0=p0, bounds=bounds, maxfev=5000)
                return power_law_decay(x, *popt), popt

        elif model == 'wind_singledecay':
            if constrain_start is not None:
                x0, y0 = constrain_start
                def func(x, tau, c):
                    return exp_func_single_decay_constrained(x, tau, c, x0, y0)
                p0 = [0.01, y0]
                popt, _ = curve_fit(func, x, y, p0=p0, maxfev=5000)
                return func(x, *popt), [*popt]
            else:
                p0 = [0.01, 0.01, 0.0]
                bounds = ([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
                popt, _ = curve_fit(exp_func_single_decay, x, y, p0=p0, bounds=bounds, maxfev=5000)
                return exp_func_single_decay(x, *popt), popt

        elif model == 'wind_doubledecay':
            if len(x) < 5:
                return np.zeros_like(x), [np.nan] * 5
            if constrain_start is not None:
                x0, y0 = constrain_start
                def func(x, tau1, tau2, a2, c):
                    return exp_func_double_decay_constrained(x, tau1, tau2, a2, c, x0, y0)
                p0 = [0.01, 0.1, y0/2, y0]
                popt, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
                return func(x, *popt), [*popt]
            else:
                p0 = [0.01, 0.1, 0.01, 0.01, 0.0]
                popt, _ = curve_fit(exp_func_double_decay, x, y, p0=p0, maxfev=10000)
                return exp_func_double_decay(x, *popt), popt

        
        elif model == 'nowind_biexp':
            try:
                if constrain_start:
                    x0, y0 = constrain_start
                    def func(x, A, tau_rise, tau_decay):
                        return exp_func_biexp_constrained(x, A, tau_rise, tau_decay, x0, y0)
                    p0 = [0, 0.01, 0.05]
                    bounds = ([-np.inf, 1e-6, 1e-6], [np.inf, 1.0, 1.0])
                    popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=10000)
                    yfit = func(x, *popt)
                    return yfit, list(popt) + [x0, y0]
                else:
                    p0 = [0, 0.01, 0.05, 0.0]  # A, tau_rise, tau_decay, c
                    bounds = ([-np.inf, 1e-6, 1e-6, -np.inf], [np.inf, 1.0, 1.0, np.inf])
                    popt, _ = curve_fit(exp_func_biexp, x, y, p0=p0, bounds=bounds, maxfev=10000)
                    yfit = exp_func_biexp(x, *popt)
                    return yfit, popt
            except Exception as e:
                print(f"[fit_exp_model] Fit failed for model biexp: {e}")
                print(f"[DEBUG] p0 = {p0}")
                print(f"[DEBUG] bounds = {bounds}")
                return np.full_like(x, np.nan), [np.nan] * 4

        elif model == 'nowind_biexpdouble':
            try:
                if constrain_start:
                    x0, y0 = constrain_start
                    def func(x, A, tau_rise, tau_decay1, tau_decay2):
                        return exp_func_biexp_double_decay_constrained(x, A, tau_rise, tau_decay1, tau_decay2, x0, y0)
                    p0 = [0, 0.01, 0.05, 0.1, 0.5]
                    bounds = ([-np.inf, 1e-6, 1e-6, 1e-6, 0], [np.inf, 1.0, 1.0, 1.0, 1])
                    popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=10000)
                    yfit = func(x, *popt)
                    return yfit, list(popt) + [x0, y0]
                else:
                    # A, tau_rise, tau_decay1, tau_decay2, B, c
                    p0 = [0, 0.01, 0.05, 0.1, 0.5, 0.0]
                    bounds = ([-np.inf, 1e-6, 1e-6, 1e-6, 0, -np.inf], [np.inf, 1.0, 1.0, 1.0, 1, np.inf])
                    popt, _ = curve_fit(exp_func_biexp_double_decay, x, y, p0=p0, bounds=bounds, maxfev=10000)
                    yfit = exp_func_biexp_double_decay(x, *popt)
                    return yfit, popt
            except Exception as e:
                print(f"[fit_exp_model] Fit failed for model biexpdouble: {e}")
                print(f"[DEBUG] p0 = {p0}")
                print(f"[DEBUG] bounds = {bounds}")
                print("[INFO] Falling back to biexp model.")
                # Fallback automatico a biexp
                try:
                    if constrain_start:
                        x0, y0 = constrain_start
                        def func(x, A, tau_rise, tau_decay):
                            return exp_func_biexp_constrained(x, A, tau_rise, tau_decay, x0, y0)
                        p0 = [0, 0.01, 0.05]
                        bounds = ([-np.inf, 1e-6, 1e-6], [np.inf, 1.0, 1.0])
                        popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=10000)
                        yfit = func(x, *popt)
                        return yfit, list(popt) + [x0, y0]
                    else:
                        p0 = [0, 0.01, 0.05, 0.0]
                        bounds = ([-np.inf, 1e-6, 1e-6, -np.inf], [np.inf, 1.0, 1.0, np.inf])
                        popt, _ = curve_fit(exp_func_biexp, x, y, p0=p0, bounds=bounds, maxfev=10000)
                        yfit = exp_func_biexp(x, *popt)
                        return yfit, popt
                except Exception as e2:
                    print(f"[fit_exp_model] Fallback to biexp also failed: {e2}")
                    return np.full_like(x, np.nan), [np.nan] * (4 if not constrain_start else 3)

        elif model == 'nowind_biexptriple':
            try:
                if constrain_start:
                    x0, y0 = constrain_start
                    def func(x, A, tau_rise, tau_decay):
                        return exp_func_triexp_decay_constrained(x, A, tau_rise, tau_decay1, tau_decay2, tau_decay3, x0, y0)                   
                    p0 = [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.3]
                    bounds = ([-np.inf, 1e-6, 1e-6, 1e-6, 1e-6, 0, 0],
                              [np.inf, 1.0, 1.0, 1.0, 1.0, 1, 1])
                    popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=10000)
                    yfit = func(x, *popt)
                    return yfit, list(popt) + [x0, y0]
                else:
                    # A, tau_rise, tau_decay1, tau_decay2, tau_decay3, B1, B2, c
                    p0 = [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.3, 0.0]
                    bounds = ([-np.inf, 1e-6, 1e-6, 1e-6, 1e-6, 0, 0, -np.inf],
                              [np.inf, 1.0, 1.0, 1.0, 1.0, 1, 1, np.inf])
                    popt, _ = curve_fit(exp_func_triexp_decay, x, y, p0=p0, bounds=bounds, maxfev=10000)
                    yfit = exp_func_triexp_decay(x, *popt)
                    return yfit, popt
            except Exception as e:
                print(f"[fit_exp_model] Fit failed for model biexpdouble: {e}")
                print(f"[DEBUG] p0 = {p0}")
                print(f"[DEBUG] bounds = {bounds}")
                print("[INFO] Falling back to biexp model.")
                # Fallback automatico al modello biexp
                try:
                    if constrain_start:
                        x0, y0 = constrain_start
                        def func(x, A, tau_rise, tau_decay):
                            return exp_func_biexp_constrained(x, A, tau_rise, tau_decay, x0, y0)    
                        p0 = [0, 0.01, 0.05]  # A, tau_rise, tau_decay
                        bounds = ([-np.inf, 1e-6, 1e-6], [np.inf, 1.0, 1.0])
                        popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=10000)
                        yfit = func(x, *popt)
                        return yfit, list(popt) + [x0, y0]
                    else:
                        p0 = [0, 0.01, 0.05, 0.0]  # A, tau_rise, tau_decay, c
                        bounds = ([-np.inf, 1e-6, 1e-6, -np.inf], [np.inf, 1.0, 1.0, np.inf])
                        popt, _ = curve_fit(exp_func_biexp, x, y, p0=p0, bounds=bounds, maxfev=10000)
                        yfit = exp_func_biexp(x, *popt)
                        return yfit, popt
                except Exception as e2:
                    print(f"[fit_exp_model] Fallback to biexp also failed: {e2}")
                    return np.full_like(x, np.nan), [np.nan] * (4 if not constrain_start else 3)

        elif model == 'nowind_biexpdoublerd':
            try:
                if constrain_start:
                    x0, y0 = constrain_start
                    def func(x, A, tau_rise, tau_decay):
                        return exp_func_biexp_double_rise_decay_constrained(x, A, tau_rise, tau_decay, x0, y0)   
                    p0 = [0, 0.01, 0.05, 0.5, 0.05, 0.1, 0.5]
                    bounds = ([-np.inf, 1e-6, 1e-6, 0, 1e-6, 1e-6, 0],
                              [np.inf, 1.0, 1.0, 1, 1.0, 1.0, 1])
                    popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=10000)
                    yfit = func(x, *popt)
                    return yfit, list(popt) + [x0, y0]
                else:
                    # A, tau_rise1, tau_rise2, B_r, tau_decay1, tau_decay2, B_d, c
                    p0 = [0, 0.01, 0.05, 0.5, 0.05, 0.1, 0.5, 0.0]
                    bounds = ([-np.inf, 1e-6, 1e-6, 0, 1e-6, 1e-6, 0, -np.inf],
                              [np.inf, 1.0, 1.0, 1, 1.0, 1.0, 1, np.inf])
                    popt, _ = curve_fit(exp_func_biexp_double_rise_decay, x, y, p0=p0, bounds=bounds, maxfev=10000)
                    yfit = exp_func_biexp_double_rise_decay(x, *popt)
                    return yfit, popt
            except Exception as e:
                print(f"[fit_exp_model] Fit failed for model biexpdoublerd: {e}")
                print(f"[DEBUG] p0 = {p0}")
                print(f"[DEBUG] bounds = {bounds}")
                print("[INFO] Falling back to biexp model.")
                # Fallback automatico al modello biexp
                try:
                    if constrain_start:
                        x0, y0 = constrain_start
                        def func(x, A, tau_rise, tau_decay):
                            return exp_func_biexp_constrained(x, A, tau_rise, tau_decay, x0, y0)   
                        p0 = [0, 0.01, 0.05]  # A, tau_rise, tau_decay
                        bounds = ([-np.inf, 1e-6, 1e-6], [np.inf, 1.0, 1.0])
                        popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=10000)
                        yfit = func(x, *popt)
                        return yfit, list(popt) + [x0, y0]
                    else:
                        p0 = [0, 0.01, 0.05, 0.0]  # A, tau_rise, tau_decay, c
                        bounds = ([-np.inf, 1e-6, 1e-6, -np.inf], [np.inf, 1.0, 1.0, np.inf])
                        popt, _ = curve_fit(exp_func_biexp, x, y, p0=p0, bounds=bounds, maxfev=10000)
                        yfit = exp_func_biexp(x, *popt)
                        return yfit, popt
                except Exception as e2:
                    print(f"[fit_exp_model] Fallback to biexp also failed: {e2}")
                    return np.full_like(x, np.nan), [np.nan] * (4 if not constrain_start else 3)

        elif model == 'nowind_alpha':
            if constrain_start is not None:
                x0, y0 = constrain_start
                def func(x, A, tau_rise, tau_decay):
                    return alpha_func_constrained(x, A, tau_rise, x0, y0)   
                p0 = [0, 0.01]  # A, tau
                bounds = ([-np.inf, 1e-6], [np.inf, np.inf])
                try:
                    popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=5000)
                    yfit = func(x, *popt)
                    return yfit, list(popt)  + [x0, y0]
                except Exception as e:
                    print(f"[fit_exp_model] Fit failed for alpha (c=0): {e}")
                    return np.full_like(x, np.nan), [np.nan] * 3
            else:
                p0 = [0, 0.01, np.min(y)]
                bounds = ([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
                try:
                    popt, _ = curve_fit(alpha_func, x, y, p0=p0, bounds=bounds, maxfev=5000)
                    yfit = alpha_func(x, *popt)
                    return yfit, popt
                except Exception as e:
                    print(f"[fit_exp_model] Fit failed for alpha: {e}")
                    return np.full_like(x, np.nan), [np.nan] * 3
        else:
            raise ValueError("Unsupported model. Available models: 'singlerise', 'singledecay', 'doubledecay', 'biexp', 'biexpdouble', 'biexptriple', 'biexpdoublerd', 'alpha', 'powerlaw'")
   
    times = EPOCHS.times
    data_detrended = EPOCHS.get_data().copy()
    tabStat = []
    mse_list = []
    fitted_params_post = []
    max_order_offset_list = []
    markerChans = offsetChans
    nplot = 10
    totplot = EPOCHS.get_data().shape[0]*len(markerChans)
    print(totplot)
    PROB = nplot/totplot
    print('p plot', PROB)

    for chan in tqdm(EPOCHS.ch_names):

        if doDetrendOnlyOffsetChans and chan not in offsetChans:
            continue  # salta questo canale, mantiene i dati originali

        id_chan = np.where(np.array(EPOCHS.ch_names) == chan)[0][0]
        for epoch_idx in range(data_detrended.shape[0]):
            tep = data_detrended[epoch_idx, id_chan, :].reshape(-1, 1)
            #print("len(tep) =", len(tep))
            #assert len(tep) == len(times), f"len(tep)={len(tep)} len(times)={len(times)}"
            
            TEP = tep.copy()
            timeMask = computeTimeMasks(EPOCHS, id_chan, epoch_idx, json_data, offset=json_data['detrend_maxTimeWindowOffset'])
            """
            print("len(times) =", len(times))
            print("len(mask0) =", len(timeMask[0]))
            print("len(mask1) =", len(timeMask[1]))
            print("len(mask2) =", len(timeMask[2]))
            assert len(timeMask[0]) == len(times)
            assert len(timeMask[1]) == len(times)
            assert len(timeMask[2]) == len(times)
            """
            # --- PRE OFFSET detrending (segmento centrale: timeMask[0]) ---
            poly_coeffs_A = np.polyfit(times[timeMask[0]], tep[timeMask[0]].flatten(), json_data['detrend_polOrder_preOffset'])
            trend_line_A = np.polyval(poly_coeffs_A, times[timeMask[0]])
            tepA = tep[timeMask[0]].squeeze() - trend_line_A
            mseA = np.mean((tepA) ** 2)
            OPTPARS_A = f'wind_poly_{json_data['detrend_polOrder_preOffset']}'

            # --- NOWIND FIT ---
            if typeOffsetRise == typeOffsetDecay and typeOffsetRise.startswith('nowind_'):
                start_idx = np.where(timeMask[1])[0][0]
                timeMask_ext = np.zeros_like(times, dtype=bool)
                timeMask_ext[start_idx:] = True
                x_fit = times[timeMask_ext]
                y_fit = tep[timeMask_ext].flatten()
                if typeOffsetRise == 'nowind_poly':
                    # === Ordine ottimale come sqrt(round) dell'ordine lagrange ===
                    n_fit = len(x_fit)
                    order_lagrange = n_fit - 1
                    order_guess = int(np.round(np.sqrt(order_lagrange)))
                    order_guess = 5 #int(np.round(max(1, order_guess)*0.5))
                    if fitConstraint:
                        x0 = times[timeMask[0]][-1]
                        y0 = tep[timeMask[0]][-1][0]
                        best_order = order_guess
                        best_mse = np.inf
                        trend_line, coeffs = polyfit_constrained_start(x_fit, y_fit, order_guess, x0=x0, y0=y0)
                        mse = np.mean((y_fit - trend_line) ** 2)
                        trend_line_B = trend_line
                        popt_B = best_order
                    else:
                        best_order, best_coeffs = fit_polynomial_curvefit(x_fit, y_fit, max_order=order_guess)
                        trend_line_B = poly_func(x_fit, *best_coeffs)
                        popt_B = best_order
                else:
                    # === Modelli esponenziali, alpha, biexp, ecc. ===
                    trend_line_B, popt_B = fit_exp_model(
                        x_fit,
                        y_fit,
                        model=typeOffsetRise,
                        constrain_start=(times[timeMask[0]][-1], tep[timeMask[0]][-1][0]) if fitConstraint else None
                    )
                # === Common detrending post-model ===
                y_detrended = y_fit - trend_line_B
                tepB = y_detrended[:np.sum(timeMask[1])]
                tepC = y_detrended[np.sum(timeMask[1]):]
                mseB = np.mean(tepB ** 2)
                mseC = np.mean(tepC ** 2)
                max_order_offset_list.append([chan, epoch_idx, f"{typeOffsetRise}_({popt_B})"])
                OPTPARS_B = popt_B
                OPTPARS_C = popt_B
                tabStat.append([chan, epoch_idx, mseA, mseB, mseC, OPTPARS_A, OPTPARS_B, OPTPARS_C])
                mse_list.append(mseA + mseB + mseC)
                # 16102025
                #tep_agg = np.concatenate((tepA, tepB, tepC), axis=0)
                # === dopo aver calcolato tepA, tepB, tepC ===
                # Costruisco un vettore full-length (stessa lunghezza di tep / EPOCHS.times)
                tep_agg = tep.flatten().copy()
                tep_agg[timeMask[0]] = tepA
                tep_agg[timeMask[1]] = tepB
                tep_agg[timeMask[2]] = tepC
                
                if lag_correction:
                    tep_agg_shifted, n_shift = shift_signal_by_mask(tep_agg, timeMask[1])
                if correctMode!=False:
                    tep_agg = apply_offset_correction(tep_agg, tep, timeMask, correctMode, oddSamples, EPOCHS, supported_models)
                data_detrended[epoch_idx, id_chan, :] = tep_agg
                is_marker = chan in markerChans
                plot_detrend_example_v3(
                    typeOffsetRise=typeOffsetRise,
                    typeOffsetDecay=typeOffsetDecay,
                    OPTPARS_A=OPTPARS_A,
                    OPTPARS_B=OPTPARS_B,
                    OPTPARS_C=OPTPARS_C,
                    sub=sub,
                    chan=chan,
                    epoch_idx=epoch_idx,
                    times=times,
                    TEP=TEP,
                    trend_line_A=trend_line_A,
                    trend_line_B=trend_line_B[:np.sum(timeMask[1])],
                    trend_line_C=trend_line_B[np.sum(timeMask[1]):],
                    tep_agg=tep_agg,
                    timeMask=timeMask,
                    fitConstraint=fitConstraint,
                    markerChan=is_marker,
                    experiment_dir=experiment_dir,
                    PROB=PROB
                )
                continue  # Salta post-processing standard
            
            # --- WIND FIT ---
            n_offset = len(times[timeMask[1]])
            x_offset = times[timeMask[1]]
            y_offset = tep[timeMask[1]].flatten()
            if isinstance(typeOffsetRise, str):
                typeOffsetRise_lower = typeOffsetRise.lower()
            # -- parsing per wind_poly_xx --
            if typeOffsetRise_lower.startswith("wind_poly_"):
                suffix = typeOffsetRise_lower.split('_')[-1]
                if suffix == "lagrange":
                    order_poly_offset = n_offset - 1
                    poly_coeffs_B = np.polyfit(x_offset, y_offset, order_poly_offset)
                    trend_line_B = np.polyval(poly_coeffs_B, x_offset)
                    tepB = y_offset - trend_line_B
                    mseB = np.mean((tepB) ** 2)
                    max_order_offset_list.append([chan, epoch_idx, f"lagrange({order_poly_offset})"])
                    OPTPARS_B = order_poly_offset
                elif suffix == "opt":
                    if json_data.get('detrend_offsetStart', False):
                        target_time = json_data.get('detrend_minTimeWindowOffset', x_offset[0])
                        idx_local = np.argmin(np.abs(x_offset - target_time))
                        x0_start = x_offset[idx_local]
                        y0_start = y_offset[idx_local]
                        best_order = 1
                        best_mse = np.inf
                        best_coeffs = None
                        best_trend = None
                        for order in range(1, 4):
                            try:
                                trend, coeffs = polyfit_constrained_start(x_offset, y_offset, order, x0=x0_start, y0=y0_start)
                                mse = np.mean((y_offset - trend) ** 2)
                                if mse < best_mse:
                                    best_order = order
                                    best_mse = mse
                                    best_coeffs = coeffs
                                    best_trend = trend
                            except Exception:
                                continue
                        trend_line_B = best_trend
                        tepB = y_offset - trend_line_B
                        mseB = best_mse
                        max_order_offset_list.append([chan, epoch_idx, f"poly_opt_constr({best_order})"])
                        OPTPARS_B = best_order
                    else:
                        best_order, best_coeffs = fit_polynomial_curvefit(x_offset, y_offset, max_order=3)
                        trend_line_B = poly_func(x_offset, *best_coeffs)
                        tepB = y_offset - trend_line_B
                        mseB = np.mean((tepB) ** 2)
                        max_order_offset_list.append([chan, epoch_idx, f"poly_opt({best_order})"])
                        OPTPARS_B = best_order
                elif suffix.isdigit():
                    order_poly_offset = int(suffix)
                    if json_data.get('detrend_offsetStart', False):
                        target_time = json_data.get('detrend_minTimeWindowOffset', x_offset[0])
                        idx_local = np.argmin(np.abs(x_offset - target_time))
                        x0_start = x_offset[idx_local]
                        y0_start = y_offset[idx_local]
                        trend_line_B, poly_coeffs_B = polyfit_constrained_start(x_offset, y_offset, order_poly_offset, x0_start, y0_start)
                    else:
                        poly_coeffs_B = np.polyfit(x_offset, y_offset, order_poly_offset)
                        trend_line_B = np.polyval(poly_coeffs_B, x_offset)
                    tepB = y_offset - trend_line_B
                    mseB = np.mean((tepB) ** 2)
                    max_order_offset_list.append([chan, epoch_idx, f"poly_{order_poly_offset}"])
                    OPTPARS_B = order_poly_offset
            
                else:
                    raise ValueError(f"❌ Suffix '{suffix}' in '{typeOffsetRise}' non riconosciuto.")
            elif typeOffsetRise_lower == 'wind_spline':
                from scipy.interpolate import UnivariateSpline
                spline = UnivariateSpline(x_offset, y_offset, s=1e-14)
                trend_line_B = spline(x_offset)
                tepB = y_offset - trend_line_B
                mseB = np.mean(tepB ** 2)
                max_order_offset_list.append([chan, epoch_idx, 'wind_spline'])
                OPTPARS_B = typeOffsetRise_lower
            elif typeOffsetRise_lower in ['wind_singlerise', 'wind_powerlaw']:
                try:
                    target_time = json_data['detrend_minTimeWindowOffset']
                    idx_local = np.argmin(np.abs(x_offset - target_time))
                    x0_start = x_offset[idx_local]
                    y0_start = y_offset[idx_local]
                    trend_line_B, popt_B = fit_exp_model(
                        x_offset,
                        y_offset,
                        model=typeOffsetRise_lower,
                        constrain_start=(x0_start, y0_start) if json_data.get('detrend_offsetStart', False) else None
                    )
                    tepB = y_offset - trend_line_B
                    mseB = np.mean(tepB ** 2)
                    max_order_offset_list.append([chan, epoch_idx, f"{typeOffsetRise_lower}_exp"])
                    OPTPARS_B = popt_B
                except Exception as e:
                    order_poly_offset = n_offset - 1
                    print(f"[WARNING] Fitting {typeOffsetRise_lower} failed on chan {chan}, trial {epoch_idx} – fallback to lagrange. Error: {e}")
                    poly_coeffs_B = np.polyfit(x_offset, y_offset, order_poly_offset)
                    trend_line_B = np.polyval(poly_coeffs_B, x_offset)
                    tepB = y_offset - trend_line_B
                    mseB = np.mean((tepB) ** 2)
                    max_order_offset_list.append([chan, epoch_idx, f"fallback_lagrange({order_poly_offset})"])
                    OPTPARS_B = order_poly_offset
            else:
                raise ValueError(f"typeOffset '{typeOffsetRise}' non riconosciuto.")

            # --- POST OFFSET detrending (segment: timeMask[2]) ---
            # exp power law
            if typeOffsetDecay in ['wind_singledecay', 'wind_doubledecay', 'wind_powerlaw']:
                y0_start = tep[timeMask[1]][-1][0]
                x0_start = times[timeMask[1]][-1]
                try:
                    trend_line_C, popt_C = fit_exp_model(
                        times[timeMask[2]],
                        tep[timeMask[2]].flatten(),
                        model=typeOffsetDecay,
                        constrain_start=(x0_start, y0_start) if fitConstraint else None
                    )
                    model_used = typeOffsetDecay
                except Exception as e:
                    print(f"[⚠️] Errore nel fit '{typeOffsetDecay}' su {chan}, epoca {epoch_idx}: {e} – fallback su 'single'")
                    trend_line_C, popt_C = fit_exp_model(
                        times[timeMask[2]],
                        tep[timeMask[2]].flatten(),
                        model='wind_singledecay',
                        constrain_start=(x0_start, y0_start) if fitConstraint else None
                    )
                    model_used = 'wind_singledecay'
                OPTPARS_C = popt_C
                tepC = tep[timeMask[2]].squeeze() - trend_line_C
                mseC = np.mean((tep[timeMask[2]].squeeze() - trend_line_C) ** 2)
                
            # --- POST OFFSET DECAY detrending con wind_poly_* ---
            if isinstance(typeOffsetDecay, str) and typeOffsetDecay.lower().startswith('wind_poly_'):
                suffix_decay = typeOffsetDecay.lower().split('_')[-1]
                maskPostOffset = timeMask[2]
                x_post = times[maskPostOffset]
                y_post = tep[maskPostOffset].flatten()
            
                if suffix_decay == 'opt':
                    if fitConstraint:
                        target_time_post = json_data.get('detrend_minTimeWindowPostOffset', x_post[0])
                        idx_post = np.argmin(np.abs(x_post - target_time_post))
                        x0_post = x_post[idx_post]
                        y0_post = y_post[idx_post]
                        best_order_post = 1
                        best_mse_post = np.inf
                        best_trend_post = None
                        best_coeffs_post = None
                        for order in range(1, 4):
                            try:
                                trend_post, coeffs_post = polyfit_constrained_start(
                                    x_post, y_post, order=order, x0=x0_post, y0=y0_post
                                )
                                mse = np.mean((y_post - trend_post) ** 2)
                                if mse < best_mse_post:
                                    best_order_post = order
                                    best_mse_post = mse
                                    best_trend_post = trend_post
                                    best_coeffs_post = coeffs_post
                            except Exception:
                                continue
                        trend_line_C = best_trend_post
                        tepC = y_post - trend_line_C
                        mseC = best_mse_post
                        OPTPARS_C = best_order_post
                    else:
                        best_order_post, best_coeffs_post = fit_polynomial_curvefit(x_post, y_post, max_order=3)
                        trend_line_C = poly_func(x_post, *best_coeffs_post)
                        tepC = y_post - trend_line_C
                        mseC = np.mean(tepC ** 2)
                        OPTPARS_C = best_order_post
            
                elif suffix_decay.isdigit():
                    order_poly_decay = int(suffix_decay) if chan in offsetChans else max(0, int(suffix_decay) - 1)

                    if fitConstraint:
                        end_idx = np.where(timeMask[1])[0][-1]
                        x0_post = times[end_idx]
                        y0_post = tep[end_idx][0] if tep.ndim == 2 else tep[end_idx]
                        trend_line_C, poly_coeffs_C = polyfit_constrained_start(
                            x_post, y_post, order=order_poly_decay, x0=x0_post, y0=y0_post
                        )
                        tepC = y_post - trend_line_C
                        mseC = np.mean(tepC ** 2)
                    else:
                        poly_coeffs_C = np.polyfit(x_post, y_post, order_poly_decay)
                        trend_line_C = np.polyval(poly_coeffs_C, x_post)
                        tepC = y_post - trend_line_C
                        mseC = np.mean(tepC ** 2)
                    OPTPARS_C = order_poly_decay
                else:
                    raise ValueError(f"❌ Suffix '{suffix_decay}' in '{typeOffsetDecay}' non riconosciuto.")

            # spline
            if typeOffsetDecay == 'wind_spline':
                from scipy.interpolate import UnivariateSpline
                mask_post = timeMask[2]
                x_post = times[mask_post]
                y_post = tep[mask_post].flatten()
                if fitConstraint:
                    # Trova punto iniziale per vincolo
                    end_idx = np.where(timeMask[1])[0][-1]
                    x0 = times[end_idx]
                    y0 = tep[end_idx][0] if tep.ndim == 2 else tep[end_idx]
                    # Shift y così che passi da y0 in x0
                    y_shift = y_post - (y_post[0] - y0)  # forza partenza da y0
                    spline = UnivariateSpline(x_post, y_shift, s=1e-14)
                else:
                    spline = UnivariateSpline(x_post, y_post, s=1e-14)
                trend_line_C = spline(x_post)
                tepC = y_post - trend_line_C
                mseC = np.mean(tepC ** 2)
                mse_list.append(mseA + mseB + mseC)
                OPTPARS_C = typeOffsetDecay
            # pchip
            elif typeOffsetDecay == 'wind_pchip':
                from scipy.interpolate import PchipInterpolator
                mask_post = timeMask[2]
                x_post = times[mask_post]
                y_post = tep[mask_post].flatten()
                if fitConstraint:
                    end_idx = np.where(timeMask[1])[0][-1]
                    x0 = times[end_idx]
                    y0 = tep[end_idx][0] if tep.ndim == 2 else tep[end_idx]
                    delta_y = y_post[0] - y0
                    y_shift = y_post - delta_y
                    pchip = PchipInterpolator(x_post, y_shift)
                else:
                    pchip = PchipInterpolator(x_post, y_post)
                trend_line_C = pchip(x_post)
                tepC = y_post - trend_line_C
                mseC = np.mean(tepC ** 2)
                mse_list.append(mseA + mseB + mseC)
                OPTPARS_C = typeOffsetDecay

            #####################################################################################
            tabStat.append([chan, epoch_idx, mseA, mseB, mseC, OPTPARS_A, OPTPARS_B, OPTPARS_C])
            mse_list.append(mseA + mseB + mseC)
            # 16102025
            #tep_agg = np.concatenate((tepA, tepB, tepC), axis=0)
            # === dopo aver calcolato tepA, tepB, tepC ===
            # Costruisco un vettore full-length (stessa lunghezza di tep / EPOCHS.times)
            tep_agg = tep.flatten().copy()
            tep_agg[timeMask[0]] = tepA
            tep_agg[timeMask[1]] = tepB
            tep_agg[timeMask[2]] = tepC
            #####################################################################################
            if lag_correction:
                tep_agg_shifted, n_shift = shift_signal_by_mask(tep_agg, timeMask[1])
            ######################################################################################
            if correctMode!=False:
                tep_agg = apply_offset_correction(tep_agg, tep, timeMask, correctMode, oddSamples, EPOCHS, supported_models)
            ######################################################################################            
            data_detrended[epoch_idx, id_chan, :] = tep_agg
            ######################################################################################            
            is_marker = chan in markerChans
            plot_detrend_example_v3(
                    typeOffsetRise=typeOffsetRise,
                    typeOffsetDecay=typeOffsetDecay,
                sub=sub,
                chan=chan,
                epoch_idx=epoch_idx,
                times=times,
                TEP=TEP,
                OPTPARS_A=OPTPARS_A,
                OPTPARS_B=OPTPARS_B,
                OPTPARS_C=OPTPARS_C,
                trend_line_A=trend_line_A,
                trend_line_B=trend_line_B,
                trend_line_C=trend_line_C,
                tep_agg=tep_agg,
                timeMask=timeMask,
                fitConstraint=fitConstraint,
                markerChan=is_marker,
                experiment_dir=experiment_dir,
                PROB=PROB
                )

    MSE = np.mean(mse_list)
    EPOCHS_DETRENDED = EPOCHS.copy()
    EPOCHS_DETRENDED._data = np.asarray(data_detrended).copy()   
    print("INPUT data shape:", EPOCHS.get_data().shape)
    print("INPUT times len:", len(EPOCHS.times))
    print("DETRENDED data shape:", EPOCHS_DETRENDED.get_data().shape)
    print("DETRENDED times len:", len(EPOCHS_DETRENDED.times))
    assert EPOCHS_DETRENDED.get_data().shape[2] == len(EPOCHS_DETRENDED.times), \
        f"{EPOCHS_DETRENDED.get_data().shape[2]} vs {len(EPOCHS_DETRENDED.times)}"
    df_tabStat = pd.DataFrame(
        tabStat,
        columns=['chan', 'epoch_idx', 'mseA', 'mseB', 'mseC', 'OPTPARS_A', 'OPTPARS_B', 'OPTPARS_C']
    )
    tabStat_path = os.path.join(experiment_dir, '2.detrend', f'tabStatDetrend_{typeOffsetDecay}.csv')
    df_tabStat.to_csv(tabStat_path, index=False)
    print(f"[INFO] Salvato tabStat in: {tabStat_path}")

    return EPOCHS_DETRENDED, MSE, max_order_offset_list


def plot_detrend_example_v3(sub,   
                            typeOffsetRise,
                            typeOffsetDecay,
                            OPTPARS_A,
                            OPTPARS_B,
                            OPTPARS_C,
                            chan, epoch_idx, times, TEP, trend_line_A, trend_line_B, trend_line_C,
                            tep_agg, timeMask, fitConstraint,
                         #detrendMode, orderPreOffset, orderOffset,
                         #orderPostOffset_temp, popt_C=None, popt_B=None, 
                         markerChan=False,
                         experiment_dir=None, PROB=1.0):

    # Decide number of subplots and layout
    if markerChan and (epoch_idx in [1, 10, 20, 30, 40]):
        fig, ax = plt.subplots(2, 2, figsize=(18, 11))
        ax = ax.flatten()
    elif not markerChan and np.random.rand() < PROB * 0:
        fig, ax = plt.subplots(1, 2, figsize=(13, 7), sharex=True, sharey=True)
    else:
        return  # Skip plotting

    min_offset_time = times[timeMask[1]].min()
    max_offset_time = times[timeMask[1]].max()
    title = (
        f'{sub} | {chan} | epoch={epoch_idx} \n fitConstrain={fitConstraint} \n '
#        f'orderPre={orderPreOffset} | orderArt={orderOffset} | orderPost={orderPostOffset_temp}\n'
        f'Min Offset Time: {min_offset_time:.4f} | Max Offset Time: {max_offset_time:.4f} \n'
        f'typeOffsetRise={typeOffsetRise} with pars={OPTPARS_B} \n'
        f'typeOffsetDecay={typeOffsetDecay} with pars={OPTPARS_C} \n'     
    )
    filename = f'{sub}_{chan}_{epoch_idx}_{fitConstraint}.png'

    fig.suptitle(title)

    # First subplot: full fit
    ax[0].plot(times, TEP, 'k', label='Original', marker='+')
    ax[0].plot(times[timeMask[0]], trend_line_A, c='b', label='fit prestimulus', alpha=0.5, linewidth=4)
    ax[0].plot(times[timeMask[1]], trend_line_B, c='orange', label='fit artifacts', alpha=0.5, linewidth=4)
    ax[0].plot(times[timeMask[2]], trend_line_C, c='g', label='fit poststimulus', alpha=.5, linewidth=4)
    ax[0].set_title('Fit')

    # Second subplot: detrended
    ax[1].plot(times, tep_agg, c='r', marker='+')
    ax[1].set_title('TEP Detrended')

    if len(ax) > 2:
        # Third subplot: artifact zoom
        ax[2].plot(times[timeMask[1]], TEP[timeMask[1]], 'k+', label='Original')
        ax[2].plot(times[timeMask[1]], trend_line_B, 'orange', linewidth=3, label='Exp Fit')
        ax[2].axvline(0)
        ax[2].set_title('Artifact Window Focus')
        ax[2].legend()

        # Fourth subplot: prestimulus zoom
        ax[3].plot(times[timeMask[0]], TEP[timeMask[0]], 'k+', label='Original')
        ax[3].plot(times[timeMask[0]], trend_line_A, 'orange', linewidth=3, label='Exp Fit')
        ax[3].axvline(0)
        ax[3].set_title('Prestimulus Window Focus')
        ax[3].legend()

    ax[0].legend(loc='upper left', bbox_to_anchor=(-0.65 if len(ax) > 2 else -1, 1), borderaxespad=0.)
    plt.tight_layout()
    out_path = os.path.join(experiment_dir, '2.detrend', 'examples', filename)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_detrend_example(sub, chan, epoch_idx, times, TEP, trend_line_A, trend_line_B, trend_line_C,
                         tep_agg, timeMask, 
                         detrendMode, fitConstraint, orderPreOffset, orderOffset,
                         orderPostOffset_temp, popt_C=None, popt_B=None, 
                         markerChan=False,
                         experiment_dir=None, PROB=1.0):

    # Decide number of subplots and layout
    if markerChan and (epoch_idx in [1, 10, 20]):
        fig, ax = plt.subplots(2, 2, figsize=(18, 11))
        ax = ax.flatten()
    elif not markerChan and np.random.rand() < PROB * 1:
        fig, ax = plt.subplots(1, 2, figsize=(13, 7), sharex=True, sharey=True)
    else:
        return  # Skip plotting

    min_offset_time = times[timeMask[1]].min()
    max_offset_time = times[timeMask[1]].max()

    # Common title logic
    if detrendMode == 'poly':
        title = (
            f'{sub} | {chan} | epoch={epoch_idx} \n fitConstrain={fitConstraint} \n '
            f'orderPre={orderPreOffset} | orderArt={orderOffset} | orderPost={orderPostOffset_temp}\n'
            f'Min Offset Time: {min_offset_time:.4f} | Max Offset Time: {max_offset_time:.4f}'
        )
        filename = f'{sub}_{chan}_{epoch_idx}_{fitConstraint}_{orderPostOffset_temp}.png'
        
    if detrendMode in ['singleDecay', 'doubleDecay']:
        """
        title = (
            f'{sub} | {chan} | epoch={epoch_idx} \n fitConstrain={fitConstraint} \n '
            f'orderPre={orderPreOffset} | Art exp={orderOffset} with pars={popt_B}\n if popt_C != None else noExp but Lagrange with order={orderOffset}\n'
            f'Min Offset Time: {min_offset_time:.4f} | Max Offset Time: {max_offset_time:.4f} \n '
            f'postArt exp={detrendMode} with pars={popt_C}'
        )
        """
        if popt_C is not None:
            exp_info = f'Art exp={orderOffset} with pars={popt_B}\n'
            postart_info = f'postArt exp={detrendMode} with pars={popt_C}'
        else:
            exp_info = f'noExp but Lagrange with order={orderOffset}\n'
            postart_info = 'no postArt exp fit'
        
        title = (
            f'{sub} | {chan} | epoch={epoch_idx} \n'
            f'fitConstrain={fitConstraint} \n'
            f'orderPre={orderPreOffset} | {exp_info}'
            f'Min Offset Time: {min_offset_time:.4f} | Max Offset Time: {max_offset_time:.4f} \n'
            f'{postart_info}'
        )
        filename = f'{sub}_detrendExample_{chan}_{epoch_idx}_{fitConstraint}_{detrendMode}.png'

    fig.suptitle(title)

    # First subplot: full fit
    ax[0].plot(times, TEP, 'k', label='Original', marker='+')
    ax[0].plot(times[timeMask[0]], trend_line_A, c='b', label='fit prestimulus', alpha=0.5, linewidth=4)
    ax[0].plot(times[timeMask[1]], trend_line_B, c='orange', label='fit artifacts', alpha=0.5, linewidth=4)
    ax[0].plot(times[timeMask[2]], trend_line_C, c='g', label='fit poststimulus', alpha=.5, linewidth=4)
    ax[0].set_title('Fit')

    # Second subplot: detrended
    ax[1].plot(times, tep_agg, c='r', marker='+')
    ax[1].set_title('TEP Detrended')

    if len(ax) > 2:
        # Third subplot: artifact zoom
        ax[2].plot(times[timeMask[1]], TEP[timeMask[1]], 'k+', label='Original')
        ax[2].plot(times[timeMask[1]], trend_line_B, 'orange', linewidth=3, label='Exp Fit')
        ax[2].axvline(0)
        ax[2].set_title('Artifact Window Focus')
        ax[2].legend()

        # Fourth subplot: prestimulus zoom
        ax[3].plot(times[timeMask[0]], TEP[timeMask[0]], 'k+', label='Original')
        ax[3].plot(times[timeMask[0]], trend_line_A, 'orange', linewidth=3, label='Exp Fit')
        ax[3].axvline(0)
        ax[3].set_title('Prestimulus Window Focus')
        ax[3].legend()

    ax[0].legend(loc='upper left', bbox_to_anchor=(-0.65 if len(ax) > 2 else -1, 1), borderaxespad=0.)
    plt.tight_layout()
    out_path = os.path.join(experiment_dir, '2.detrend', 'examples', filename)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_tabStat(df_tabStat, experiment_dir):
    # Crea la cartella per i plot
    output_dir = os.path.join(experiment_dir, '2.detrend', 'statDetrend')
    os.makedirs(output_dir, exist_ok=True)

    # Assicurati che la colonna totale esista
    df_tabStat['MSE_total'] = df_tabStat[['mseA', 'mseB', 'mseC']].sum(axis=1)

    # --- 1. Boxplot per segmento ---
    df_melted = df_tabStat.melt(
        id_vars=['chan', 'epoch_idx'],
        value_vars=['mseA', 'mseB', 'mseC'],
        var_name='Segment',
        value_name='MSE'
    )
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_melted, x='Segment', y='MSE')
    plt.title('Distribuzione MSE per segmento')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_MSE_per_segment.png'), dpi=300)
    plt.close()

    # --- 2. Heatmap MSE totale ---
    pivot = df_tabStat.pivot(index='chan', columns='epoch_idx', values='MSE_total')
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap='viridis')
    plt.title('Heatmap MSE totale per canale ed epoca')
    plt.xlabel('Epoch')
    plt.ylabel('Canale')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_MSE_total.png'), dpi=300)
    plt.close()

    # --- 3. Line plot per epoca ---
    plt.figure(figsize=(10, 5))
    for chan in df_tabStat['chan'].unique():
        df_chan = df_tabStat[df_tabStat['chan'] == chan]
        plt.plot(df_chan['epoch_idx'], df_chan['MSE_total'], label=chan)
    plt.title('MSE totale per epoca per canale')
    plt.xlabel('Epoch')
    plt.ylabel('MSE totale')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lineplot_MSE_total_per_epoch.png'), dpi=300)
    plt.close()

    # --- 4. Barplot per canale ---
    df_grouped = df_tabStat.groupby('chan')['MSE_total'].mean().reset_index()
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_grouped, x='chan', y='MSE_total')
    plt.title('MSE medio per canale')
    plt.ylabel('MSE totale (media su epoche)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'barplot_MSE_per_channel.png'), dpi=300)
    plt.close()

    print(f"[INFO] Plot salvati in: {output_dir}")


def apply_offset_correction(tep_agg, tep, timeMask, correctMode, oddSamples, EPOCHS, supported_models):
    # lunghezze nei dati originali
    n_pre  = int(np.count_nonzero(timeMask[0]))
    n_off  = int(np.count_nonzero(timeMask[1]))
    n_post = int(np.count_nonzero(timeMask[2]))
    n_agg  = int(tep_agg.shape[0])

    # Caso “pulito”: tep_agg = concat([pre, offset, post])
    if n_pre + n_off + n_post == n_agg:
        correctionMask      = np.r_[np.zeros(n_pre,  dtype=bool),
                                    np.ones(n_off,   dtype=bool),
                                    np.zeros(n_post, dtype=bool)]
        # pre-correzione: ultimi oddSamples del pre (se oddSamples=0 usa tutto il pre)
        k = n_pre if (oddSamples is False or oddSamples is None or oddSamples<=0) else min(oddSamples, n_pre)
        precorrectionMask   = np.r_[np.zeros(n_pre - k, dtype=bool),
                                    np.ones(k,          dtype=bool),
                                    np.zeros(n_off + n_post, dtype=bool)]
        pre_series = tep_agg[precorrectionMask]

    else:
        # fallback: riallinea maschere dalla base originale a tep_agg (trim/pad)
        # NB: meno ideale, ma evita crash
        baseMask = timeMask[1].astype(bool)  # offset sui dati originali
        correctionMask = baseMask
        if correctionMask.size > n_agg:
            correctionMask = correctionMask[:n_agg]
        elif correctionMask.size < n_agg:
            correctionMask = np.pad(correctionMask, (0, n_agg - correctionMask.size), constant_values=False)

        basePre = timeMask[0].astype(bool)
        precorrectionMask = basePre
        if precorrectionMask.size > n_agg:
            precorrectionMask = precorrectionMask[-n_agg:]  # tieni coda (più vicina all'offset)
        elif precorrectionMask.size < n_agg:
            precorrectionMask = np.pad(precorrectionMask, (n_agg - precorrectionMask.size, 0), constant_values=False)

        # limita a ultimi k campioni del pre
        idx_pre = np.flatnonzero(precorrectionMask)
        if idx_pre.size:
            k = idx_pre.size if (oddSamples is False or oddSamples is None or oddSamples<=0) else min(oddSamples, idx_pre.size)
            keep = idx_pre[-k:]
            precorrectionMask[:] = False
            precorrectionMask[keep] = True

        pre_series = tep_agg[precorrectionMask]

    # Se non c’è nulla da correggere o pre vuoto, esci senza errore
    num_samples = int(np.sum(correctionMask))
    if num_samples == 0: 
        return tep_agg
    if pre_series.size == 0:
        # no-op sicuro
        return tep_agg

    # Genera i nuovi campioni (usa la tua versione robusta)
    new_samples = generate_noise_from_distribution(pre_series, model=correctMode, n_samples=num_samples)

    # Applica correzione (ora le lunghezze combaciano)
    tep_agg[correctionMask] = new_samples
    return tep_agg


def apply_offset_correction(tep_agg, tep, timeMask, correctMode, oddSamples, EPOCHS, supported_models):
    k = oddSamples / 1000  # da ms a secondi

    times = EPOCHS.times
    precorrectionMask = np.logical_and(times >= times[timeMask[0]].min(),
                                       times < (times[timeMask[0]].max() - k))
    correctionMask = np.logical_and(times >= (times[timeMask[1]].min() - k),
                                    times <= (times[timeMask[1]].max() + k))

    if correctMode == 'moving_average':
        tep_flat = tep[precorrectionMask].flatten()
        window_size = oddSamples
        if len(tep_flat) >= window_size:
            new_samples = np.array([
                np.mean(tep_flat[max(0, i - window_size//2):i + window_size//2])
                for i in range(len(tep_flat))
            ])[-1]
        else:
            new_samples = np.mean(tep_flat) if len(tep_flat) > 0 else 0
        tep_agg[correctionMask] = new_samples

    elif correctMode == 'median':
        new_samples = np.median(tep[precorrectionMask].flatten())
        tep_agg[correctionMask] = new_samples

    elif correctMode == 'zeros':
        num_samples = sum(correctionMask)
        tep_agg[correctionMask] = np.zeros(num_samples)

    elif correctMode == 'resample':
        num_samples = sum(correctionMask)
        new_samples = resample(tep[precorrectionMask].flatten(), num=num_samples)
        tep_agg[correctionMask] = new_samples

    elif correctMode in supported_models:
        num_samples = sum(correctionMask)
        new_samples = generate_noise_from_distribution(
            tep[precorrectionMask].flatten(),
            model=correctMode,
            n_samples=num_samples
        )
        tep_agg[correctionMask] = new_samples

    return tep_agg



def apply_offset_correction_old15102025(tep_agg, tep, timeMask, correctMode, oddSamples, EPOCHS, supported_models):
    k = oddSamples / 1000  # da ms a secondi

    times = EPOCHS.times
    precorrectionMask = np.logical_and(times >= times[timeMask[0]].min(),
                                       times < (times[timeMask[0]].max() - k))
    correctionMask = np.logical_and(times >= (times[timeMask[1]].min() - k),
                                    times <= (times[timeMask[1]].max() + k))

    if correctMode == 'moving_average':
        tep_flat = tep[precorrectionMask].flatten()
        window_size = oddSamples
        if len(tep_flat) >= window_size:
            new_samples = np.array([
                np.mean(tep_flat[max(0, i - window_size//2):i + window_size//2])
                for i in range(len(tep_flat))
            ])[-1]
        else:
            new_samples = np.mean(tep_flat) if len(tep_flat) > 0 else 0
        tep_agg[correctionMask] = new_samples

    elif correctMode == 'median':
        new_samples = np.median(tep[precorrectionMask].flatten())
        tep_agg[correctionMask] = new_samples

    elif correctMode == 'zeros':
        num_samples = sum(correctionMask)
        tep_agg[correctionMask] = np.zeros(num_samples)

    elif correctMode == 'resample':
        num_samples = sum(correctionMask)
        new_samples = resample(tep[precorrectionMask].flatten(), num=num_samples)
        tep_agg[correctionMask] = new_samples

    elif correctMode in supported_models:
        num_samples = sum(correctionMask)
        new_samples = generate_noise_from_distribution(
            tep[precorrectionMask].flatten(),
            model=correctMode,
            n_samples=num_samples
        )
        tep_agg[correctionMask] = new_samples

    return tep_agg



def plot_slope_resonances(PSTATS, PSTATS2, saveNote='pol_degree_estimate', subPath='2.detrend'):

    neural_params_dfs = []
    offset_params_dfs = []
    
    for entry in PSTATS:
        integer_value = entry[0]  # Numero intero (grado del polinomio)
        #offset_data = entry[1]    # Offset con p-values
        df_data = entry[2]        # DataFrame con dati EEG
        mse_value = entry[3]      # Valore MSE
        
        # Creazione del DataFrame dei parametri neurali
        df_neural = df_data.copy()
        df_neural['pol_degree'] = integer_value
        neural_params_dfs.append(df_neural)
        
        # Creazione del DataFrame dei parametri di offset
        #df_offset = pd.DataFrame(offset_data, columns=['offset_type', 'p_value'])
        #df_offset['pol_degree'] = integer_value
        #df_offset['mse'] = mse_value
        #offset_params_dfs.append(df_offset)
    
    df_neural_params = pd.concat(neural_params_dfs, ignore_index=True)
    #df_offset_params = pd.concat(offset_params_dfs, ignore_index=True)
    #min_resonance_row = df_neural_params.loc[df_neural_params['n_resonances'].idxmin()]
    #pol_degree_min_resonances = min_resonance_row['pol_degree']
    df_mean_resonances = df_neural_params.groupby('pol_degree')['n_resonances'].mean()
    pol_degree_min_resonances = np.argmin(df_neural_params.groupby('pol_degree')['n_resonances'].mean())+1
    
    fig1, ax1 = plt.subplots(figsize=(13, 6))
    sns.swarmplot(data=df_neural_params, x='pol_degree', y='slope', ax=ax1, color='b', alpha=.25)
    sns.pointplot(data=df_neural_params, x='pol_degree', y='slope', ax=ax1, color='b')
    ax1.set_ylabel('exp of 1/f^exp', color='b')
    ax1.set_xlabel('Polynomial Degree')
    plt.savefig(f'{experiment_dir}\\{subPath}\\{saveNote}_slope.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(13, 6))
    #sns.swarmplot(data=df_neural_params, x='pol_degree', y='n_resonances', ax=ax2, color='r', alpha=.25)
    sns.pointplot(data=df_neural_params, x='pol_degree', y='n_resonances', ax=ax2, color='r')
    ax2.set_title(f"Estimated Pol Degree (min res)={pol_degree_min_resonances}")
    ax2.set_ylabel('Number of Resonances', color='r')
    ax2.set_xlabel('Polynomial Degree')
    plt.savefig(f'{experiment_dir}\\{subPath}\\{saveNote}_resonances.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # F and p da PSTATS2
    df = pd.DataFrame(PSTATS2, columns=['Polynomial Degree', 'Stats'])
    df = df.join(pd.json_normalize(df['Stats'])).drop(columns=['Stats'])
    pol_degree_min_F = df['Polynomial Degree'][np.argmin(df['F'])]

    df = pd.DataFrame(PSTATS2, columns=['Polynomial Degree', 'Stats'])
    df = df.join(pd.json_normalize(df['Stats'])).drop(columns=['Stats'])
    fig3, ax1 = plt.subplots(figsize=(13, 6))
    sns.lineplot(data=df, x='Polynomial Degree', y='p_value', ax=ax1, color='b', label='p-value')
    ax1.set_ylabel('p-value (log scale)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_yscale('log')
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x='Polynomial Degree', y='F', ax=ax2, color='r', label='F-statistic')
    ax2.set_ylabel('F-statistic (log scale)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_yscale('log')
    ax1.set_xlabel('Polynomial Degree')
    ax1.set_title(f"Estimated Pol Degree (min F)={pol_degree_min_F}")
    plt.savefig(f'{experiment_dir}\\{subPath}\\{saveNote}_fp.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(13, 6))
    #sns.swarmplot(data=df_neural_params, x='pol_degree', y='n_resonances', ax=ax2, color='r', alpha=.25)
    sns.pointplot(data=df_neural_params, x='pol_degree', y='fiterror', ax=ax4, color='r')
    ax2.set_title(f"Estimated Pol Degree (min res)={pol_degree_min_resonances}")
    ax2.set_ylabel('Fit Error', color='r')
    ax2.set_xlabel('Polynomial Degree')
    plt.savefig(f'{experiment_dir}\\{subPath}\\{saveNote}_fitError.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)

    fig5, ax5 = plt.subplots(figsize=(13, 6))
    #sns.swarmplot(data=df_neural_params, x='pol_degree', y='n_resonances', ax=ax2, color='r', alpha=.25)
    sns.pointplot(data=df_neural_params, x='pol_degree', y='r2', ax=ax4, color='r')
    ax2.set_title(f"Estimated Pol Degree (min res)={pol_degree_min_resonances}")
    ax2.set_ylabel('Fit Error', color='r')
    ax2.set_xlabel('Polynomial Degree')
    plt.savefig(f'{experiment_dir}\\{subPath}\\{saveNote}_r2.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)

    return df_neural_params, pol_degree_min_resonances, pol_degree_min_F






















    
 