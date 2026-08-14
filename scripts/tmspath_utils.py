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


def replace_raw_with_synthetic_tep_ground_truth(
    raw,
    events,
    json_data,
    frequency_hz=None,
    onset_sec=None,
    tau_sec=None,
    amplitude_uv=None,
    noise_uv=None,
    frequency_jitter_hz=None,
    amplitude_jitter=None,
    phase_jitter_rad=None,
    add_aperiodic_noise=None,
    add_tms_artifact=None,
    tms_artifact_uv=None,
    seed=None
):
    import numpy as np
    import mne

    raw_out=raw.copy().load_data()

    if events is None or len(events)==0:
        raise ValueError(
            "Per creare il ground truth TEP servono eventi TMS validi."
        )

    frequency_hz=float(
        json_data.get(
            "synthetic_tep_frequency_hz",
            20.0 if frequency_hz is None else frequency_hz
        )
    )
    onset_sec=float(
        json_data.get(
            "synthetic_tep_onset_sec",
            0.020 if onset_sec is None else onset_sec
        )
    )
    tau_sec=float(
        json_data.get(
            "synthetic_tep_tau_sec",
            0.180 if tau_sec is None else tau_sec
        )
    )
    amplitude_uv=float(
        json_data.get(
            "synthetic_tep_amplitude_uv",
            12.0 if amplitude_uv is None else amplitude_uv
        )
    )
    noise_uv=float(
        json_data.get(
            "synthetic_tep_noise_uv",
            4.0 if noise_uv is None else noise_uv
        )
    )
    frequency_jitter_hz=float(
        json_data.get(
            "synthetic_tep_frequency_jitter_hz",
            0.0 if frequency_jitter_hz is None else frequency_jitter_hz
        )
    )
    amplitude_jitter=float(
        json_data.get(
            "synthetic_tep_amplitude_jitter",
            0.15 if amplitude_jitter is None else amplitude_jitter
        )
    )
    phase_jitter_rad=float(
        json_data.get(
            "synthetic_tep_phase_jitter_rad",
            0.0 if phase_jitter_rad is None else phase_jitter_rad
        )
    )
    add_aperiodic_noise=bool(
        json_data.get(
            "synthetic_tep_add_aperiodic_noise",
            True if add_aperiodic_noise is None else add_aperiodic_noise
        )
    )
    add_tms_artifact=bool(
        json_data.get(
            "synthetic_tep_add_tms_artifact",
            False if add_tms_artifact is None else add_tms_artifact
        )
    )
    tms_artifact_uv=float(
        json_data.get(
            "synthetic_tep_tms_artifact_uv",
            1000.0 if tms_artifact_uv is None else tms_artifact_uv
        )
    )
    seed=int(
        json_data.get(
            "synthetic_tep_seed",
            42 if seed is None else seed
        )
    )

    if frequency_hz<=0:
        raise ValueError("synthetic_tep_frequency_hz deve essere > 0.")
    if onset_sec<0:
        raise ValueError("synthetic_tep_onset_sec deve essere >= 0.")
    if tau_sec<=0:
        raise ValueError("synthetic_tep_tau_sec deve essere > 0.")

    rng=np.random.default_rng(seed)
    sfreq=float(raw_out.info["sfreq"])
    n_times=int(raw_out.n_times)
    first_samp=int(raw_out.first_samp)

    eeg_picks=mne.pick_types(
        raw_out.info,
        eeg=True,
        exclude=[]
    )

    if len(eeg_picks)==0:
        raise ValueError("Nessun canale EEG disponibile nel Raw.")

    eeg_names=[
        raw_out.ch_names[index]
        for index in eeg_picks
    ]

    seed_channels=[
        channel
        for channel in json_data.get(
            "seedChans",
            []
        )
        if channel in eeg_names
    ]

    if not seed_channels:
        seed_channels=eeg_names[:min(4,len(eeg_names))]

    positions={}
    for channel in raw_out.info["chs"]:
        name=channel["ch_name"]
        loc=np.asarray(channel["loc"][:3],dtype=float)
        if np.all(np.isfinite(loc)) and np.linalg.norm(loc)>0:
            positions[name]=loc

    seed_positions=[
        positions[channel]
        for channel in seed_channels
        if channel in positions
    ]

    if seed_positions:
        seed_center=np.mean(seed_positions,axis=0)
        distances={}
        for channel in eeg_names:
            if channel in positions:
                distances[channel]=float(
                    np.linalg.norm(
                        positions[channel]-seed_center
                    )
                )
            else:
                distances[channel]=np.nan

        finite_distances=np.asarray(
            [value for value in distances.values() if np.isfinite(value)],
            dtype=float
        )

        spatial_scale=float(
            np.nanmedian(finite_distances)
        ) if finite_distances.size else 0.08

        spatial_scale=max(spatial_scale,0.03)

        spatial_weights={
            channel:(
                float(np.exp(-(distances[channel]/spatial_scale)**2))
                if np.isfinite(distances[channel])
                else 0.10
            )
            for channel in eeg_names
        }
    else:
        spatial_weights={
            channel:(1.0 if channel in seed_channels else 0.10)
            for channel in eeg_names
        }

    for channel in seed_channels:
        spatial_weights[channel]=max(
            spatial_weights.get(channel,0.0),
            0.85
        )

    def make_aperiodic_noise():
        white=rng.normal(size=n_times)
        spectrum=np.fft.rfft(white)
        frequencies=np.fft.rfftfreq(
            n_times,
            d=1.0/sfreq
        )

        scaling=np.zeros_like(
            frequencies,
            dtype=float
        )

        valid=frequencies>=0.5
        scaling[valid]=1.0/np.sqrt(
            frequencies[valid]
        )

        spectrum*=scaling
        noise=np.fft.irfft(
            spectrum,
            n=n_times
        )

        noise-=np.mean(noise)
        standard_deviation=np.std(noise)

        if standard_deviation>0:
            noise/=standard_deviation

        return noise

    synthetic_data=np.zeros(
        (len(eeg_picks),n_times),
        dtype=float
    )

    for channel_index in range(len(eeg_picks)):
        if add_aperiodic_noise:
            background=make_aperiodic_noise()
        else:
            background=rng.normal(size=n_times)

        synthetic_data[channel_index]=background*noise_uv*1e-6

    response_duration_sec=float(
        json_data.get(
            "synthetic_tep_response_duration_sec",
            max(0.8,5.0*tau_sec)
        )
    )

    response_samples=max(
        1,
        int(np.ceil(response_duration_sec*sfreq))
    )

    event_samples=np.asarray(events[:,0],dtype=int)-first_samp

    event_frequencies=[]
    event_amplitudes=[]
    event_phases=[]

    for event_sample in event_samples:
        event_frequency=frequency_hz+rng.normal(
            0.0,
            frequency_jitter_hz
        )
        event_frequency=max(0.1,float(event_frequency))

        event_amplitude=amplitude_uv*(
            1.0+rng.normal(0.0,amplitude_jitter)
        )
        event_amplitude=max(0.0,float(event_amplitude))

        event_phase=rng.normal(
            0.0,
            phase_jitter_rad
        )

        start_sample=int(
            event_sample+round(onset_sec*sfreq)
        )
        stop_sample=min(
            n_times,
            start_sample+response_samples
        )

        if start_sample<0 or start_sample>=n_times or stop_sample<=start_sample:
            continue

        relative_time=np.arange(
            stop_sample-start_sample,
            dtype=float
        )/sfreq

        oscillation=(
            event_amplitude
            *1e-6
            *np.exp(-relative_time/tau_sec)
            *np.sin(
                2.0*np.pi*event_frequency*relative_time
                +event_phase
            )
        )

        for channel_index,channel_name in enumerate(eeg_names):
            channel_variability=max(
                0.0,
                1.0+rng.normal(0.0,0.05)
            )

            synthetic_data[
                channel_index,
                start_sample:stop_sample
            ]+=(
                spatial_weights[channel_name]
                *channel_variability
                *oscillation
            )

        if add_tms_artifact:
            artifact_start=max(
                0,
                int(event_sample+round(-0.002*sfreq))
            )
            artifact_stop=min(
                n_times,
                int(event_sample+round(0.008*sfreq))+1
            )

            if artifact_stop>artifact_start:
                artifact_length=artifact_stop-artifact_start
                artifact_envelope=np.hanning(
                    max(3,artifact_length)
                )[:artifact_length]

                signs=rng.choice(
                    [-1.0,1.0],
                    size=(len(eeg_picks),1)
                )

                synthetic_data[
                    :,
                    artifact_start:artifact_stop
                ]+=(
                    signs
                    *tms_artifact_uv
                    *1e-6
                    *artifact_envelope[np.newaxis,:]
                )

        event_frequencies.append(float(event_frequency))
        event_amplitudes.append(float(event_amplitude))
        event_phases.append(float(event_phase))

    raw_out._data[eeg_picks,:]=synthetic_data

    json_data["data_replaced_with_synthetic_tep"]=True
    json_data["synthetic_tep_ground_truth_hz"]=float(frequency_hz)
    json_data["synthetic_tep_frequency_hz"]=float(frequency_hz)
    json_data["synthetic_tep_onset_sec"]=float(onset_sec)
    json_data["synthetic_tep_tau_sec"]=float(tau_sec)
    json_data["synthetic_tep_amplitude_uv"]=float(amplitude_uv)
    json_data["synthetic_tep_noise_uv"]=float(noise_uv)
    json_data["synthetic_tep_frequency_jitter_hz"]=float(
        frequency_jitter_hz
    )
    json_data["synthetic_tep_amplitude_jitter"]=float(
        amplitude_jitter
    )
    json_data["synthetic_tep_phase_jitter_rad"]=float(
        phase_jitter_rad
    )
    json_data["synthetic_tep_add_aperiodic_noise"]=bool(
        add_aperiodic_noise
    )
    json_data["synthetic_tep_add_tms_artifact"]=bool(
        add_tms_artifact
    )
    json_data["synthetic_tep_tms_artifact_uv"]=float(
        tms_artifact_uv
    )
    json_data["synthetic_tep_seed"]=int(seed)
    json_data["synthetic_tep_seed_channels"]=list(
        seed_channels
    )
    json_data["synthetic_tep_spatial_weights"]={
        channel:float(weight)
        for channel,weight in spatial_weights.items()
    }
    json_data["synthetic_tep_event_frequencies_hz"]=event_frequencies
    json_data["synthetic_tep_event_amplitudes_uv"]=event_amplitudes
    json_data["synthetic_tep_event_phases_rad"]=event_phases
    json_data["synthetic_tep_n_events_injected"]=int(
        len(event_frequencies)
    )
    json_data["synthetic_tep_original_patient_metadata_preserved"]=True
    json_data["synthetic_tep_replaced_channel_type"]="EEG only"

    print("🧪 TEP synthetic ground-truth mode")
    print("   Patient metadata, montage, duration and events preserved")
    print("   EEG time series replaced with synthetic event-locked responses")
    print(f"   Ground-truth Natural Frequency: {frequency_hz:.3f} Hz")
    print(f"   Seed channels: {seed_channels}")
    print(f"   Events injected: {len(event_frequencies)}")
    print(f"   Noise: {noise_uv:.3f} µV")
    print(f"   TMS artifact added: {add_tms_artifact}")

    return raw_out,json_data


def setup_tep_analysis(json_data):
    from pathlib import Path

    subject=str(
        json_data["subject"]
    ).strip().upper()

    hemisphere=str(
        json_data["emispheric_stimulation"]
    ).strip().upper()

    seed_map={
        "SX":["AF3","F3","Fz","FC1"],
        "DX":["Fz","AF4","F4","FC2"]
    }

    if hemisphere not in seed_map:
        raise ValueError(
            "emispheric_stimulation deve essere 'SX' oppure 'DX'"
        )

    json_data["subject"]=subject
    json_data["subject_id"]=subject
    json_data["emispheric_stimulation"]=hemisphere
    json_data["seedChans"]=seed_map[hemisphere]

    main_dir=Path(
        json_data["mainDir"]
    ).expanduser().resolve()

    subject_dir=main_dir/subject

    fileName=str(
        subject_dir
        /f"{subject}EMISFERO{hemisphere}"
    )

    json_data.pop(
        "experiment_dir",
        None
    )

    json_data,experiment_dir,sub=directorySetup(
        json_data
    )

    experiment_dir=Path(
        experiment_dir
    ).expanduser().resolve()

    json_data["analysis_id"]=experiment_dir.name
    json_data["input_file_stem"]=fileName
    json_data["output_directory"]=str(
        experiment_dir
    )

    savePath=str(
        subject_dir
    )

    print("Subject:",sub)
    print("Hemisphere:",hemisphere)
    print("Seed channels:",json_data["seedChans"])
    print("Subject directory:",subject_dir)
    print("Subject directory exists:",subject_dir.exists())
    print("Input stem:",fileName)
    print("Output directory:",experiment_dir)
    print("Analysis ID:",json_data["analysis_id"])

    return (
        json_data,
        experiment_dir,
        sub,
        fileName,
        savePath
    )


def make_json_serializable(value):
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from datetime import datetime,date

    if isinstance(value,dict):
        return {
            str(key):make_json_serializable(item)
            for key,item in value.items()
        }

    if isinstance(value,(list,tuple,set)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    if isinstance(value,np.ndarray):
        return make_json_serializable(
            value.tolist()
        )

    if isinstance(value,np.generic):
        return value.item()

    if isinstance(value,pd.DataFrame):
        return make_json_serializable(
            value.to_dict(
                orient="records"
            )
        )

    if isinstance(value,pd.Series):
        return make_json_serializable(
            value.to_list()
        )

    if isinstance(value,Path):
        return str(value)

    if isinstance(value,(datetime,date)):
        return value.isoformat()

    if value is None:
        return None

    if isinstance(
        value,
        (str,int,float,bool)
    ):
        if isinstance(value,float):
            if np.isnan(value) or np.isinf(value):
                return None

        return value

    return str(value)

def directorySetup(json_data):
    import os
    import json
    from pathlib import Path
    from datetime import datetime

    sub=str(
        json_data["subject"]
    ).strip().upper()

    pipeline_info=load_pipeline_version()
    json_data.update(pipeline_info)

    hemisphere=str(
        json_data.get(
            "emispheric_stimulation",
            json_data.get(
                "hemisphere",
                json_data.get(
                    "stimulation_side",
                    ""
                )
            )
        )
    ).strip().upper()

    hemisphere=(
        hemisphere
        .replace(" ","")
        .replace("_","")
    )

    aliases={
        "DX":"DX",
        "RIGHT":"DX",
        "R":"DX",
        "DESTRA":"DX",
        "SX":"SX",
        "LEFT":"SX",
        "L":"SX",
        "SINISTRA":"SX"
    }

    hemisphere=aliases.get(
        hemisphere,
        hemisphere
    )

    if hemisphere not in ["DX","SX"]:
        raise ValueError(
            "Impostare "
            "json_data['emispheric_stimulation'] "
            "a 'DX' oppure 'SX'."
        )

    timestamp=datetime.now().strftime(
        "%Y%m%d%H%M%S"
    )

    recording_name=f"{sub}{hemisphere}"

    json_data["subject"]=sub
    json_data["subject_id"]=sub
    json_data["emispheric_stimulation"]=hemisphere
    json_data["recording_name"]=recording_name
    json_data["pipeline_timestamp"]=timestamp

    if json_data.get("experiment_dir"):
        experiment_path=Path(
            json_data["experiment_dir"]
        ).expanduser().resolve()

        print(
            "📌 Using provided experiment_dir:",
            experiment_path
        )

    else:
        main_dir=Path(
            json_data["mainDir"]
        ).expanduser().resolve()

        experiment_path=(
            main_dir
            /sub
            /f"{timestamp}_{recording_name}"
        )

        print(
            "📁 Generated TEP experiment_dir:",
            experiment_path
        )

    subdirs=[
        "1.basic",

        "2.trials",
        os.path.join(
            "2.trials",
            "preDetrend"
        ),
        os.path.join(
            "2.trials",
            "postDetrend"
        ),

        "3.detrend",
        os.path.join(
            "3.detrend",
            "examples"
        ),

        "4.postICA",

        "5.Extra",
        os.path.join(
            "5.Extra",
            "FE"
        ),
        os.path.join(
            "5.Extra",
            "FE",
            "PCIst"
        ),
        os.path.join(
            "5.Extra",
            "FE",
            "Fingerprint"
        ),
        os.path.join(
            "5.Extra",
            "FE",
            "NaturalFrequency"
        ),

        "6.FOOOF",
        "7.pkls"
    ]

    experiment_path.mkdir(
        parents=True,
        exist_ok=True
    )

    for subdir in subdirs:
        (
            experiment_path
            /subdir
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    json_data["experiment_dir"]=str(
        experiment_path
    )

    json_data["tep_directory_structure"]=list(
        subdirs
    )

    json_data["basic_dir"]=str(
        experiment_path
        /"1.basic"
    )

    json_data["trials_dir"]=str(
        experiment_path
        /"2.trials"
    )

    json_data["trials_pre_detrend_dir"]=str(
        experiment_path
        /"2.trials"
        /"preDetrend"
    )

    json_data["trials_post_detrend_dir"]=str(
        experiment_path
        /"2.trials"
        /"postDetrend"
    )

    json_data["detrend_dir"]=str(
        experiment_path
        /"3.detrend"
    )

    json_data["detrend_examples_dir"]=str(
        experiment_path
        /"3.detrend"
        /"examples"
    )

    json_data["postICA_dir_base"]=str(
        experiment_path
        /"4.postICA"
    )

    json_data["features_dir"]=str(
        experiment_path
        /"5.Extra"
        /"FE"
    )

    json_data["fooof_dir"]=str(
        experiment_path
        /"6.FOOOF"
    )

    json_data["pkls_dir"]=str(
        experiment_path
        /"7.pkls"
    )

    pars_path=(
        experiment_path
        /f"{sub}_pars.json"
    )

    with open(
        pars_path,
        "w",
        encoding="utf-8"
    ) as json_file:
        json.dump(
            make_json_serializable(
                json_data
            ),
            json_file,
            indent=4,
            sort_keys=True,
            ensure_ascii=False
        )

    print(
        "✅ TEP directory structure created"
    )

    for subdir in subdirs:
        print(
            "   ",
            subdir
        )

    return (
        json_data,
        str(experiment_path),
        sub
    )


def load_pipeline_version(version_file=None):
    from pathlib import Path

    if version_file is None:
        version_file=Path(__file__).resolve().parent/"TMSpathPipeline_versions.txt"
    else:
        version_file=Path(version_file).expanduser().resolve()

    if not version_file.exists():
        raise FileNotFoundError(
            f"File versioni non trovato: {version_file}"
        )

    lines=[
        line.strip()
        for line in version_file.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not lines:
        raise ValueError(
            f"Il file versioni è vuoto: {version_file}"
        )

    fields=[
        field.strip()
        for field in lines[-1].split("|")
    ]

    if len(fields)!=3:
        raise ValueError(
            "L'ultima riga deve avere il formato "
            "YYYY-MM-DD|nome_pipeline|versione"
        )

    release_date,pipeline_name,pipeline_version=fields

    return {
        "pipeline_name":pipeline_name,
        "pipeline_version":pipeline_version,
        "pipeline_release_date":release_date,
        "pipeline_version_file":str(version_file),
        "pipeline_version_record":lines[-1]
    }





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
    print(f"Esperimento ha durata time={time} con sr={lenChannel/time} e samples={lenChannel}")

    

    return df

def computeDetrendSteps(epochs, json_data, experiment_dir, sub, computeFOOOF=True):
    print(f"\n🔍 [{sub}] Step 1: Verifica della necessità di detrending...")
    json_data = check_detrend_need(epochs, json_data, experiment_dir, sub)

    print(f"🧼 [{sub}] Step 2: Esecuzione del pipeline di detrending sui canali selezionati...")
    detrendedEpochs, json_data = run_detrend_pipeline(epochs, json_data, sub, experiment_dir)

    print(f"✅ [{sub}] Detrending completato.\n")
    print(f"\n🔍 [{sub}] Step 3: Compute FOOOF")

    if computeFOOOF:
        df = extract_psd_features(detrendedEpochs, 'postDetrend', experiment_dir, json_data)

    # Salva parametri
    with open(Path(experiment_dir) / f"{sub}_pars.json", 'w') as json_file:
            json.dump(json_data, json_file, indent=4, sort_keys=True)
        
    return detrendedEpochs, json_data


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

 
def computeBasicSteps(raw, events, json_data, experiment_dir, sub, 
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
        print(f"🔧 [{sub}] Step 3: Pulizia epoche e canali artefattati")
        temp_epochs, json_data = clean_trials_channels(
            raw=raw,
            events=events,
            json_data=json_data,
            experiment_dir=experiment_dir,
            sub=sub
        )

    if json_data['do_prepare_epochs']:
        print(f"🔧 [{sub}] Step 4: Creazione oggetto Epochs finale")
        epochs, json_data = prepare_epochs(raw, events, temp_epochs, json_data, experiment_dir, sub)

    else:
        epochs = raw
        temp_epochs = raw
    print(f"✅ [{sub}] Completato. Informazioni finali su Epochs:")
    print(epochs.info)

    if computeFOOOF:
        print(f"🔧 [{sub}] Step 5: FOOOF computation")
        df = extract_psd_features(epochs, 'preDetrend', experiment_dir, json_data)
    
    # Salva parametri
    with open(Path(experiment_dir) / f"{sub}_pars.json", 'w') as json_file:
            json.dump(json_data, json_file, indent=4, sort_keys=True)
        
    if json_data['do_artifact']:
        epochs = add_exp_artifact(epochs,json_data, experiment_dir, sub, 
                                  tau_rise=json_data['do_artifact_rise'], 
                                  tau_decay=json_data['do_artifact_decay'], 
                                  gain=json_data['do_artifact_gain'], 
                                  chans=json_data['do_artifact_chans'])
        basicPlots(epochs, 
                   json_data, experiment_dir, 
                   sub, key='epochs_artifacted', subPath='3.detrend', show=False)


    detrendedEpochs, json_data = computeDetrendSteps(epochs, 
                                            json_data, experiment_dir, sub, 
                                            computeFOOOF=computeFOOOF)

    return raw, epochs, detrendedEpochs, temp_epochs, json_data


def add_intertrial_statistics(events, sfreq, json_data):
    import numpy as np

    event_samples=np.asarray(events[:,0],dtype=int)
    intertrial_times_s=np.diff(event_samples)/float(sfreq)

    json_data["intertrial_n_trials"]=int(len(event_samples))
    json_data["intertrial_n_intervals"]=int(len(intertrial_times_s))

    if intertrial_times_s.size:
        json_data["intertrial_mean_s"]=float(np.mean(intertrial_times_s))
        json_data["intertrial_min_s"]=float(np.min(intertrial_times_s))
        json_data["intertrial_max_s"]=float(np.max(intertrial_times_s))
        json_data["intertrial_median_s"]=float(np.median(intertrial_times_s))
        json_data["intertrial_std_s"]=float(
            np.std(intertrial_times_s,ddof=1)
        ) if intertrial_times_s.size>1 else 0.0
        json_data["intertrial_times_s"]=intertrial_times_s.tolist()
    else:
        json_data["intertrial_mean_s"]=None
        json_data["intertrial_min_s"]=None
        json_data["intertrial_max_s"]=None
        json_data["intertrial_median_s"]=None
        json_data["intertrial_std_s"]=None
        json_data["intertrial_times_s"]=[]

    print("\n⏱️ Inter-trial statistics")
    print(f"   Numero trial: {json_data['intertrial_n_trials']}")
    print(f"   Numero intervalli: {json_data['intertrial_n_intervals']}")

    if intertrial_times_s.size:
        print(
            f"   Inter-trial medio: "
            f"{json_data['intertrial_mean_s']:.3f} s"
        )
        print(
            f"   Inter-trial minimo: "
            f"{json_data['intertrial_min_s']:.3f} s"
        )
        print(
            f"   Inter-trial massimo: "
            f"{json_data['intertrial_max_s']:.3f} s"
        )
        print(
            f"   Inter-trial mediano: "
            f"{json_data['intertrial_median_s']:.3f} s"
        )
        print(
            f"   Inter-trial SD: "
            f"{json_data['intertrial_std_s']:.3f} s"
        )
    else:
        print("   Nessun intervallo disponibile.")

    return json_data

def load_and_prepare_raw_data(fileName, json_data, experiment_dir, sub):
    import os
    import pickle
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import mne
    from tmspath_utils import loadASCII, loadEDF  # definizioni esterne

    json_data, experiment_dir, sub = directorySetup(json_data)

    if 'sourceData' not in json_data:
        raise KeyError("⚠️ 'sourceData' non è definito in json_data.")

    source = json_data['sourceData']
    dataType = json_data['dataType']

    json_data['pulse_artifact_rej_smoothingvalue'] = 0.002
    saveNote = ''
    raw, events = None, None


    def add_intertrial_statistics(events, sfreq):
        event_samples=np.asarray(events[:,0],dtype=int)
        intertrial_times_s=np.diff(event_samples)/float(sfreq)

        json_data["intertrial_n_trials"]=int(len(event_samples))
        json_data["intertrial_n_intervals"]=int(len(intertrial_times_s))
        json_data["intertrial_mean_s"]=float(np.mean(intertrial_times_s)) if intertrial_times_s.size else None
        json_data["intertrial_min_s"]=float(np.min(intertrial_times_s)) if intertrial_times_s.size else None
        json_data["intertrial_max_s"]=float(np.max(intertrial_times_s)) if intertrial_times_s.size else None
        json_data["intertrial_times_s"]=intertrial_times_s.tolist()

        print("\n⏱️ Inter-trial statistics")
        print(f"   Numero trial: {json_data['intertrial_n_trials']}")
        print(f"   Numero intervalli: {json_data['intertrial_n_intervals']}")

        if intertrial_times_s.size:
            print(f"   Inter-trial medio: {json_data['intertrial_mean_s']:.3f} s")
            print(f"   Inter-trial minimo: {json_data['intertrial_min_s']:.3f} s")
            print(f"   Inter-trial massimo: {json_data['intertrial_max_s']:.3f} s")
        else:
            print("   Inter-trial non disponibile: meno di 2 eventi.")

        with open(
            Path(experiment_dir)/f"{sub}_pars.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                make_json_serializable(json_data),
                file,
                indent=4,
                sort_keys=True
            )

    def save_layout_and_metadata(raw, note):
        # Salva layout
        fig = raw.plot_sensors(show_names=True)
        layout_path = Path(experiment_dir) / '1.basic' / f"{sub}_{note}_scalplayout.png"
        fig.savefig(layout_path)
        plt.close(fig)
        # Salva parametri
        with open(Path(experiment_dir) / f"{sub}_pars.json", 'w') as txt_file:
            for key, value in sorted(json_data.items()):
                txt_file.write(f"{key}: {value}\n")   

    # === CASO SIMS ===
    if source == 'SIMS':
        json_data['pulse_artifact_rej_timewindow_min'] = -0.002  # not used in sims
        json_data['pulse_artifact_rej_timewindow_max'] = 0.008  # not used in sims
        #json_data['detrend_minTimeWindowOffset'] = json_data['pulse_artifact_rej_timewindow_max'] # not used in sims

        fileName = f"{json_data['mainDir']}/{json_data['subject']}.fif"
        epochs = mne.read_epochs(fileName, preload=True)
        basicPlots(epochs, json_data, experiment_dir, sub, key='epochsOK', subPath='1.basic')
        with open(Path(experiment_dir) / '7.pkls' / f"{sub}_epochsOK.pkl", 'wb') as f:
            pickle.dump(epochs, f)
        json_data['sfreq'] = epochs.info['sfreq']
        json_data['r_sfreq'] = 512
        raw = epochs
        events = raw.events
        add_intertrial_statistics(events, epochs.info['sfreq'])
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

            if bool(json_data.get("use_synthetic_tep_ground_truth",False)):
                raw,json_data=replace_raw_with_synthetic_tep_ground_truth(
                    raw=raw,
                    events=events,
                    json_data=json_data
                )
            else:
                json_data["data_replaced_with_synthetic_tep"]=False

            saveNote = 'EDF_events'
            json_data['TEP_ID_events'] = saveNote
            json_data['sfreq'] = raw.info['sfreq']
            save_layout_and_metadata(raw, saveNote)
            return raw, events, json_data
    """
    # === CASO MAYER === NEW (30/06/2026) (AS THE OLD!)
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
            # print("TP9:", ch_pos["TP9"])
            # print("TP10:", ch_pos["TP10"])
            montage = mne.channels.make_dig_montage(
                ch_pos=ch_pos,
                coord_frame="head"
            )
            raw.set_montage(montage, on_missing="warn")
        
            raw._data = data[:len(raw.ch_names)]
            json_data['ch_names'] = raw.ch_names
            events, event_id = mne.events_from_annotations(raw, verbose=False)

            if len(events)==0:
                raise ValueError(
                    "Nessun evento trovato nelle annotazioni EDF."
                )

            TMScode = np.unique(events[:, 2])[0]
            events = events[events[:, 2] == TMScode]

            if len(events)==0:
                raise ValueError(
                    f"Nessun evento trovato per TMScode={TMScode}."
                )

            add_intertrial_statistics(events, raw.info['sfreq'])

            if bool(
                json_data.get(
                    "use_synthetic_tep_ground_truth",
                    False
                )
            ):
                print(
                    "🧪 Replacing patient EEG time series "
                    "with synthetic TEP ground truth"
                )

                raw,json_data=replace_raw_with_synthetic_tep_ground_truth(
                    raw=raw,
                    events=events,
                    json_data=json_data
                )

                if not json_data.get(
                    "data_replaced_with_synthetic_tep",
                    False
                ):
                    raise RuntimeError(
                        "La sostituzione sintetica TEP non è stata applicata."
                    )
            else:
                json_data[
                    "data_replaced_with_synthetic_tep"
                ]=False

            saveNote = 'EDF_events'
            json_data['TEP_ID_events'] = saveNote
            json_data['sfreq'] = raw.info['sfreq']
            save_layout_and_metadata(raw, saveNote)
        return raw, events, json_data

    # === CASO CHALFONT ===
    if 'Chalfont' in source:
        json_data['pulse_artifact_rej_timewindow_min'] = -0.002 * 1.5
        json_data['pulse_artifact_rej_timewindow_max'] = 0.008 * 1.5
        #json_data['detrend_minTimeWindowOffset'] = json_data['pulse_artifact_rej_timewindow_max']
        if dataType == 'VHDR':
            raw = mne.io.read_raw_brainvision(f"{fileName}.vhdr", eog=['VEOG', 'HEOG'], preload=True)
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
            add_intertrial_statistics(events, raw.info['sfreq'])
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
            raw = mne.io.read_raw_brainvision(f"{fileName}.vhdr", eog=['VEOG', 'HEOG'], preload=True)
            raw.set_montage('easycap-M1', verbose=True)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            TMScode = 1128
            events = events[events[:, 2] == TMScode]
            saveNote = 'MI_events'
            if Path(fileName).stem == 'prova_Betta_0002':
                shift_sec = - ((0.0001 * 4) + 0.002)
                shift_samples = int(shift_sec * raw.info['sfreq'])
                events[:, 0] += shift_samples
            add_intertrial_statistics(events, raw.info['sfreq'])
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
            add_intertrial_statistics(events, raw.info['sfreq'])
            saveNote = 'EDF_events'
            json_data['TEP_ID_events'] = saveNote
            json_data['sfreq'] = raw.info['sfreq']

            save_layout_and_metadata(raw, saveNote)
            return raw, events, json_data

    raise ValueError(f"⚠️ Origine dati '{source}' non riconosciuta o mal configurata.")

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
            key=f"epochs_continumm_tr{ica_threshold_uv}",
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

def filter_and_plot_raw(raw, json_data, experiment_dir, sub, figsize=(10, 6), subPath='1.basic'):
    print('DO-BROADBAND------------------------------------')

    raw = raw.filter(
        l_freq=json_data['l_freq'],
        h_freq=json_data['broad_band_h_freq'],
        method='iir',
        iir_params=dict(order=3, ftype='butter', phase='zero-double', btype='bandpass'),
        verbose=True
    )

    # === PSD after broadband filtering
    fig = raw.plot_psd(fmin=json_data['l_freq'], fmax=json_data['broad_band_h_freq'], xscale='log', show=False)
    fig.set_size_inches(figsize)
    psd_path = Path(experiment_dir) / subPath / '3.psdrawwithPulseRemovalwithBroadBand_noNotch.png'
    fig.savefig(psd_path)
    plt.close(fig)

    print('DO-NOTCH------------------------------------')
    base_freq = json_data['powerline_freq']
    centers = [base_freq * i for i in range(1, 6)]
    print('DO-NOTCH------------------------------------')
    centers = [json_data['powerline_freq'], json_data['powerline_freq']*2, json_data['powerline_freq']*3, json_data['powerline_freq']*4, json_data['powerline_freq']*5]
    raw = raw.notch_filter(freqs=centers)

    # === PSD after notch filtering
    fig = raw.plot_psd(fmin=json_data['l_freq'], fmax=json_data['broad_band_h_freq'], xscale='log', show=False)
    fig.set_size_inches(figsize)
    psd_path_notch = Path(experiment_dir) / subPath / '4.psdrawwithPulseRemovalwithBroadBandwithNotch.png'
    fig.savefig(psd_path_notch)
    plt.close(fig)

    # === Save raw object
    raw_pkl_path = Path(experiment_dir) / '7.pkls' / f"{sub}_raw.pkl"
    with open(raw_pkl_path, 'wb') as f:
        pickle.dump(raw, f)

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

    n_trials_before = len(temp_epochs)
    all_eeg_chans = list(temp_epochs.ch_names)

    json_data["bad_channels"] = clean_bad_channels_list(
        json_data.get("bad_channels", []),
        all_eeg_chans
    )

    json_data["bad_trials"] = clean_bad_trials_list(
        json_data.get("bad_trials", [])
    )

    data = temp_epochs.get_data()

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

        final_bad_channels = clean_bad_channels_list(auto_bad_channels, all_eeg_chans)
        final_bad_trials = clean_bad_trials_list(auto_bad_trials)

        json_data["bad_channels"] = final_bad_channels
        json_data["bad_trials"] = final_bad_trials

        if len(final_bad_trials) > 0:
            print(f"🧹 Dropping automatic bad trials: {final_bad_trials}")
            keep_idx = np.where(~np.isin(temp_epochs.selection, final_bad_trials))[0]
            temp_epochs = temp_epochs[keep_idx]

        if len(final_bad_channels) > 0:
            print(f"📌 Marking automatic bad channels, not dropping: {final_bad_channels}")
            temp_epochs.info["bads"] = final_bad_channels
        else:
            temp_epochs.info["bads"] = []

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
            scalings={"eeg": 50e-6}
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
            print(f"📌 Marking bad channels, not dropping: {final_bad_channels}")
            temp_epochs.info["bads"] = final_bad_channels
        else:
            temp_epochs.info["bads"] = []

    json_data["bad_trials"] = clean_bad_trials_list(json_data.get("bad_trials", []))
    json_data["bad_channels"] = clean_bad_channels_list(
        json_data.get("bad_channels", []),
        all_eeg_chans
    )

    temp_epochs.info["bads"] = json_data["bad_channels"]

    json_data["trials_tot"] = int(n_trials_before)
    json_data["trials_selected"] = int(len(temp_epochs))
    json_data["channels_tot"] = int(len(all_eeg_chans))
    json_data["channels_dropped"] = []
    json_data["channels_marked_bad"] = list(json_data["bad_channels"])
    json_data["channels_selected"] = int(len(temp_epochs.ch_names))
    json_data["ch_names_after_cleaning"] = list(temp_epochs.ch_names)
    json_data["channel_rejection_policy"] = "mark_bad_not_drop"
    json_data["reference_during_clean_trials_channels"] = "none"

    temp_epochs = temp_epochs.resample(sfreq=json_data["r_sfreq"])
    temp_epochs.info["bads"] = json_data["bad_channels"]

    with open(Path(experiment_dir) / f"{sub}_pars.json", "w") as json_file:
        json.dump(json_data, json_file, indent=4, sort_keys=True)

    print("✅ clean_trials_channels completed")
    print(f"   Trials kept: {json_data['trials_selected']} / {json_data['trials_tot']}")
    print(f"   Channels kept in object: {json_data['channels_selected']} / {json_data['channels_tot']}")
    print(f"   Marked bad channels: {json_data['bad_channels']}")
    print("   Dropped channels: none")

    return temp_epochs, json_data

def clean_trials_channels_20072026(raw, events, json_data, experiment_dir, sub, seedChans=None):
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
    temp_epochs = temp_epochs.set_eeg_reference("average")

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
    temp_epochs = temp_epochs.set_eeg_reference("average")

    with open(Path(experiment_dir) / f"{sub}_pars.json", "w") as json_file:
        json.dump(json_data, json_file, indent=4, sort_keys=True)

    print("✅ clean_trials_channels completed")
    print(f"   Trials kept: {json_data['trials_selected']} / {json_data['trials_tot']}")
    print(f"   Channels kept: {json_data['channels_selected']} / {json_data['channels_tot']}")
    print(f"   Dropped channels: {json_data['bad_channels']}")

    return temp_epochs, json_data

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
    with open(Path(experiment_dir) / f"{sub}_pars.json", 'w') as json_file:
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
    with open(Path(experiment_dir) / f"{sub}_pars.json", 'w') as json_file:
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

        json_data[f"detrend_{json_data['detrend_type']}_pars"] = [
            json_data['detrend_typeOffsetRise'],
            json_data['detrend_typeOffsetDecay']
        ]
        json_data['detrend_MSE'] = mse_detrend

        df_slopes_detrended = computeSlopes_v4(epochs_detrended, json_data, experiment_dir, sub)
        computeSlopesPlot(
            df_slopes_detrended,
            json_data, experiment_dir, sub,
            saveNote=f"ALL-DET_fit{json_data['detrend_fitConstraint']}",
            subPath='3.detrend',
            sharex=True
        )
        detrendedEpochs = epochs_detrended
        if json_data['sourceData']!='SIMS':
            # detrendedEpochs, json_data = notch_filter_offset_chans(detrendedEpochs, json_data) # ulteriore notch solo sui offset chans inutile
            post_label = f"fit{json_data['detrend_fitConstraint']}"
            basicPlots(detrendedEpochs, json_data, experiment_dir, sub, key=f"{post_label}", subPath='3.detrend', show=False)
        else:
            post_label = f"fit{json_data['detrend_fitConstraint']}"
            basicPlots(detrendedEpochs, json_data, experiment_dir, sub, key=f"{post_label}", subPath='3.detrend', show=False)

        
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
            basicPlots(detrendedEpochs, json_data, experiment_dir, sub, key=f"overallPolyOrder{order}", subPath='3.detrend', show=False)

    # === Salvataggio ===
    pkl_path = Path(experiment_dir) / '7.pkls' / f"{sub}_detrendedEpochs.pkl"
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
        print(f"📌 Marking bad channels in final epochs, not dropping: {bad_channels}")
        # epochs.drop_channels(bad_channels)
        epochs.info["bads"] = bad_channels
    
    epochs = epochs.resample(json_data["r_sfreq"])
    epochs = epochs.set_eeg_reference("average")

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
    df.to_csv(Path(experiment_dir) / '3.detrend' / 'offsetTimes_df.csv', index=False)

    mean_offset = df['toffsetmax'].mean()
    std_offset = df['toffsetmax'].std()
    hist_values, bin_edges = np.histogram(df['toffsetmax'], bins=15)
    mode_offset = bin_edges[np.argmax(hist_values)] + np.diff(bin_edges)[0]/2

    json_data['detrend_modeTimeWindowOffset'] = round(mode_offset, 3)
    json_data['detrend_meanTimeWindowOffset'] = round(mean_offset, 4)
    json_data['detrend_stdTimeWindowOffset'] = round(std_offset, 4)

    # Salva parametri
    with open(Path(experiment_dir) / f"{sub}_pars.json", 'w') as json_file:
            json.dump(json_data, json_file, indent=4)


    if do_plot_variability:
        for name in tqdm(epochs.ch_names):
            plotTrialTepVariability(epochs, json_data, experiment_dir, sub, chanNAME=name, operator=np.mean, save=True, parDir='preDetrend')
    
    # see results in 2.trials
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
        ['All Channels', f"offsetChans Channels: {json_data['offsetChans']}"],
        [df['chan'].isin(df['chan']), df['chan'].isin(json_data['offsetChans'])]
    ):
        hist_values, bin_edges = np.histogram(df[mask]['toffsetmax'], bins=15)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, hist_values, width=np.diff(bin_edges), alpha=0.3, edgecolor='black')
        ax.axvline(json_data['detrend_modeTimeWindowOffset'], color='purple', linestyle='-.', linewidth=3, label=f"Mode {json_data['detrend_modeTimeWindowOffset']:.4f}")
        ax.axvline(json_data['detrend_minTimeWindowOffset'], color='red', linestyle='--', linewidth=3, label='Min Detrend')
        ax.axvline(json_data['detrend_maxTimeWindowOffset'], color='red', linestyle='--', linewidth=3, label='Max Detrend')
        ax.axvspan(json_data['pulse_artifact_rej_timewindow_min'],
                   json_data['pulse_artifact_rej_timewindow_max'],
                   color='k', alpha=0.3, label='Pulse Artifact Window')
        ax.set_title(f"Histogram of toffsetmax — {title}", fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency')
        #ax.set_xlim(json_data['detrend_minTimeWindowOffset'], json_data['detrend_maxTimeWindowOffset'])
        ax.grid(True)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.tight_layout()
    plot_path = Path(experiment_dir) / '3.detrend' / 'histogram_toffsetmax_subplots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n📌 [{sub}] Risultati:")
    print(f"   • Zslope threshold = {threshold}")
    print(f"   • Canali oltre soglia ({len(json_data['offsetChans'])}): {json_data['offsetChans']}")
    print(f"   • do_detrend = {json_data['do_detrend']}\n")

    # Salva parametri
    with open(Path(experiment_dir) / f"{sub}_pars.json", 'w') as json_file:
            json.dump(json_data, json_file, indent=4, sort_keys=True)
        
    # === Salvataggio CSV dei risultati ===
    detrend_dir = Path(experiment_dir) / '3.detrend'
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

def plot_pcist_transitions(
    result,
    pars,
    out_dir,
    sub,
    max_components=6,
    dpi=300
):
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_dir=Path(out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    dnst=np.asarray(
        result.get(
            "dNST",
            []
        ),
        dtype=float
    )

    signal_svd=np.asarray(
        result.get(
            "signal_svd",
            []
        ),
        dtype=float
    )

    times=np.asarray(
        result.get(
            "times",
            []
        ),
        dtype=float
    )

    d_base=np.asarray(
        result.get(
            "D_base",
            []
        ),
        dtype=float
    )

    d_resp=np.asarray(
        result.get(
            "D_resp",
            []
        ),
        dtype=float
    )

    t_base=np.asarray(
        result.get(
            "T_base",
            []
        ),
        dtype=float
    )

    t_resp=np.asarray(
        result.get(
            "T_resp",
            []
        ),
        dtype=float
    )

    thresholds=np.asarray(
        result.get(
            "thresholds",
            []
        ),
        dtype=float
    )

    nst_base=np.asarray(
        result.get(
            "NST_base",
            []
        ),
        dtype=float
    )

    nst_resp=np.asarray(
        result.get(
            "NST_resp",
            []
        ),
        dtype=float
    )

    nst_diff=np.asarray(
        result.get(
            "NST_diff",
            []
        ),
        dtype=float
    )

    max_thresholds=np.asarray(
        result.get(
            "max_thresholds",
            []
        ),
        dtype=float
    )

    if dnst.size==0:
        print(
            "⚠️ Nessuna componente PCIst da plottare."
        )
        return {}

    component_order=np.argsort(
        dnst
    )[::-1]

    component_order=component_order[
        :min(
            max_components,
            len(component_order)
        )
    ]

    saved={}

    if (
        signal_svd.ndim==2
        and times.size==signal_svd.shape[1]
    ):
        fig,axes=plt.subplots(
            len(component_order),
            1,
            figsize=(
                12,
                max(
                    3,
                    2.5*len(component_order)
                )
            ),
            sharex=True,
            squeeze=False
        )

        for row,component in enumerate(
            component_order
        ):
            axis=axes[row,0]

            axis.plot(
                times,
                signal_svd[component],
                linewidth=1.5
            )

            axis.axvspan(
                pars["baseline_window"][0],
                pars["baseline_window"][1],
                alpha=0.15,
                label="Baseline"
            )

            axis.axvspan(
                pars["response_window"][0],
                pars["response_window"][1],
                alpha=0.15,
                label="Response"
            )

            axis.axvline(
                0,
                linestyle="--",
                linewidth=1
            )

            axis.set_ylabel(
                f"PC {component+1}"
            )

            axis.set_title(
                f"ΔNST={dnst[component]:.3f}"
            )

        axes[-1,0].set_xlabel(
            "Time [ms]"
        )

        axes[0,0].legend(
            loc="upper right"
        )

        fig.suptitle(
            f"{sub} PCIst retained SVD components"
        )

        fig.tight_layout()

        path=(
            out_dir
            /f"{sub}_PCIst_SVD_components.png"
        )

        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight"
        )

        plt.close(fig)

        saved[
            "svd_components"
        ]=str(path)

    if (
        d_base.ndim==3
        and d_resp.ndim==3
    ):
        fig,axes=plt.subplots(
            len(component_order),
            2,
            figsize=(
                10,
                max(
                    4,
                    4*len(component_order)
                )
            ),
            squeeze=False
        )

        for row,component in enumerate(
            component_order
        ):
            vmax=max(
                float(
                    np.nanmax(
                        d_base[component]
                    )
                ),
                float(
                    np.nanmax(
                        d_resp[component]
                    )
                )
            )

            if not np.isfinite(vmax) or vmax<=0:
                vmax=1.0

            axes[row,0].imshow(
                d_base[component],
                origin="lower",
                aspect="auto",
                vmin=0,
                vmax=vmax
            )

            image=axes[row,1].imshow(
                d_resp[component],
                origin="lower",
                aspect="auto",
                vmin=0,
                vmax=vmax
            )

            axes[row,0].set_title(
                f"PC {component+1} baseline distance"
            )

            axes[row,1].set_title(
                f"PC {component+1} response distance"
            )

            axes[row,0].set_ylabel(
                "Time sample"
            )

            axes[row,1].set_ylabel(
                "Time sample"
            )

            fig.colorbar(
                image,
                ax=axes[row,:],
                shrink=0.8,
                label="State-space distance"
            )

        for axis in axes[-1,:]:
            axis.set_xlabel(
                "Time sample"
            )

        fig.suptitle(
            f"{sub} PCIst distance matrices"
        )

        fig.tight_layout()

        path=(
            out_dir
            /f"{sub}_PCIst_distance_matrices.png"
        )

        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight"
        )

        plt.close(fig)

        saved[
            "distance_matrices"
        ]=str(path)

    if (
        t_base.ndim==3
        and t_resp.ndim==3
    ):
        fig,axes=plt.subplots(
            len(component_order),
            2,
            figsize=(
                10,
                max(
                    4,
                    4*len(component_order)
                )
            ),
            squeeze=False
        )

        for row,component in enumerate(
            component_order
        ):
            axes[row,0].imshow(
                t_base[component],
                origin="lower",
                aspect="auto",
                vmin=0,
                vmax=1,
                interpolation="nearest"
            )

            axes[row,1].imshow(
                t_resp[component],
                origin="lower",
                aspect="auto",
                vmin=0,
                vmax=1,
                interpolation="nearest"
            )

            axes[row,0].set_title(
                f"PC {component+1} baseline transitions"
            )

            axes[row,1].set_title(
                f"PC {component+1} response transitions"
            )

            axes[row,0].set_ylabel(
                "Time sample"
            )

            axes[row,1].set_ylabel(
                "Time sample"
            )

        for axis in axes[-1,:]:
            axis.set_xlabel(
                "Time sample"
            )

        fig.suptitle(
            f"{sub} PCIst optimal transition matrices"
        )

        fig.tight_layout()

        path=(
            out_dir
            /f"{sub}_PCIst_transition_matrices.png"
        )

        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight"
        )

        plt.close(fig)

        saved[
            "transition_matrices"
        ]=str(path)

    if (
        thresholds.ndim==2
        and nst_diff.ndim==2
    ):
        fig,axes=plt.subplots(
            len(component_order),
            1,
            figsize=(
                10,
                max(
                    3,
                    2.8*len(component_order)
                )
            ),
            squeeze=False
        )

        for row,component in enumerate(
            component_order
        ):
            axis=axes[row,0]

            axis.plot(
                thresholds[:,component],
                nst_diff[:,component],
                linewidth=2,
                label="NST response − k·NST baseline"
            )

            if (
                nst_base.ndim==2
                and nst_resp.ndim==2
            ):
                axis.plot(
                    thresholds[:,component],
                    nst_resp[:,component],
                    linestyle="--",
                    label="NST response"
                )

                axis.plot(
                    thresholds[:,component],
                    pars["k"]
                    *nst_base[:,component],
                    linestyle=":",
                    label="k·NST baseline"
                )

            if component<max_thresholds.size:
                axis.axvline(
                    max_thresholds[component],
                    linestyle="--",
                    linewidth=1.5,
                    label=(
                        "Optimal threshold="
                        f"{max_thresholds[component]:.3g}"
                    )
                )

            axis.axhline(
                0,
                linewidth=1
            )

            axis.set_ylabel(
                f"PC {component+1}"
            )

            axis.set_title(
                f"ΔNST={dnst[component]:.3f}"
            )

            axis.legend(
                loc="best",
                fontsize=8
            )

        axes[-1,0].set_xlabel(
            "State-distance threshold"
        )

        fig.suptitle(
            f"{sub} PCIst threshold optimization"
        )

        fig.tight_layout()

        path=(
            out_dir
            /f"{sub}_PCIst_NST_thresholds.png"
        )

        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight"
        )

        plt.close(fig)

        saved[
            "nst_thresholds"
        ]=str(path)

    if bool(
        pars.get(
            "embed",
            False
        )
    ):
        embedding_dimension=int(
            pars.get(
                "L",
                0
            )
        )

        embedding_delay=int(
            pars.get(
                "tau",
                0
            )
        )

        if (
            signal_svd.ndim==2
            and embedding_dimension>=2
            and embedding_delay>=1
        ):
            component=int(
                component_order[0]
            )

            x=signal_svd[
                component
            ]

            cut=(
                embedding_dimension-1
            )*embedding_delay

            if len(x)>cut:
                embedded=np.vstack([
                    x[
                        cut-delay:
                        len(x)-delay
                    ]
                    for delay in range(
                        0,
                        embedding_dimension
                        *embedding_delay,
                        embedding_delay
                    )
                ])

                fig,axis=plt.subplots(
                    figsize=(7,7)
                )

                axis.plot(
                    embedded[0],
                    embedded[1],
                    linewidth=1
                )

                axis.scatter(
                    embedded[0,0],
                    embedded[1,0],
                    s=60,
                    label="Start"
                )

                axis.scatter(
                    embedded[0,-1],
                    embedded[1,-1],
                    s=60,
                    label="End"
                )

                axis.set_xlabel(
                    "x(t)"
                )

                axis.set_ylabel(
                    f"x(t−{embedding_delay} samples)"
                )

                axis.set_title(
                    f"{sub} PCIst state-space trajectory\n"
                    f"PC {component+1}, "
                    f"L={embedding_dimension}, "
                    f"τ={embedding_delay}"
                )

                axis.legend()
                axis.grid(False)
                fig.tight_layout()

                path=(
                    out_dir
                    /f"{sub}_PCIst_state_space.png"
                )

                fig.savefig(
                    path,
                    dpi=dpi,
                    bbox_inches="tight"
                )

                plt.close(fig)

                saved[
                    "state_space"
                ]=str(path)

    return saved




def compute_pcist_baseline_sweep(
    pci_st,
    signal,
    times_ms,
    base_pars,
    out_dir,
    sub,
    json_data
):
    from pathlib import Path
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if not bool(json_data.get("pcist_baseline_sweep",True)):
        return None,None

    out_dir=Path(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)

    configured_start_ms=float(base_pars["baseline_window"][0])
    configured_end_ms=float(base_pars["baseline_window"][1])

    sweep_start_ms=float(
        json_data.get(
            "pcist_baseline_sweep_start_ms",
            configured_start_ms
        )
    )

    sweep_end_ms=float(
        json_data.get(
            "pcist_baseline_sweep_end_ms",
            configured_end_ms
        )
    )

    step_ms=float(
        json_data.get(
            "pcist_baseline_sweep_step_ms",
            10.0
        )
    )

    minimum_duration_ms=float(
        json_data.get(
            "pcist_baseline_sweep_min_duration_ms",
            50.0
        )
    )

    if step_ms<=0:
        raise ValueError(
            "pcist_baseline_sweep_step_ms deve essere > 0."
        )

    if minimum_duration_ms<=0:
        raise ValueError(
            "pcist_baseline_sweep_min_duration_ms deve essere > 0."
        )

    data_min_ms=float(np.min(times_ms))
    data_max_ms=float(np.max(times_ms))

    sweep_start_ms=max(sweep_start_ms,data_min_ms)
    sweep_end_ms=min(sweep_end_ms,data_max_ms)

    latest_start_ms=sweep_end_ms-minimum_duration_ms

    if sweep_start_ms>=latest_start_ms:
        raise ValueError(
            "Intervallo insufficiente per il baseline sweep: "
            f"start={sweep_start_ms:.3f} ms, "
            f"end={sweep_end_ms:.3f} ms, "
            f"minimum_duration={minimum_duration_ms:.3f} ms."
        )

    baseline_starts=np.arange(
        latest_start_ms,
        sweep_start_ms-step_ms*0.5,
        -step_ms,
        dtype=float
    )

    baseline_starts=np.sort(
        np.unique(
            np.append(
                baseline_starts,
                sweep_start_ms
            )
        )
    )[::-1]

    rows=[]

    print("\n🔁 PCIst baseline sweep")
    print(f"   Fixed baseline end: {sweep_end_ms:.1f} ms")
    print(
        f"   Start range: {baseline_starts.min():.1f}–"
        f"{baseline_starts.max():.1f} ms"
    )
    print(f"   Step: {step_ms:.1f} ms")

    for baseline_start_ms in baseline_starts:
        duration_ms=float(
            sweep_end_ms-baseline_start_ms
        )

        sweep_pars=dict(base_pars)
        sweep_pars["baseline_window"]=(
            float(baseline_start_ms),
            float(sweep_end_ms)
        )

        try:
            sweep_result=pci_st.calc_PCIst(
                signal,
                times_ms,
                full_return=False,
                **sweep_pars
            )

            if isinstance(sweep_result,dict):
                pci_sweep=float(sweep_result["PCI"])
            elif isinstance(sweep_result,(tuple,list)):
                pci_sweep=float(sweep_result[0])
            else:
                pci_sweep=float(sweep_result)

            status="ok"

            print(
                f"   [{baseline_start_ms:.1f},"
                f"{sweep_end_ms:.1f}] ms | "
                f"duration={duration_ms:.1f} ms | "
                f"PCIst={pci_sweep:.4f}"
            )

        except Exception as error:
            pci_sweep=np.nan
            status=repr(error)

            print(
                f"⚠️ Sweep baseline "
                f"[{baseline_start_ms:.1f},"
                f"{sweep_end_ms:.1f}] ms: {error}"
            )

        rows.append({
            "subject":str(sub),
            "baseline_start_ms":float(baseline_start_ms),
            "baseline_end_ms":float(sweep_end_ms),
            "baseline_duration_ms":duration_ms,
            "PCIst":(
                float(pci_sweep)
                if np.isfinite(pci_sweep)
                else np.nan
            ),
            "status":status
        })

    df_sweep=pd.DataFrame(rows)

    valid_df=(
        df_sweep[
            np.isfinite(df_sweep["PCIst"])
        ]
        .copy()
        .sort_values("baseline_start_ms")
    )

    if valid_df.empty:
        raise RuntimeError(
            "Nessun valore PCIst valido nel baseline sweep."
        )

    csv_path=out_dir/f"{sub}_PCIst_baseline_sweep.csv"

    df_sweep.to_csv(
        csv_path,
        index=False
    )

    curve_path=(
        out_dir
        /f"{sub}_PCIst_baseline_start_vs_PCIst.png"
    )

    fig,ax=plt.subplots(figsize=(9,6))

    ax.plot(
        valid_df["baseline_start_ms"],
        valid_df["PCIst"],
        marker="o",
        linewidth=2
    )

    ax.set_xlabel("Baseline start [ms]")
    ax.set_ylabel("PCIst")
    ax.set_title(
        f"{sub} PCIst versus baseline start\n"
        f"fixed baseline end = {sweep_end_ms:.1f} ms"
    )
    ax.grid(True,alpha=0.3)
    fig.tight_layout()

    fig.savefig(
        curve_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    distribution_path=(
        out_dir
        /f"{sub}_PCIst_baseline_sweep_distribution.png"
    )

    pci_values=valid_df["PCIst"].to_numpy(dtype=float)

    n_bins=min(
        20,
        max(
            5,
            int(np.ceil(np.sqrt(len(pci_values))))
        )
    )

    pci_mean=float(np.mean(pci_values))
    pci_median=float(np.median(pci_values))
    pci_std=(
        float(np.std(pci_values,ddof=1))
        if len(pci_values)>1
        else 0.0
    )

    fig,ax=plt.subplots(figsize=(8,6))

    ax.hist(
        pci_values,
        bins=n_bins,
        edgecolor="black",
        alpha=0.75
    )

    ax.axvline(
        pci_mean,
        linestyle="--",
        linewidth=2,
        label=f"Mean={pci_mean:.3f}"
    )

    ax.axvline(
        pci_median,
        linestyle=":",
        linewidth=2,
        label=f"Median={pci_median:.3f}"
    )

    ax.set_xlabel("PCIst")
    ax.set_ylabel("Baseline windows")
    ax.set_title(
        f"{sub} PCIst baseline-sweep distribution\n"
        f"mean ± SD = {pci_mean:.3f} ± {pci_std:.3f}"
    )
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        distribution_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    summary={
        "enabled":True,
        "fixed_baseline_end_ms":float(sweep_end_ms),
        "maximum_baseline_start_ms":float(sweep_start_ms),
        "minimum_baseline_duration_ms":float(minimum_duration_ms),
        "step_ms":float(step_ms),
        "n_windows_requested":int(len(df_sweep)),
        "n_windows_valid":int(len(valid_df)),
        "PCIst_mean":pci_mean,
        "PCIst_median":pci_median,
        "PCIst_std":pci_std,
        "PCIst_min":float(np.min(pci_values)),
        "PCIst_max":float(np.max(pci_values)),
        "PCIst_range":float(np.ptp(pci_values)),
        "PCIst_coefficient_of_variation":(
            float(pci_std/abs(pci_mean))
            if pci_mean!=0
            else None
        ),
        "csv":str(csv_path),
        "curve":str(curve_path),
        "distribution":str(distribution_path)
    }

    summary_path=(
        out_dir
        /f"{sub}_PCIst_baseline_sweep_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    summary["summary_json"]=str(summary_path)

    print("✅ PCIst baseline sweep completed")
    print(
        f"   Valid windows: "
        f"{len(valid_df)}/{len(df_sweep)}"
    )
    print(
        f"   PCIst mean ± SD: "
        f"{pci_mean:.3f} ± {pci_std:.3f}"
    )

    return df_sweep,summary


def compute_pcist(postICA_final,json_data,experiment_dir,sub):
    from pathlib import Path
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    pcist_reference={
        "software":{
            "name":"PCIst",
            "repository":"https://github.com/renzocom/PCIst",
            "license":"GPL-3.0"
        },
        "paper":{
            "citation":(
                "Comolatti R et al. A fast and general method to empirically "
                "estimate the complexity of brain responses to transcranial "
                "and intracranial stimulations. Brain Stimulation. "
                "2019;12(5):1280-1289."
            ),
            "title":(
                "A fast and general method to empirically estimate the complexity "
                "of brain responses to transcranial and intracranial stimulations"
            ),
            "journal":"Brain Stimulation",
            "year":2019,
            "volume":12,
            "issue":5,
            "pages":"1280-1289",
            "doi":"10.1016/j.brs.2019.05.013",
            "article_url":(
                "https://www.sciencedirect.com/science/article/"
                "pii/S1935861X19302207"
            )
        }
    }

    try:
        from PCIst import pci_st
    except ImportError as exc:
        raise ImportError(
            "PCIst non installato. Eseguire: pip install PCIst"
        ) from exc

    out_dir=(
        Path(experiment_dir)
        /"5.Extra"
        /"FE"
        /"PCIst"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    evoked=postICA_final.average()

    signal=np.asarray(
        evoked.get_data(),
        dtype=float
    )

    times_ms=np.asarray(
        evoked.times,
        dtype=float
    )*1000.0

    embed=bool(
        json_data.get(
            "pcist_embed",
            False
        )
    )

    pars={
        "baseline_window":tuple(
            json_data.get(
                "pcist_baseline_window_ms",
                (-400,-50)
            )
        ),
        "response_window":tuple(
            json_data.get(
                "pcist_response_window_ms",
                (0,300)
            )
        ),
        "k":float(
            json_data.get(
                "pcist_k",
                1.2
            )
        ),
        "min_snr":float(
            json_data.get(
                "pcist_min_snr",
                1.1
            )
        ),
        "max_var":float(
            json_data.get(
                "pcist_max_var",
                99
            )
        ),
        "embed":embed,
        "n_steps":int(
            json_data.get(
                "pcist_n_steps",
                100
            )
        ),
        "avgref":False,
        "baseline_corr":bool(
            json_data.get(
                "pcist_baseline_corr",
                False
            )
        )
    }

    if embed:
        pars["L"]=int(
            json_data.get(
                "pcist_embedding_dimension",
                3
            )
        )

        pars["tau"]=int(
            json_data.get(
                "pcist_embedding_delay_samples",
                4
            )
        )

        if pars["L"]<2:
            raise ValueError(
                "pcist_embedding_dimension deve essere almeno 2."
            )

        if pars["tau"]<1:
            raise ValueError(
                "pcist_embedding_delay_samples deve essere almeno 1."
            )

    safe_margin_ms=float(
        json_data.get(
            "pcist_safe_margin_ms",
            2.0
        )
    )

    data_min_ms=float(
        times_ms.min()
    )

    data_max_ms=float(
        times_ms.max()
    )

    baseline_start_ms=float(
        pars["baseline_window"][0]
    )

    baseline_end_ms=float(
        pars["baseline_window"][1]
    )

    response_start_ms=float(
        pars["response_window"][0]
    )

    response_end_ms=float(
        pars["response_window"][1]
    )

    if baseline_start_ms<data_min_ms:
        baseline_start_ms=(
            data_min_ms
            +safe_margin_ms
        )

    if baseline_end_ms>data_max_ms:
        baseline_end_ms=(
            data_max_ms
            -safe_margin_ms
        )

    if response_start_ms<data_min_ms:
        response_start_ms=(
            data_min_ms
            +safe_margin_ms
        )

    if response_end_ms>data_max_ms:
        response_end_ms=(
            data_max_ms
            -safe_margin_ms
        )

    pars["baseline_window"]=(
        baseline_start_ms,
        baseline_end_ms
    )

    pars["response_window"]=(
        response_start_ms,
        response_end_ms
    )

    if baseline_start_ms>=baseline_end_ms:
        raise ValueError(
            "Baseline PCIst non valida dopo l'adattamento: "
            f"{pars['baseline_window']} ms."
        )

    if response_start_ms>=response_end_ms:
        raise ValueError(
            "Finestra di risposta PCIst non valida dopo l'adattamento: "
            f"{pars['response_window']} ms."
        )

    print(
        "🕒 PCIst windows | "
        f"baseline={pars['baseline_window']} ms | "
        f"response={pars['response_window']} ms | "
        f"available=({data_min_ms:.3f},{data_max_ms:.3f}) ms"
    )

    print(
        "🧭 PCIst embedding | "
        f"enabled={embed}"
        +(
            f" | L={pars['L']} | tau={pars['tau']} samples"
            if embed
            else ""
        )
    )

    result=pci_st.calc_PCIst(
        signal,
        times_ms,
        full_return=True,
        **pars
    )

    pci_value=float(
        result["PCI"]
    )

    baseline_sweep_df,baseline_sweep_summary=(
        compute_pcist_baseline_sweep(
            pci_st=pci_st,
            signal=signal,
            times_ms=times_ms,
            base_pars=pars,
            out_dir=out_dir,
            sub=sub,
            json_data=json_data
        )
    )

    dnst=np.asarray(
        result.get(
            "dNST",
            []
        ),
        dtype=float
    )

    n_dims=int(
        result.get(
            "n_dims",
            len(dnst)
        )
    )

    dnst_mean=(
        float(
            np.mean(
                dnst
            )
        )
        if dnst.size
        else float("nan")
    )

    def get_array(key):
        value=result.get(
            key,
            None
        )

        if value is None:
            return None

        try:
            array=np.asarray(
                value
            )

            if array.size==0:
                return None

            return array

        except Exception:
            return None

    signal_svd=get_array(
        "signal_svd"
    )

    result_times=get_array(
        "times"
    )

    if result_times is None:
        result_times=times_ms.copy()

    pd.DataFrame({
        "component":np.arange(
            1,
            len(dnst)+1
        ),
        "dNST":dnst
    }).to_csv(
        out_dir/f"{sub}_PCIst_components.csv",
        index=False
    )

    npz_content={
        "PCI":np.asarray(
            pci_value
        ),
        "dNST":dnst,
        "n_dims":np.asarray(
            n_dims
        )
    }

    possible_arrays=[
        "signal_evk",
        "signal_svd",
        "eigenvalues",
        "var_exp",
        "snrs",
        "times",
        "D_base",
        "D_resp",
        "T_base",
        "T_resp",
        "thresholds",
        "NST_base",
        "NST_resp",
        "NST_diff",
        "max_thresholds"
    ]

    for key in possible_arrays:
        value=get_array(
            key
        )

        if value is not None:
            npz_content[
                key
            ]=value

    np.savez_compressed(
        out_dir/f"{sub}_PCIst_full.npz",
        **npz_content
    )

    generated_plots={}

    fig,ax=plt.subplots(
        figsize=(8,5)
    )

    ax.bar(
        np.arange(
            1,
            len(dnst)+1
        ),
        dnst
    )

    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_xlabel(
        "SVD component"
    )

    ax.set_ylabel(
        "ΔNST"
    )

    ax.set_title(
        f"{sub} PCIst = {pci_value:.3f}"
    )

    fig.tight_layout()

    components_path=(
        out_dir
        /f"{sub}_PCIst_components.png"
    )

    fig.savefig(
        components_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    generated_plots[
        "components"
    ]=str(
        components_path
    )

    max_plot_components=int(
        json_data.get(
            "pcist_plot_max_components",
            6
        )
    )

    if dnst.size:
        component_order=np.argsort(
            dnst
        )[::-1]

        component_order=component_order[
            :min(
                max_plot_components,
                len(component_order)
            )
        ]

    else:
        component_order=np.asarray(
            [],
            dtype=int
        )

    if (
        signal_svd is not None
        and signal_svd.ndim==2
        and component_order.size
    ):
        component_times=np.asarray(
            result_times,
            dtype=float
        ).ravel()

        if (
            component_times.size
            !=signal_svd.shape[1]
        ):
            component_times=np.arange(
                signal_svd.shape[1],
                dtype=float
            )

            x_label="Sample"

        else:
            x_label="Time [ms]"

        fig,axes=plt.subplots(
            len(component_order),
            1,
            figsize=(
                12,
                max(
                    3,
                    2.4*len(component_order)
                )
            ),
            sharex=True,
            squeeze=False
        )

        for row,component in enumerate(
            component_order
        ):
            axis=axes[
                row,
                0
            ]

            axis.plot(
                component_times,
                signal_svd[
                    component
                ],
                linewidth=1.5
            )

            if x_label=="Time [ms]":
                axis.axvspan(
                    baseline_start_ms,
                    baseline_end_ms,
                    alpha=0.12,
                    label="Baseline"
                )

                axis.axvspan(
                    response_start_ms,
                    response_end_ms,
                    alpha=0.12,
                    label="Response"
                )

                axis.axvline(
                    0,
                    linestyle="--",
                    linewidth=1
                )

            axis.set_ylabel(
                f"PC {component+1}"
            )

            axis.set_title(
                f"ΔNST={dnst[component]:.3f}"
            )

        axes[-1,0].set_xlabel(
            x_label
        )

        axes[0,0].legend(
            loc="upper right"
        )

        fig.suptitle(
            f"{sub} retained PCIst SVD components"
        )

        fig.tight_layout()

        path=(
            out_dir
            /f"{sub}_PCIst_SVD_components.png"
        )

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        generated_plots[
            "svd_components"
        ]=str(path)

    d_base=get_array(
        "D_base"
    )

    d_resp=get_array(
        "D_resp"
    )

    if (
        d_base is not None
        and d_resp is not None
        and d_base.ndim==3
        and d_resp.ndim==3
        and component_order.size
    ):
        valid_components=[
            int(component)
            for component in component_order
            if (
                component<d_base.shape[0]
                and component<d_resp.shape[0]
            )
        ]

        if valid_components:
            fig,axes=plt.subplots(
                len(valid_components),
                2,
                figsize=(
                    11,
                    max(
                        4,
                        3.8*len(valid_components)
                    )
                ),
                squeeze=False
            )

            for row,component in enumerate(
                valid_components
            ):
                vmax=float(
                    np.nanmax([
                        np.nanmax(
                            d_base[component]
                        ),
                        np.nanmax(
                            d_resp[component]
                        )
                    ])
                )

                if (
                    not np.isfinite(vmax)
                    or vmax<=0
                ):
                    vmax=1.0

                axes[row,0].imshow(
                    d_base[component],
                    origin="lower",
                    aspect="auto",
                    vmin=0,
                    vmax=vmax
                )

                image=axes[row,1].imshow(
                    d_resp[component],
                    origin="lower",
                    aspect="auto",
                    vmin=0,
                    vmax=vmax
                )

                axes[row,0].set_title(
                    f"PC {component+1} baseline distances"
                )

                axes[row,1].set_title(
                    f"PC {component+1} response distances"
                )

                axes[row,0].set_ylabel(
                    "State-time sample"
                )

                axes[row,1].set_ylabel(
                    "State-time sample"
                )

                fig.colorbar(
                    image,
                    ax=axes[row,:],
                    shrink=0.8,
                    label="Distance"
                )

            axes[-1,0].set_xlabel(
                "State-time sample"
            )

            axes[-1,1].set_xlabel(
                "State-time sample"
            )

            fig.suptitle(
                f"{sub} PCIst state-distance matrices"
            )

            fig.tight_layout()

            path=(
                out_dir
                /f"{sub}_PCIst_distance_matrices.png"
            )

            fig.savefig(
                path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

            generated_plots[
                "distance_matrices"
            ]=str(path)

    t_base=get_array(
        "T_base"
    )

    t_resp=get_array(
        "T_resp"
    )

    if (
        t_base is not None
        and t_resp is not None
        and t_base.ndim==3
        and t_resp.ndim==3
        and component_order.size
    ):
        valid_components=[
            int(component)
            for component in component_order
            if (
                component<t_base.shape[0]
                and component<t_resp.shape[0]
            )
        ]

        if valid_components:
            fig,axes=plt.subplots(
                len(valid_components),
                2,
                figsize=(
                    11,
                    max(
                        4,
                        3.8*len(valid_components)
                    )
                ),
                squeeze=False
            )

            for row,component in enumerate(
                valid_components
            ):
                axes[row,0].imshow(
                    t_base[component],
                    origin="lower",
                    aspect="auto",
                    interpolation="nearest",
                    vmin=0,
                    vmax=1
                )

                axes[row,1].imshow(
                    t_resp[component],
                    origin="lower",
                    aspect="auto",
                    interpolation="nearest",
                    vmin=0,
                    vmax=1
                )

                axes[row,0].set_title(
                    f"PC {component+1} baseline transitions"
                )

                axes[row,1].set_title(
                    f"PC {component+1} response transitions"
                )

                axes[row,0].set_ylabel(
                    "State-time sample"
                )

                axes[row,1].set_ylabel(
                    "State-time sample"
                )

            axes[-1,0].set_xlabel(
                "State-time sample"
            )

            axes[-1,1].set_xlabel(
                "State-time sample"
            )

            fig.suptitle(
                f"{sub} PCIst optimal transition matrices"
            )

            fig.tight_layout()

            path=(
                out_dir
                /f"{sub}_PCIst_transition_matrices.png"
            )

            fig.savefig(
                path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

            generated_plots[
                "transition_matrices"
            ]=str(path)

    thresholds=get_array(
        "thresholds"
    )

    nst_base=get_array(
        "NST_base"
    )

    nst_resp=get_array(
        "NST_resp"
    )

    nst_diff=get_array(
        "NST_diff"
    )

    max_thresholds=get_array(
        "max_thresholds"
    )

    if (
        thresholds is not None
        and nst_diff is not None
        and thresholds.ndim==2
        and nst_diff.ndim==2
        and component_order.size
    ):
        valid_components=[
            int(component)
            for component in component_order
            if (
                component<thresholds.shape[1]
                and component<nst_diff.shape[1]
            )
        ]

        if valid_components:
            fig,axes=plt.subplots(
                len(valid_components),
                1,
                figsize=(
                    11,
                    max(
                        3,
                        2.8*len(valid_components)
                    )
                ),
                squeeze=False
            )

            for row,component in enumerate(
                valid_components
            ):
                axis=axes[
                    row,
                    0
                ]

                axis.plot(
                    thresholds[
                        :,
                        component
                    ],
                    nst_diff[
                        :,
                        component
                    ],
                    linewidth=2,
                    label="NST difference"
                )

                if (
                    nst_resp is not None
                    and nst_base is not None
                    and nst_resp.ndim==2
                    and nst_base.ndim==2
                    and component<nst_resp.shape[1]
                    and component<nst_base.shape[1]
                ):
                    axis.plot(
                        thresholds[
                            :,
                            component
                        ],
                        nst_resp[
                            :,
                            component
                        ],
                        linestyle="--",
                        label="NST response"
                    )

                    axis.plot(
                        thresholds[
                            :,
                            component
                        ],
                        pars["k"]
                        *nst_base[
                            :,
                            component
                        ],
                        linestyle=":",
                        label="k × NST baseline"
                    )

                if (
                    max_thresholds is not None
                    and component
                    <max_thresholds.size
                ):
                    axis.axvline(
                        float(
                            max_thresholds[
                                component
                            ]
                        ),
                        linestyle="--",
                        linewidth=1.5,
                        label="Optimal threshold"
                    )

                axis.axhline(
                    0,
                    linewidth=1
                )

                axis.set_ylabel(
                    f"PC {component+1}"
                )

                axis.set_title(
                    f"ΔNST={dnst[component]:.3f}"
                )

                axis.legend(
                    loc="best",
                    fontsize=8
                )

            axes[-1,0].set_xlabel(
                "Distance threshold"
            )

            fig.suptitle(
                f"{sub} PCIst threshold optimization"
            )

            fig.tight_layout()

            path=(
                out_dir
                /f"{sub}_PCIst_NST_thresholds.png"
            )

            fig.savefig(
                path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

            generated_plots[
                "threshold_optimization"
            ]=str(path)

    if (
        embed
        and signal_svd is not None
        and signal_svd.ndim==2
        and component_order.size
    ):
        embedding_dimension=int(
            pars["L"]
        )

        embedding_delay=int(
            pars["tau"]
        )

        component=int(
            component_order[0]
        )

        x=np.asarray(
            signal_svd[
                component
            ],
            dtype=float
        )

        maximum_lag=(
            embedding_dimension-1
        )*embedding_delay

        if len(x)>maximum_lag:
            embedded=np.column_stack([
                x[
                    maximum_lag-lag:
                    len(x)-lag
                ]
                for lag in range(
                    0,
                    embedding_dimension
                    *embedding_delay,
                    embedding_delay
                )
            ])

            embedded_times=np.asarray(
                result_times
            ).ravel()

            if (
                embedded_times.size
                ==len(x)
            ):
                embedded_times=embedded_times[
                    maximum_lag:
                ]
            else:
                embedded_times=np.arange(
                    embedded.shape[0]
                )

            if embedding_dimension>=3:
                fig=plt.figure(
                    figsize=(9,7)
                )

                axis=fig.add_subplot(
                    111,
                    projection="3d"
                )

                scatter=axis.scatter(
                    embedded[:,0],
                    embedded[:,1],
                    embedded[:,2],
                    c=embedded_times,
                    s=12
                )

                axis.plot(
                    embedded[:,0],
                    embedded[:,1],
                    embedded[:,2],
                    linewidth=0.6,
                    alpha=0.6
                )

                axis.set_xlabel(
                    "x(t)"
                )

                axis.set_ylabel(
                    f"x(t-{embedding_delay})"
                )

                axis.set_zlabel(
                    f"x(t-{2*embedding_delay})"
                )

                fig.colorbar(
                    scatter,
                    ax=axis,
                    label="Time [ms]"
                )

            else:
                fig,axis=plt.subplots(
                    figsize=(8,7)
                )

                scatter=axis.scatter(
                    embedded[:,0],
                    embedded[:,1],
                    c=embedded_times,
                    s=14
                )

                axis.plot(
                    embedded[:,0],
                    embedded[:,1],
                    linewidth=0.6,
                    alpha=0.6
                )

                axis.set_xlabel(
                    "x(t)"
                )

                axis.set_ylabel(
                    f"x(t-{embedding_delay})"
                )

                fig.colorbar(
                    scatter,
                    ax=axis,
                    label="Time [ms]"
                )

            axis.set_title(
                f"{sub} PCIst delay embedding\n"
                f"PC {component+1}, "
                f"L={embedding_dimension}, "
                f"τ={embedding_delay} samples"
            )

            fig.tight_layout()

            path=(
                out_dir
                /f"{sub}_PCIst_state_space_embedding.png"
            )

            fig.savefig(
                path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

            generated_plots[
                "state_space_embedding"
            ]=str(path)

    result_keys=sorted(
        [
            str(key)
            for key in result.keys()
        ]
    )

    summary={
        "subject":str(sub),
        "PCI":pci_value,
        "n_dims":n_dims,
        "dNST":dnst.tolist(),
        "dNST_mean":dnst_mean,
        "parameters":pars,
        "embedding_enabled":embed,
        "input_shape":list(
            signal.shape
        ),
        "time_range_ms":[
            data_min_ms,
            data_max_ms
        ],
        "result_keys":result_keys,
        "generated_plots":generated_plots,
        "baseline_sweep":baseline_sweep_summary,
        "reference":pcist_reference
    }

    summary_path=(
        out_dir
        /f"{sub}_PCIst_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    json_data[
        "PCIst"
    ]=pci_value

    json_data[
        "PCIst_n_dims"
    ]=n_dims

    json_data[
        "PCIst_dNST"
    ]=dnst.tolist()

    json_data[
        "PCIst_dNST_mean"
    ]=dnst_mean

    json_data[
        "PCIst_parameters"
    ]=pars

    json_data[
        "PCIst_embedding_used"
    ]=embed

    json_data[
        "PCIst_result_keys"
    ]=result_keys

    json_data[
        "PCIst_generated_plots"
    ]=generated_plots

    json_data[
        "PCIst_baseline_sweep"
    ]=baseline_sweep_summary

    if baseline_sweep_summary is not None:
        json_data[
            "PCIst_baseline_sweep_mean"
        ]=float(
            baseline_sweep_summary["PCIst_mean"]
        )

        json_data[
            "PCIst_baseline_sweep_std"
        ]=float(
            baseline_sweep_summary["PCIst_std"]
        )

        json_data[
            "PCIst_baseline_sweep_range"
        ]=float(
            baseline_sweep_summary["PCIst_range"]
        )

    json_data[
        "PCIst_output_dir"
    ]=str(
        out_dir
    )

    json_data[
        "PCIst_summary_json"
    ]=str(
        summary_path
    )

    json_data[
        "PCIst_full_npz"
    ]=str(
        out_dir
        /f"{sub}_PCIst_full.npz"
    )

    json_data[
        "PCIst_reference"
    ]=pcist_reference

    json_data[
        "PCIst_software_repository"
    ]=pcist_reference[
        "software"
    ][
        "repository"
    ]

    json_data[
        "PCIst_paper_citation"
    ]=pcist_reference[
        "paper"
    ][
        "citation"
    ]

    json_data[
        "PCIst_paper_doi"
    ]=pcist_reference[
        "paper"
    ][
        "doi"
    ]

    json_data[
        "PCIst_paper_url"
    ]=pcist_reference[
        "paper"
    ][
        "article_url"
    ]

    print(
        f"✅ PCIst = {pci_value:.3f} | "
        f"retained dimensions = {n_dims}"
    )

    print(
        "   Result keys:",
        result_keys
    )

    print(
        "   Generated plots:",
        list(
            generated_plots.keys()
        )
    )

    if embed:
        print(
            "   Delay embedding enabled: "
            f"L={pars['L']}, "
            f"tau={pars['tau']} samples"
        )
    else:
        print(
            "   Delay embedding disabled; "
            "transition matrices are still plotted "
            "when returned by PCIst."
        )

    print(
        "📚 PCIst reference: Comolatti R et al., "
        "Brain Stimulation, 2019;12(5):1280-1289. "
        "doi:10.1016/j.brs.2019.05.013"
    )

    print(
        "💻 PCIst software: "
        "https://github.com/renzocom/PCIst"
    )

    return pci_value,result,json_data

def ICAprocessing(
    file,
    json_data,
    experiment_dir,
    sub,
    autoReject=True,
    manualCheck=True,
    computeFOOOF=False
):
    from pathlib import Path
    from datetime import datetime
    import os
    import json
    import pickle

    label_prob_threshold=float(
        json_data.get(
            "do_label_prob_threshold",
            0.80
        )
    )

    threshold_percentile=float(
        json_data.get(
            "do_ica_eigThresh",
            0
        )
    )

    autoReject=bool(
        json_data.get(
            "do_ica_automaticRej",
            True
        )
    )
    
    manualCheck=bool(
        json_data.get(
            "do_ica_manualCheck",
            True
        )
    )
    
    print(
        f"⚙️ ICA eigThresh={threshold_percentile}, "
        f"label_prob_threshold={label_prob_threshold}"
    )

    timestamp=datetime.now().strftime(
        "%Y%m%d%H%M%S"
    )

    postica_dir=(
        Path(experiment_dir)
        /"4.postICA"
        /timestamp
    )

    postica_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    json_data["postICA_dir"]=str(
        postica_dir
    )

    json_data["ICA_timestamp"]=timestamp

    if isinstance(file,str) and file.endswith(".pkl"):
        if not os.path.isfile(file):
            raise FileNotFoundError(
                f"File non trovato: {file}"
            )

        with open(
            file,
            "rb"
        ) as f:
            detrendedEpochs=pickle.load(f)

        print(
            f"[INFO] Oggetto caricato da: {file}"
        )

    else:
        detrendedEpochs=file

        print(
            "[INFO] Oggetto passato direttamente"
        )

    if detrendedEpochs is None:
        raise ValueError(
            "ICAprocessing ha ricevuto un oggetto None."
        )

    if not json_data.get("do_ica",False):
        print(
            "⏭️ ICA disattivata: "
            "restituisco direttamente le epoche detrendate."
        )

        json_data["ICA_applied"]=False
        json_data["ICA_components_tot"]=0
        json_data["ICA_includedComponents_tot"]=0
        json_data["ICA_excludedComponents"]=[]
        json_data["ICA_output_type"]="detrendedEpochs_noICA"

        return (
            detrendedEpochs.copy(),
            None,
            json_data
        )

    postICA_raw,ica_model=run_ica_filtering_v3(
        detrendedEpochs,
        json_data,
        postica_dir,
        sub,
        autoReject=autoReject,
        manualCheck=manualCheck,
        label_prob_threshold=label_prob_threshold,
        threshold_percentile=threshold_percentile
    )

    if computeFOOOF:
        print(
            "Computing ICA-corrected raw FOOOF"
        )

        extract_psd_features(
            postICA_raw,
            "ICA_corrected_raw",
            experiment_dir,
            json_data
        )

    json_data["ICA_applied"]=True

    json_data["ICA_components_tot"]=int(
        ica_model.n_components_
    )

    json_data["ICA_excludedComponents"]=[
        int(component)
        for component in ica_model.exclude
    ]

    json_data["ICA_includedComponents_tot"]=int(
        ica_model.n_components_
        -len(ica_model.exclude)
    )

    json_data["ICA_output_type"]="ICA_corrected_epochs"

    pkl_dir=(
        Path(experiment_dir)
        /"7.pkls"
    )

    pkl_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        pkl_dir
        /f"{timestamp}_{sub}_ICA_corrected_raw.pkl",
        "wb"
    ) as f:
        pickle.dump(
            postICA_raw,
            f
        )

    with open(
        pkl_dir
        /f"{timestamp}_{sub}_ica_model.pkl",
        "wb"
    ) as f:
        pickle.dump(
            ica_model,
            f
        )

    json_data_clean=make_json_serializable(
        json_data
    )

    with open(
        Path(experiment_dir)
        /f"{sub}_pars.json",
        "w",
        encoding="utf-8"
    ) as json_file:
        json.dump(
            json_data_clean,
            json_file,
            indent=4,
            sort_keys=True
        )

    print(
        f"✅ ICA completata | "
        f"tot={ica_model.n_components_} | "
        f"excluded={len(ica_model.exclude)}"
    )

    return (
        postICA_raw,
        ica_model,
        json_data
    )

def finalize_tep_epochs(
    epochs_input,
    json_data,
    experiment_dir,
    sub,
    computeFOOOF=False,
    save_gif=True
):
    from pathlib import Path
    from datetime import datetime
    import json
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt

    if epochs_input is None:
        raise ValueError(
            "finalize_tep_epochs ha ricevuto epochs_input=None."
        )

    timestamp=json_data.get(
        "ICA_timestamp"
    )

    if timestamp is None:
        timestamp=datetime.now().strftime(
            "%Y%m%d%H%M%S"
        )

        json_data["finalization_timestamp"]=timestamp

    final_dir=(
        Path(experiment_dir)
        /"4.postICA"
        /str(timestamp)
    )

    final_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    json_data["finalization_dir"]=str(
        final_dir
    )

    if json_data.get("ICA_applied",False):
        json_data["finalization_input"]="ICA_corrected_epochs"
    else:
        json_data["finalization_input"]="detrendedEpochs_noICA"

    postICA_final=postICAsteps(
        epochs_input,
        json_data,
        experiment_dir,
        sub
    )

    data_tmin=float(
        postICA_final.times.min()
    )

    data_tmax=float(
        postICA_final.times.max()
    )

    plot_tmin=max(
        float(
            json_data.get(
                "epochs_plot_timewindow_min",
                data_tmin
            )
        ),
        data_tmin
    )

    plot_tmax=min(
        float(
            json_data.get(
                "epochs_plot_timewindow_max",
                data_tmax
            )
        ),
        data_tmax
    )

    if plot_tmin>=plot_tmax:
        raise ValueError(
            f"Finestra grafica non valida: "
            f"[{plot_tmin},{plot_tmax}] s; "
            f"dati disponibili "
            f"[{data_tmin},{data_tmax}] s."
        )

    json_data[
        "epochs_plot_timewindow_effective"
    ]=[
        float(plot_tmin),
        float(plot_tmax)
    ]

    print(
        "📊 Final plot window: "
        f"{plot_tmin*1000:.1f}–"
        f"{plot_tmax*1000:.1f} ms"
    )

    basicPlots(
        postICA_final,
        json_data,
        experiment_dir,
        sub,
        key="final",
        subPath=str(
            Path("4.postICA")
            /str(timestamp)
        ),
        show=False
    )

    if save_gif:
        try:
            gif_path,json_data=(
                create_butterfly_topomap_gif(
                    epochs=postICA_final,
                    json_data=json_data,
                    experiment_dir=experiment_dir,
                    sub=sub,
                    saveNote="final",
                    subPath=str(
                        Path("4.postICA")
                        /str(timestamp)
                    ),
                    tmin=plot_tmin,
                    tmax=plot_tmax,
                    step=float(
                        json_data.get(
                            "postICA_gif_step_s",
                            0.002
                        )
                    ),
                    xlim=(
                        plot_tmin,
                        plot_tmax
                    ),
                    vlim=tuple(
                        json_data.get(
                            "postICA_gif_vlim_V",
                            (-5e-6,5e-6)
                        )
                    ),
                    sphere=float(
                        json_data.get(
                            "postICA_gif_sphere_m",
                            0.095
                        )
                    ),
                    duration=int(
                        json_data.get(
                            "postICA_gif_duration_ms",
                            50
                        )
                    ),
                    transparency=bool(
                        json_data.get(
                            "postICA_gif_transparency",
                            False
                        )
                    ),
                    save_static=True
                )
            )

            json_data["final_gif"]=str(
                gif_path
            )

            json_data["final_gif_tmin_s"]=float(
                plot_tmin
            )

            json_data["final_gif_tmax_s"]=float(
                plot_tmax
            )

            json_data.pop(
                "final_gif_error",
                None
            )

            print(
                f"✅ Final butterfly-topomap GIF: "
                f"{gif_path}"
            )

        except Exception as error:
            json_data["final_gif"]=None
            json_data["final_gif_error"]=str(
                error
            )

            print(
                "⚠️ Impossibile generare la GIF:",
                error
            )

    condition_number_evoked=(
        compute_condition_number_epochs_average(
            postICA_final
        )
    )

    json_data["cn_final"]=float(
        condition_number_evoked
    )

    plot_topomap(
        postICA_final,
        json_data,
        experiment_dir,
        sub,
        subDir=str(
            Path("4.postICA")
            /str(timestamp)
        ),
        saveNote="final"
    )

    json_data["feats_smfp"]=plot_gmfp(
        postICA_final,
        json_data,
        experiment_dir,
        sub,
        FEAT=json_data["seedChans"]
    )

    fig,ax=plt.subplots(
        figsize=(10,5)
    )

    ax.plot(
        postICA_final.times*1e3,
        np.mean(
            postICA_final.get_data(),
            axis=0
        ).T*1e6,
        color="k",
        linewidth=0.2
    )

    ax.axvline(
        0,
        color="k",
        linestyle="--",
        linewidth=1
    )

    ax.set_xlabel(
        "Time [ms]"
    )

    ax.set_ylabel(
        "Amplitude [µV]"
    )

    ax.set_xlim(
        plot_tmin*1000,
        plot_tmax*1000
    )

    ax.grid(
        False
    )

    fig.tight_layout()

    fig.savefig(
        final_dir/"tep_final_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    if computeFOOOF:
        print(
            "Computing final FOOOF"
        )

        extract_psd_features(
            postICA_final,
            "final",
            experiment_dir,
            json_data
        )

    pkl_dir=(
        Path(experiment_dir)
        /"7.pkls"
    )

    pkl_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    final_pkl_path=(
        pkl_dir
        /f"{timestamp}_{sub}_final.pkl"
    )

    with open(
        final_pkl_path,
        "wb"
    ) as f:
        pickle.dump(
            postICA_final,
            f
        )

    json_data["final_epochs_pkl"]=str(
        final_pkl_path
    )

    json_data["final_epochs_type"]=(
        "postICA_final"
        if json_data.get(
            "ICA_applied",
            False
        )
        else "final_noICA"
    )

    json_data["finalization_completed"]=True

    json_data_clean=make_json_serializable(
        json_data
    )

    with open(
        Path(experiment_dir)
        /f"{sub}_pars.json",
        "w",
        encoding="utf-8"
    ) as json_file:
        json.dump(
            json_data_clean,
            json_file,
            indent=4,
            sort_keys=True
        )

    print(
        "✅ Finalizzazione TEP completata | "
        f"ICA applied={json_data.get('ICA_applied',False)}"
    )

    return (
        postICA_final,
        json_data
    )


def compute_tep_natural_frequency(
    postICA_final,
    json_data,
    experiment_dir,
    sub,
    save=True
):
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from mne.time_frequency import tfr_array_morlet

    sub=str(sub).strip()

    out_dir=(
        Path(experiment_dir)
        /"5.Extra"
        /"FE"
        /"NaturalFrequency"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    seed_channels=[
        channel
        for channel in json_data.get(
            "seedChans",
            []
        )
        if channel in postICA_final.ch_names
    ]

    if not seed_channels:
        raise ValueError(
            "Nessun seed channel valido per la Natural Frequency."
        )

    epochs=(
        postICA_final
        .copy()
        .pick(seed_channels)
    )

    data=np.asarray(
        epochs.get_data(),
        dtype=float
    )

    times_ms=np.asarray(
        epochs.times,
        dtype=float
    )*1000.0

    sfreq=float(
        epochs.info["sfreq"]
    )

    n_times=int(
        data.shape[-1]
    )

    fmin=float(
        json_data.get(
            "tep_nf_fmin",
            8.0
        )
    )

    fmax=float(
        json_data.get(
            "tep_nf_fmax",
            45.0
        )
    )

    frequency_step_hz=float(
        json_data.get(
            "tep_nf_frequency_step_hz",
            1.0
        )
    )

    morlet_cycles=float(
        json_data.get(
            "tep_nf_n_cycles",
            3.5
        )
    )

    if fmin<=0:
        raise ValueError(
            f"tep_nf_fmin deve essere > 0: {fmin}"
        )

    if fmax<=fmin:
        raise ValueError(
            "Intervallo Natural Frequency non valido: "
            f"fmin={fmin}, fmax={fmax} Hz."
        )

    if frequency_step_hz<=0:
        raise ValueError(
            "tep_nf_frequency_step_hz deve essere > 0."
        )

    if morlet_cycles<=0:
        raise ValueError(
            "tep_nf_n_cycles deve essere > 0."
        )

    nyquist=sfreq/2.0

    effective_fmax=min(
        fmax,
        np.nextafter(
            nyquist,
            0.0
        )
    )

    freqs=np.arange(
        fmin,
        effective_fmax
        +frequency_step_hz*0.5,
        frequency_step_hz,
        dtype=float
    )

    freqs=freqs[
        freqs<nyquist
    ]

    if len(freqs)<2:
        raise ValueError(
            "Intervallo di frequenze insufficiente: "
            f"fmin={fmin}, fmax={fmax}, "
            f"step={frequency_step_hz}, "
            f"Nyquist={nyquist:.3f} Hz."
        )

    n_cycles=np.full(
        freqs.shape,
        morlet_cycles,
        dtype=float
    )

    baseline_window_ms=tuple(
        json_data.get(
            "tep_nf_baseline_window_ms",
            (-300,-50)
        )
    )

    response_window_ms=tuple(
        json_data.get(
            "tep_nf_response_window_ms",
            (20,200)
        )
    )

    if len(baseline_window_ms)!=2:
        raise ValueError(
            "tep_nf_baseline_window_ms deve contenere "
            "esattamente due valori."
        )

    if len(response_window_ms)!=2:
        raise ValueError(
            "tep_nf_response_window_ms deve contenere "
            "esattamente due valori."
        )

    baseline_start_ms=float(
        baseline_window_ms[0]
    )

    baseline_end_ms=float(
        baseline_window_ms[1]
    )

    response_start_ms=float(
        response_window_ms[0]
    )

    response_end_ms=float(
        response_window_ms[1]
    )

    if baseline_start_ms>=baseline_end_ms:
        raise ValueError(
            "Finestra baseline Natural Frequency non valida: "
            f"{baseline_window_ms} ms."
        )

    if response_start_ms>=response_end_ms:
        raise ValueError(
            "Finestra di risposta Natural Frequency non valida: "
            f"{response_window_ms} ms."
        )

    data_min_ms=float(
        times_ms.min()
    )

    data_max_ms=float(
        times_ms.max()
    )

    if (
        baseline_start_ms<data_min_ms
        or baseline_end_ms>data_max_ms
    ):
        raise ValueError(
            "Finestra baseline Natural Frequency non contenuta "
            "nelle epoche: dati "
            f"[{data_min_ms:.3f},{data_max_ms:.3f}] ms, "
            f"richiesta [{baseline_start_ms},"
            f"{baseline_end_ms}] ms."
        )

    if (
        response_start_ms<data_min_ms
        or response_end_ms>data_max_ms
    ):
        raise ValueError(
            "Finestra post-TMS Natural Frequency non contenuta "
            "nelle epoche: dati "
            f"[{data_min_ms:.3f},{data_max_ms:.3f}] ms, "
            f"richiesta [{response_start_ms},"
            f"{response_end_ms}] ms."
        )

    baseline_mask=(
        (times_ms>=baseline_start_ms)
        &(times_ms<=baseline_end_ms)
    )

    response_mask=(
        (times_ms>=response_start_ms)
        &(times_ms<=response_end_ms)
    )

    if np.sum(baseline_mask)<3:
        raise ValueError(
            "La finestra baseline non contiene abbastanza campioni: "
            f"{baseline_window_ms} ms."
        )

    if np.sum(response_mask)<3:
        raise ValueError(
            "La finestra post-TMS non contiene abbastanza campioni: "
            f"{response_window_ms} ms."
        )

    baseline_duration_ms=float(
        baseline_end_ms
        -baseline_start_ms
    )

    minimum_frequency_period_ms=float(
        1000.0/freqs.min()
    )

    baseline_cycles_at_fmin=float(
        baseline_duration_ms
        /minimum_frequency_period_ms
    )

    nominal_wavelet_duration_sec=float(
        morlet_cycles/freqs.min()
    )

    wavelet_duration_sec=float(
        10.0
        *np.max(n_cycles)
        /(2.0*np.pi*np.min(freqs))
    )

    wavelet_samples=int(
        np.ceil(
            wavelet_duration_sec
            *sfreq
        )
    )

    minimum_total_samples=max(
        wavelet_samples+2,
        n_times
    )

    minimum_padding_seconds=float(
        json_data.get(
            "tep_nf_min_padding_s",
            0.25
        )
    )

    pad_samples=max(
        int(
            np.ceil(
                (
                    minimum_total_samples
                    -n_times
                )/2.0
            )
        ),
        int(
            np.ceil(
                minimum_padding_seconds
                *sfreq
            )
        )
    )

    padding_mode=(
        "reflect"
        if n_times>1
        else "edge"
    )

    data_padded=np.pad(
        data,
        (
            (0,0),
            (0,0),
            (
                pad_samples,
                pad_samples
            )
        ),
        mode=padding_mode
    )

    print(
        "🔧 Rosanova Morlet ERSP: "
        f"trials={data.shape[0]}, "
        f"channels={data.shape[1]}, "
        f"frequencies={freqs[0]:.1f}–"
        f"{freqs[-1]:.1f} Hz, "
        f"step={frequency_step_hz:.2f} Hz, "
        f"cycles={morlet_cycles:.2f}"
    )

    print(
        "   Baseline:",
        f"{baseline_start_ms:.1f}–"
        f"{baseline_end_ms:.1f} ms",
        f"| duration={baseline_duration_ms:.1f} ms",
        f"| periods at {freqs.min():.1f} Hz="
        f"{baseline_cycles_at_fmin:.2f}"
    )

    print(
        "   Nominal wavelet duration at "
        f"{freqs.min():.1f} Hz: "
        f"{nominal_wavelet_duration_sec*1000:.1f} ms"
    )

    print(
        "   Signal samples:",
        n_times,
        "| padded samples:",
        data_padded.shape[-1],
        "| padding per side:",
        pad_samples
    )

    power_padded=tfr_array_morlet(
        data_padded,
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        output="power",
        use_fft=True,
        zero_mean=True,
        decim=1,
        n_jobs=1,
        verbose=False
    )

    power=np.asarray(
        power_padded[
            ...,
            pad_samples:pad_samples+n_times
        ],
        dtype=float
    )

    if power.ndim!=4:
        raise RuntimeError(
            f"Shape Morlet inattesa: {power.shape}"
        )

    if power.shape[-1]!=len(times_ms):
        raise RuntimeError(
            "Numero di campioni TFR non coerente con i tempi: "
            f"{power.shape[-1]} vs {len(times_ms)}."
        )

    epsilon=np.finfo(float).eps

    baseline_power=np.mean(
        power[
            ...,
            baseline_mask
        ],
        axis=-1,
        keepdims=True
    )

    baseline_power=np.maximum(
        baseline_power,
        epsilon
    )

    ersp_db=10.0*np.log10(
        np.maximum(
            power,
            epsilon
        )
        /baseline_power
    )

    roi_ersp_db=np.mean(
        ersp_db,
        axis=(0,1)
    )

    response_ersp=roi_ersp_db[
        :,
        response_mask
    ]

    cumulative_ersp=np.sum(
        response_ersp,
        axis=1
    )

    cumulative_positive_ersp=np.sum(
        np.maximum(
            response_ersp,
            0.0
        ),
        axis=1
    )

    if not np.any(
        np.isfinite(
            cumulative_positive_ersp
        )
    ):
        raise RuntimeError(
            "Lo spettro cumulativo Natural Frequency "
            "non contiene valori finiti."
        )

    nf_index=int(
        np.nanargmax(
            cumulative_positive_ersp
        )
    )

    natural_frequency_hz=float(
        freqs[nf_index]
    )

    natural_frequency_score=float(
        cumulative_positive_ersp[
            nf_index
        ]
    )

    response_mean_ersp=np.mean(
        response_ersp,
        axis=1
    )

    boundary_frequency=bool(
        nf_index==0
        or nf_index==len(freqs)-1
    )

    band_definitions={
        "theta":(4.0,7.0),
        "alpha":(8.0,12.0),
        "beta1":(13.0,20.0),
        "beta2":(21.0,29.0),
        "gamma":(30.0,50.0)
    }

    band_power={}

    for (
        band_name,
        (
            band_min,
            band_max
        )
    ) in band_definitions.items():

        band_mask=(
            (freqs>=band_min)
            &(freqs<=band_max)
        )

        if np.any(band_mask):
            band_power[
                band_name
            ]=float(
                np.mean(
                    response_mean_ersp[
                        band_mask
                    ]
                )
            )
        else:
            band_power[
                band_name
            ]=float("nan")

    evoked_roi_uv=(
        epochs
        .average()
        .get_data()
        .mean(axis=0)
        *1e6
    )

    tfr_csv=(
        out_dir
        /f"{sub}_TEP_natural_frequency_morlet_tfr.csv"
    )

    spectrum_csv=(
        out_dir
        /f"{sub}_TEP_natural_frequency_spectrum.csv"
    )

    summary_json=(
        out_dir
        /f"{sub}_TEP_natural_frequency_summary.json"
    )

    rosanova_png=(
        out_dir
        /f"{sub}_TEP_natural_frequency_rosanova.png"
    )

    pd.DataFrame(
        roi_ersp_db,
        index=freqs,
        columns=np.round(
            times_ms,
            3
        )
    ).rename_axis(
        "frequency_hz"
    ).to_csv(
        tfr_csv
    )

    pd.DataFrame({
        "frequency_hz":freqs,
        "mean_response_ersp_db":response_mean_ersp,
        "cumulative_ersp":cumulative_ersp,
        "cumulative_positive_ersp":cumulative_positive_ersp,
        "is_natural_frequency":(
            np.arange(len(freqs))
            ==nf_index
        )
    }).to_csv(
        spectrum_csv,
        index=False
    )

    reference={
        "citation":(
            "Rosanova M, Casali A, Bellina V, Resta F, "
            "Mariotti M, Massimini M. Natural Frequencies "
            "of Human Corticothalamic Circuits. "
            "Journal of Neuroscience. 2009;29:7679-7685."
        ),
        "doi":"10.1523/JNEUROSCI.0445-09.2009"
    }

    summary={
        "subject":sub,
        "subject_id":sub,
        "method":"Rosanova-style Morlet ERSP",
        "natural_frequency_hz":natural_frequency_hz,
        "natural_frequency_score":natural_frequency_score,
        "natural_frequency_at_boundary":boundary_frequency,
        "seed_channels":seed_channels,
        "requested_frequency_range_hz":[
            float(fmin),
            float(fmax)
        ],
        "effective_frequency_range_hz":[
            float(freqs.min()),
            float(freqs.max())
        ],
        "frequencies_hz":freqs.tolist(),
        "frequency_step_hz":float(
            frequency_step_hz
        ),
        "morlet_cycles":float(
            morlet_cycles
        ),
        "nominal_wavelet_duration_at_fmin_ms":float(
            nominal_wavelet_duration_sec
            *1000.0
        ),
        "padding_mode":padding_mode,
        "padding_samples_per_side":int(
            pad_samples
        ),
        "baseline_window_ms":[
            baseline_start_ms,
            baseline_end_ms
        ],
        "baseline_duration_ms":baseline_duration_ms,
        "baseline_periods_at_fmin":baseline_cycles_at_fmin,
        "response_window_ms":[
            response_start_ms,
            response_end_ms
        ],
        "baseline_correction":(
            "10*log10(power/baseline_power)"
        ),
        "nf_selection":(
            "Maximum cumulative positive ERSP "
            "within the response window"
        ),
        "band_power_db":band_power,
        "reference":reference
    }

    with open(
        summary_json,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    if save:
        fig=plt.figure(
            figsize=(13,8),
            constrained_layout=True
        )

        grid=fig.add_gridspec(
            nrows=2,
            ncols=3,
            height_ratios=[
                1.0,
                2.0
            ],
            width_ratios=[
                6.5,
                1.3,
                0.22
            ],
            hspace=0.12,
            wspace=0.08
        )

        ax_wave=fig.add_subplot(
            grid[0,0]
        )

        ax_tf=fig.add_subplot(
            grid[1,0],
            sharex=ax_wave
        )

        ax_spectrum=fig.add_subplot(
            grid[1,1],
            sharey=ax_tf
        )

        ax_cbar=fig.add_subplot(
            grid[1,2]
        )

        time_limits=(
            float(times_ms.min()),
            float(times_ms.max())
        )

        ax_wave.plot(
            times_ms,
            evoked_roi_uv,
            linewidth=2
        )

        ax_wave.axvline(
            0,
            color="k",
            linestyle=":",
            linewidth=2
        )

        ax_wave.axvspan(
            response_start_ms,
            response_end_ms,
            alpha=0.12
        )

        ax_wave.set_xlim(
            time_limits
        )

        ax_wave.set_ylabel(
            "Amplitude [µV]"
        )

        ax_wave.set_title(
            f"{sub} Seed evoked waveform | "
            f"Seed={', '.join(seed_channels)}"
        )

        ax_wave.set_xlabel("")

        ax_wave.tick_params(
            axis="x",
            labelbottom=False
        )

        ax_wave.grid(False)

        vmax=float(
            np.nanpercentile(
                np.abs(
                    roi_ersp_db
                ),
                98
            )
        )

        if (
            not np.isfinite(vmax)
            or vmax<=0
        ):
            vmax=1.0

        vmin=-vmax

        image=ax_tf.pcolormesh(
            times_ms,
            freqs,
            roi_ersp_db,
            shading="gouraud",
            cmap="jet",
            vmin=vmin,
            vmax=vmax
        )

        ax_tf.axvline(
            0,
            color="k",
            linestyle=":",
            linewidth=2
        )

        ax_tf.axhline(
            natural_frequency_hz,
            color="k",
            linestyle=":",
            linewidth=1.5
        )

        ax_tf.text(
            times_ms.max()-5,
            natural_frequency_hz
            +frequency_step_hz*0.8,
            f"{natural_frequency_hz:.1f} Hz",
            ha="right",
            va="bottom",
            fontsize=13,
            fontweight="bold"
        )

        ax_tf.set_xlim(
            time_limits
        )

        ax_tf.set_ylim(
            float(freqs.min()),
            float(freqs.max())
        )

        ax_tf.set_xlabel(
            "Time [ms]"
        )

        ax_tf.set_ylabel(
            "Frequency [Hz]"
        )

        ax_tf.set_title(
            f"{sub} Morlet ERSP"
        )

        ax_tf.grid(False)

        ax_spectrum.plot(
            cumulative_positive_ersp,
            freqs,
            color="black",
            linewidth=2
        )

        ax_spectrum.fill_betweenx(
            freqs,
            0,
            cumulative_positive_ersp,
            color="0.85"
        )

        ax_spectrum.axhline(
            natural_frequency_hz,
            color="k",
            linestyle=":",
            linewidth=1.5
        )

        ax_spectrum.set_xlabel(
            "Cumulative positive ERSP"
        )

        ax_spectrum.tick_params(
            axis="y",
            labelleft=False
        )

        ax_spectrum.grid(False)

        xmax=float(
            np.nanmax(
                cumulative_positive_ersp
            )
        )

        if (
            not np.isfinite(xmax)
            or xmax<=0
        ):
            xmax=1.0

        band_labels=[
            ("θ",4.0,7.0),
            ("α",8.0,12.0),
            ("β1",13.0,20.0),
            ("β2",21.0,29.0),
            ("γ",30.0,50.0)
        ]

        for (
            label,
            band_min,
            band_max
        ) in band_labels:

            visible_min=max(
                band_min,
                float(freqs.min())
            )

            visible_max=min(
                band_max,
                float(freqs.max())
            )

            if visible_min<=visible_max:
                band_center=(
                    visible_min
                    +visible_max
                )/2.0

                ax_spectrum.text(
                    xmax*0.93,
                    band_center,
                    label,
                    ha="right",
                    va="center",
                    fontsize=11
                )

            if (
                band_min>=freqs.min()
                and band_min<=freqs.max()
            ):
                ax_spectrum.axhline(
                    band_min,
                    color="0.7",
                    linewidth=0.6
                )

        colorbar=fig.colorbar(
            image,
            cax=ax_cbar
        )

        colorbar.set_label(
            "ERSP [dB]"
        )

        fig.savefig(
            rosanova_png,
            dpi=300,
            bbox_inches="tight",
            facecolor="white"
        )

        plt.close(fig)

    json_data[
        "TEP_natural_frequency_computed"
    ]=True

    json_data[
        "TEP_natural_frequency_method"
    ]="Rosanova-style Morlet ERSP"

    json_data[
        "TEP_natural_frequency_hz"
    ]=natural_frequency_hz

    json_data[
        "TEP_natural_frequency_score"
    ]=natural_frequency_score

    json_data[
        "TEP_natural_frequency_power_db"
    ]=natural_frequency_score

    json_data[
        "TEP_natural_frequency_at_boundary"
    ]=boundary_frequency

    json_data[
        "TEP_natural_frequency_seed_channels"
    ]=seed_channels

    json_data[
        "TEP_natural_frequency_band_power"
    ]=band_power

    json_data[
        "TEP_natural_frequency_parameters"
    ]={
        "requested_fmin_hz":float(
            fmin
        ),
        "requested_fmax_hz":float(
            fmax
        ),
        "effective_fmin_hz":float(
            freqs.min()
        ),
        "effective_fmax_hz":float(
            freqs.max()
        ),
        "frequency_step_hz":float(
            frequency_step_hz
        ),
        "morlet_cycles":float(
            morlet_cycles
        ),
        "nominal_wavelet_duration_at_fmin_ms":float(
            nominal_wavelet_duration_sec
            *1000.0
        ),
        "padding_mode":padding_mode,
        "padding_samples_per_side":int(
            pad_samples
        ),
        "baseline_window_ms":[
            baseline_start_ms,
            baseline_end_ms
        ],
        "baseline_duration_ms":float(
            baseline_duration_ms
        ),
        "baseline_periods_at_fmin":float(
            baseline_cycles_at_fmin
        ),
        "response_window_ms":[
            response_start_ms,
            response_end_ms
        ]
    }

    json_data[
        "TEP_natural_frequency_output_dir"
    ]=str(out_dir)

    json_data[
        "TEP_natural_frequency_figure"
    ]=str(rosanova_png)

    json_data[
        "TEP_natural_frequency_tfr_csv"
    ]=str(tfr_csv)

    json_data[
        "TEP_natural_frequency_spectrum_csv"
    ]=str(spectrum_csv)

    json_data[
        "TEP_natural_frequency_summary_json"
    ]=str(summary_json)

    json_data[
        "TEP_natural_frequency_reference"
    ]=reference

    json_data[
        "TEP_natural_frequency_primary_DOI"
    ]=reference["doi"]

    results={
        "natural_frequency_hz":natural_frequency_hz,
        "natural_frequency_power_db":natural_frequency_score,
        "natural_frequency_score":natural_frequency_score,
        "natural_frequency_at_boundary":boundary_frequency,
        "band_power_db":band_power,
        "seed_channels":seed_channels,
        "method":"Rosanova-style Morlet ERSP",
        "figure":str(rosanova_png),
        "tfr_csv":str(tfr_csv),
        "spectrum_csv":str(spectrum_csv),
        "summary_json":str(summary_json),
        "output_directory":str(out_dir)
    }

    print(
        "✅ TEP Natural Frequency Rosanova completata"
    )

    print(
        f"   Seed channels: {seed_channels}"
    )

    print(
        f"   Morlet: {morlet_cycles:.2f} cycles, "
        f"{freqs.min():.1f}–"
        f"{freqs.max():.1f} Hz, "
        f"step {frequency_step_hz:.2f} Hz"
    )

    print(
        f"   Baseline: "
        f"{baseline_start_ms:.1f}–"
        f"{baseline_end_ms:.1f} ms "
        f"({baseline_cycles_at_fmin:.2f} periods "
        f"at {freqs.min():.1f} Hz)"
    )

    print(
        f"   Response: "
        f"{response_start_ms:.1f}–"
        f"{response_end_ms:.1f} ms"
    )

    print(
        f"   Padding: {pad_samples} samples "
        f"per side ({padding_mode})"
    )

    print(
        f"   Natural Frequency: "
        f"{natural_frequency_hz:.1f} Hz"
    )

    if boundary_frequency:
        print(
            "⚠️ La Natural Frequency coincide con un limite "
            "dell'intervallo spettrale. Interpretare con cautela."
        )

    print(
        f"   Figure: {rosanova_png}"
    )

    return results,json_data

def extractFeatures(
    postICA_final,
    json_data,
    experiment_dir,
    sub
):
    import json
    import pandas as pd
    from pathlib import Path

    compute_standard=bool(
        json_data.get(
            "do_standard_features",
            True
        )
    )

    compute_pcist_feature=bool(
        json_data.get(
            "do_pcist",
            False
        )
    )

    compute_fooof=bool(
        json_data.get(
            "do_fooof_features",
            False
        )
    )

    compute_tep_fingerprint_feature=bool(
        json_data.get(
            "do_tep_fingerprint",
            False
        )
    )

    compute_tep_natural_frequency_feature=bool(
        json_data.get(
            "do_tep_natural_frequency",
            False
        )
    )

    sub=str(
        sub
    ).strip()

    json_data["subject"]=sub
    json_data["subject_id"]=sub

    experiment_dir=Path(
        experiment_dir
    ).expanduser().resolve()

    fe_dir=(
        experiment_dir
        /"5.Extra"
        /"FE"
    )

    fe_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results={
        "subject":sub,
        "subject_id":sub,
        "feature_directory":str(
            fe_dir
        )
    }

    if compute_standard:
        print(
            "🔧 Feature extraction: "
            "standard post-ICA features"
        )

        scalar_metrics,json_data=computeFeatExtraction_v2(
            postICA_final,
            json_data,
            experiment_dir,
            sub
        )

        results["standard"]=dict(
            scalar_metrics
        )

        manual_peaks=json_data.get(
            "feat_tep_manual"
        )

        if manual_peaks is not None:
            results["standard"][
                "manual_peaks"
            ]=manual_peaks

    if compute_pcist_feature:
        print(
            "🔧 Feature extraction: PCIst"
        )

        (
            pci_value,
            pcist_result,
            json_data
        )=compute_pcist(
            postICA_final,
            json_data,
            experiment_dir,
            sub
        )

        results["PCIst"]={
            "value":float(
                pci_value
            ),
            "n_dims":json_data.get(
                "PCIst_n_dims"
            ),
            "dNST":json_data.get(
                "PCIst_dNST"
            ),
            "parameters":json_data.get(
                "PCIst_parameters"
            ),
            "reference":json_data.get(
                "PCIst_reference"
            ),
            "output_directory":json_data.get(
                "PCIst_output_dir"
            )
        }

    if compute_tep_fingerprint_feature:
        print(
            "🔧 Feature extraction: "
            "TEP temporal fingerprint"
        )

        (
            fingerprint,
            df_fingerprint_channels,
            df_fingerprint_roi,
            json_data
        )=compute_tep_fingerprint(
            postICA_final=postICA_final,
            json_data=json_data,
            experiment_dir=experiment_dir,
            sub=sub,
            save=True
        )

        results["TEP_fingerprint"]={
            "seed_channel":fingerprint.get(
                "seed_channel"
            ),
            "roi_channels":fingerprint.get(
                "roi_channels"
            ),
            "ntop_requested":fingerprint.get(
                "ntop_requested"
            ),
            "ntop_selected":fingerprint.get(
                "ntop_selected"
            ),
            "LP1_ms":fingerprint.get(
                "LP1_ms"
            ),
            "LP2_ms":fingerprint.get(
                "LP2_ms"
            ),
            "LP3_ms":fingerprint.get(
                "LP3_ms"
            ),
            "AP1_P2_uV":fingerprint.get(
                "AP1_P2_uV"
            ),
            "AP2_P3_uV":fingerprint.get(
                "AP2_P3_uV"
            ),
            "SP1_P2_uV_per_ms":fingerprint.get(
                "SP1_P2_uV_per_ms"
            ),
            "SP2_P3_uV_per_ms":fingerprint.get(
                "SP2_P3_uV_per_ms"
            ),
            "abs_SP1_P2_uV_per_ms":fingerprint.get(
                "abs_SP1_P2_uV_per_ms"
            ),
            "abs_SP2_P3_uV_per_ms":fingerprint.get(
                "abs_SP2_P3_uV_per_ms"
            ),
            "IPI_ms":fingerprint.get(
                "IPI_ms"
            ),
            "IPI_Hz":fingerprint.get(
                "IPI_Hz"
            ),
            "reference":fingerprint.get(
                "reference"
            ),
            "output_directory":json_data.get(
                "TEP_fingerprint_output_dir"
            ),
            "channels_csv":json_data.get(
                "TEP_fingerprint_channels_csv"
            ),
            "roi_csv":json_data.get(
                "TEP_fingerprint_ROI_csv"
            ),
            "summary_csv":json_data.get(
                "TEP_fingerprint_summary_csv"
            ),
            "summary_json":json_data.get(
                "TEP_fingerprint_summary_json"
            ),
            "waveform_png":json_data.get(
                "TEP_fingerprint_waveform_png"
            ),
            "topomap_png":json_data.get(
                "TEP_fingerprint_topomap_png"
            )
        }

    if compute_tep_natural_frequency_feature:
        print(
            "🔧 Feature extraction: "
            "TEP Natural Frequency Rosanova"
        )

        (
            natural_frequency_results,
            json_data
        )=compute_tep_natural_frequency(
            postICA_final=postICA_final,
            json_data=json_data,
            experiment_dir=experiment_dir,
            sub=sub,
            save=True
        )

        results["TEP_natural_frequency"]={
            "natural_frequency_hz":natural_frequency_results.get(
                "natural_frequency_hz"
            ),
            "natural_frequency_power_db":natural_frequency_results.get(
                "natural_frequency_power_db"
            ),
            "band_power_db":natural_frequency_results.get(
                "band_power_db"
            ),
            "seed_channels":natural_frequency_results.get(
                "seed_channels"
            ),
            "figure":natural_frequency_results.get(
                "figure"
            ),
            "output_directory":natural_frequency_results.get(
                "output_directory"
            ),
            "reference":(
                "Rosanova et al. Natural Frequencies "
                "of Human Corticothalamic Circuits. "
                "Journal of Neuroscience, 2009."
            ),
            "reference_doi":(
                "10.1523/JNEUROSCI.0445-09.2009"
            )
        }

    if compute_fooof:
        print(
            "🔧 Feature extraction: "
            "channel-wise FOOOF"
        )

        df_fooof=extract_psd_features(
            postICA_final,
            "postICA_final",
            experiment_dir,
            json_data
        )

        results["FOOOF"]={
            "n_channels":int(
                len(
                    df_fooof
                )
            ),
            "csv":str(
                experiment_dir
                /"6.FOOOF"
                /"postICA_final"
                /"postICA_final.csv"
            )
        }

    scalar_results={}

    for section,values in results.items():
        if not isinstance(
            values,
            dict
        ):
            continue

        for key,value in values.items():
            if isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool
                )
            ) or value is None:
                scalar_results[
                    f"{section}_{key}"
                ]=value

    features_json=(
        fe_dir
        /f"{sub}_features_summary.json"
    )

    features_csv=(
        fe_dir
        /f"{sub}_features_summary.csv"
    )

    with open(
        features_json,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            make_json_serializable(
                results
            ),
            file,
            indent=4,
            sort_keys=True
        )

    if scalar_results:
        pd.DataFrame([
            scalar_results
        ]).to_csv(
            features_csv,
            index=False
        )

    json_data[
        "feature_extraction_dir"
    ]=str(
        fe_dir
    )

    json_data[
        "feature_extraction_completed"
    ]=True

    json_data[
        "feature_extraction_sections"
    ]=[
        key
        for key in results
        if key not in (
            "subject",
            "subject_id",
            "feature_directory"
        )
    ]

    json_data[
        "feature_extraction_summary_json"
    ]=str(
        features_json
    )

    json_data[
        "feature_extraction_summary_csv"
    ]=str(
        features_csv
    )

    with open(
        experiment_dir
        /f"{sub}_pars.json",
        "w",
        encoding="utf-8"
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
        f"✅ Feature extraction completata: "
        f"{fe_dir}"
    )

    print(
        "   Sections:",
        json_data[
            "feature_extraction_sections"
        ]
    )

    if compute_tep_natural_frequency_feature:
        print(
            "   Natural Frequency:",
            json_data.get(
                "TEP_natural_frequency_hz"
            ),
            "Hz"
        )

        print(
            "   Rosanova figure:",
            json_data.get(
                "TEP_natural_frequency_figure"
            )
        )

    return results,json_data

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
    with open(Path(experiment_dir) / f"{sub}_parsjsontxt", 'w') as txt_file:
        for key, value in sorted(json_data.items()):
            txt_file.write(f"{key}: {value}\n")

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
    plt.savefig(f"{experiment_dir}/5.final/FE/{sub}_FE_corrMatrix.png")
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
    plt.savefig(f"{experiment_dir}/5.final/FE/{sub}_FE_corrMatrix_seed.png")
    plt.close()

    # 4
    plt.figure(figsize=(FIGSIZE))
    plt.plot(postICA_final.times, postICA_final.average().get_data()[seed_indices, :].T , label=f"seed chans {json_data['seedChans']}", c='r')
    plt.plot(postICA_final.times, json_data['feats_step'], label='average TEP', linewidth=10)
    plt.xlabel("Time (ms)")
    plt.ylabel('Amplitude (µV) Seed TEP')
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.title('Average Seed TEP')
    plt.tight_layout()
    plt.savefig(f"{experiment_dir}/5.final/FE/{sub}_FE_STEP.png")
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
    with open(Path(experiment_dir) / f"{sub}_pars.json", 'w') as json_file:
            json.dump(json_data, json_file, indent=4, sort_keys=True)
        
    return json_data

    return scalar_metrics,json_data

def computeFeatExtraction_v2(
    postICA_final,
    json_data,
    experiment_dir,
    sub
):
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import scipy.integrate
    from pathlib import Path

    if postICA_final is None:
        raise ValueError(
            "computeFeatExtraction_v2 ha ricevuto postICA_final=None."
        )

    base_dir=Path(
        experiment_dir
    ).expanduser().resolve()

    output_dir=(
        base_dir
        /"5.Extra"
        /"FE"
        /"StandardFeatures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    times_s=np.asarray(
        postICA_final.times,
        dtype=float
    )

    times_ms=times_s*1000.0

    avg_data_v=np.asarray(
        postICA_final.average().get_data(),
        dtype=float
    )

    avg_data_uv=avg_data_v*1e6

    ch_names=list(
        postICA_final.ch_names
    )

    seed_names=[
        channel
        for channel in json_data.get(
            "seedChans",
            []
        )
        if channel in ch_names
    ]

    if not seed_names:
        raise ValueError(
            "Nessun seed channel presente in postICA_final."
        )

    seed_indices=[
        ch_names.index(
            channel
        )
        for channel in seed_names
    ]

    json_data["seedChans"]=list(
        seed_names
    )

    seed_tep_uv=np.mean(
        avg_data_uv[
            seed_indices,
            :
        ],
        axis=0
    )

    feature_window_ms=json_data.get(
        "standard_features_window_ms",
        json_data.get(
            "pcist_response_window_ms",
            (10.0,300.0)
        )
    )

    if (
        not isinstance(
            feature_window_ms,
            (list,tuple)
        )
        or len(feature_window_ms)!=2
    ):
        raise ValueError(
            "standard_features_window_ms deve contenere "
            "due valori [inizio,fine] in millisecondi."
        )

    feature_start_ms=float(
        feature_window_ms[0]
    )

    feature_end_ms=float(
        feature_window_ms[1]
    )

    data_min_ms=float(
        np.min(
            times_ms
        )
    )

    data_max_ms=float(
        np.max(
            times_ms
        )
    )

    effective_start_ms=max(
        feature_start_ms,
        data_min_ms
    )

    effective_end_ms=min(
        feature_end_ms,
        data_max_ms
    )

    if effective_start_ms>=effective_end_ms:
        raise ValueError(
            "Finestra standard features non valida: "
            f"richiesta=[{feature_start_ms},"
            f"{feature_end_ms}] ms; "
            f"dati=[{data_min_ms:.3f},"
            f"{data_max_ms:.3f}] ms."
        )

    feature_mask=(
        (times_ms>=effective_start_ms)
        &(times_ms<=effective_end_ms)
    )

    if np.sum(feature_mask)<2:
        raise ValueError(
            "La finestra delle standard features "
            "contiene meno di due campioni."
        )

    feature_signal_uv=seed_tep_uv[
        feature_mask
    ]

    feature_times_ms=times_ms[
        feature_mask
    ]

    energy_uv2=float(
        np.mean(
            feature_signal_uv**2
        )
    )

    absolute_integral_uv_ms=float(
        scipy.integrate.trapezoid(
            np.abs(
                feature_signal_uv
            ),
            feature_times_ms
        )
    )

    scalar_metrics={
        "energy_uV2":energy_uv2,
        "absolute_integral_uV_ms":absolute_integral_uv_ms
    }

    json_data["feats_step"]=seed_tep_uv.tolist()

    json_data[
        "feat_step_energy"
    ]=energy_uv2

    json_data[
        "feat_step_absolute_integral"
    ]=absolute_integral_uv_ms

    json_data[
        "feat_step_integral"
    ]=absolute_integral_uv_ms

    json_data[
        "feat_step_energy_unit"
    ]="uV^2"

    json_data[
        "feat_step_absolute_integral_unit"
    ]="uV*ms"

    json_data[
        "standard_features_window_requested_ms"
    ]=[
        feature_start_ms,
        feature_end_ms
    ]

    json_data[
        "standard_features_window_effective_ms"
    ]=[
        effective_start_ms,
        effective_end_ms
    ]

    json_data[
        "standard_features_seed_channels"
    ]=list(
        seed_names
    )

    json_data[
        "standard_features_metrics"
    ]=[
        "energy_uV2",
        "absolute_integral_uV_ms"
    ]

    json_data.pop(
        "feat_step_sampleEntropy",
        None
    )

    json_data.pop(
        "feat_step_permEntropy",
        None
    )

    json_data.pop(
        "feat_step_meanPLV_seed",
        None
    )

    json_data.pop(
        "feat_step_maxPLV_seed",
        None
    )

    json_data.pop(
        "feat_step_fooofOffset",
        None
    )

    json_data.pop(
        "feat_step_fooofExponent",
        None
    )

    metrics_df=pd.DataFrame([
        {
            "subject":str(sub),
            "metric":"Energy",
            "value":energy_uv2,
            "unit":"uV^2",
            "window_start_ms":effective_start_ms,
            "window_end_ms":effective_end_ms,
            "seed_channels":";".join(
                seed_names
            )
        },
        {
            "subject":str(sub),
            "metric":"Absolute integral",
            "value":absolute_integral_uv_ms,
            "unit":"uV*ms",
            "window_start_ms":effective_start_ms,
            "window_end_ms":effective_end_ms,
            "seed_channels":";".join(
                seed_names
            )
        }
    ])

    metrics_csv=(
        output_dir
        /f"{sub}_FE_scalarMetrics.csv"
    )

    metrics_df.to_csv(
        metrics_csv,
        index=False
    )

    figure,axis=plt.subplots(
        figsize=(12,6)
    )

    for channel_index,channel_name in zip(
        seed_indices,
        seed_names
    ):
        axis.plot(
            times_ms,
            avg_data_uv[
                channel_index,
                :
            ],
            linewidth=1.2,
            alpha=0.65,
            label=channel_name
        )

    axis.plot(
        times_ms,
        seed_tep_uv,
        color="black",
        linewidth=3,
        label="Seed average"
    )

    axis.axvspan(
        effective_start_ms,
        effective_end_ms,
        alpha=0.12,
        label="Feature window"
    )

    axis.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1
    )

    axis.axhline(
        0,
        color="black",
        linewidth=0.8
    )

    plot_start_ms=float(
        json_data.get(
            "epochs_plot_timewindow_min",
            times_s.min()
        )
    )*1000.0

    plot_end_ms=float(
        json_data.get(
            "epochs_plot_timewindow_max",
            times_s.max()
        )
    )*1000.0

    plot_start_ms=max(
        plot_start_ms,
        data_min_ms
    )

    plot_end_ms=min(
        plot_end_ms,
        data_max_ms
    )

    axis.set_xlim(
        plot_start_ms,
        plot_end_ms
    )

    axis.set_xlabel(
        "Time [ms]"
    )

    axis.set_ylabel(
        "Amplitude [µV]"
    )

    axis.set_title(
        f"{sub} seed TEP\n"
        f"seed={seed_names}"
    )

    axis.legend(
        loc="best"
    )

    axis.grid(
        True,
        alpha=0.20
    )

    figure.tight_layout()

    seed_plot=(
        output_dir
        /f"{sub}_FE_STEP.png"
    )

    figure.savefig(
        seed_plot,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(
        figure
    )

    figure,axes=plt.subplots(
        1,
        2,
        figsize=(11,5)
    )

    axes[0].bar(
        ["Energy"],
        [energy_uv2]
    )

    axes[0].set_ylabel(
        "µV²"
    )

    axes[0].set_title(
        "TEP energy"
    )

    axes[0].text(
        0,
        energy_uv2,
        f"{energy_uv2:.4g}",
        ha="center",
        va="bottom"
    )

    axes[1].bar(
        ["Absolute integral"],
        [absolute_integral_uv_ms]
    )

    axes[1].set_ylabel(
        "µV·ms"
    )

    axes[1].set_title(
        "Absolute TEP integral"
    )

    axes[1].text(
        0,
        absolute_integral_uv_ms,
        f"{absolute_integral_uv_ms:.4g}",
        ha="center",
        va="bottom"
    )

    figure.suptitle(
        f"{sub} standard TEP features\n"
        f"window={effective_start_ms:.1f}–"
        f"{effective_end_ms:.1f} ms"
    )

    figure.tight_layout()

    metrics_plot=(
        output_dir
        /f"{sub}_FE_scalarMetrics.png"
    )

    figure.savefig(
        metrics_plot,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(
        figure
    )

    json_data[
        "feature_standard_output_dir"
    ]=str(
        output_dir
    )

    json_data[
        "feature_standard_scalar_metrics_csv"
    ]=str(
        metrics_csv
    )

    json_data[
        "feature_standard_scalar_metrics_plot"
    ]=str(
        metrics_plot
    )

    json_data[
        "feature_standard_seed_tep_plot"
    ]=str(
        seed_plot
    )

    json_data[
        "feature_standard_corr_matrix_plot"
    ]=None

    json_data[
        "feature_standard_seed_corr_plot"
    ]=None

    json_data_clean=make_json_serializable(
        json_data
    )

    with open(
        base_dir/f"{sub}_pars.json",
        "w",
        encoding="utf-8"
    ) as json_file:
        json.dump(
            json_data_clean,
            json_file,
            indent=4,
            sort_keys=True
        )

    print(
        "✅ Standard TEP features completate"
    )

    print(
        f"   Seed channels: {seed_names}"
    )

    print(
        f"   Window: {effective_start_ms:.1f}–"
        f"{effective_end_ms:.1f} ms"
    )

    print(
        f"   Energy: {energy_uv2:.6g} µV²"
    )

    print(
        "   Absolute integral: "
        f"{absolute_integral_uv_ms:.6g} µV·ms"
    )

    return scalar_metrics,json_data

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
        ax.plot(times, signals * 1e6, label=f"average seed of {seed}", color='k', alpha=0.8, linewidth=5)

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
    with open(Path(experiment_dir) / f"{sub}_pars.json", 'w') as json_file:
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
    import mne
    from mne.preprocessing import ICA
    from mne_icalabel import label_components
    import json

    save_dir = Path(experiment_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    bad_channels = list(EPOCHS.info.get("bads", []))
    bad_channels = [ch for ch in bad_channels if ch in EPOCHS.ch_names]

    print(f"📌 Bad channels marked before ICA: {bad_channels}")

    picks_ica = mne.pick_types(
        EPOCHS.info,
        eeg=True,
        exclude="bads"
    )

    if len(picks_ica) < 2:
        raise ValueError(
            f"Troppi pochi canali buoni per ICA. "
            f"Good EEG channels={len(picks_ica)}, bad_channels={bad_channels}"
        )

    if n_components is None:
        n_components = max(1, len(picks_ica) - 1)

    json_data["ICA_badChannelsExcludedFromFit"] = bad_channels
    json_data["ICA_goodChannelsForFit"] = [
        EPOCHS.ch_names[i] for i in picks_ica
    ]
    json_data["ICA_nGoodChannelsForFit"] = int(len(picks_ica))
    json_data["ICA_nComponentsRequested"] = int(n_components)

    print(f"🧠 ICA fit on good channels only: {len(picks_ica)} channels")
    print(f"🧠 ICA n_components: {n_components}")

    epochs_ica = EPOCHS.copy().pick(picks_ica)

    ica_random_state=int(
        json_data.get(
            "ICA_seed",
            42
        )
    )
    json_data["ICA_seed"]=ica_random_state
    ica=ICA(
        n_components=n_components,
        method="fastica",
        random_state=ica_random_state,
        max_iter="auto"
    )
 
    ica.fit(epochs_ica)

    ic_labels = label_components(
        epochs_ica,
        ica,
        method="iclabel"
    )

    labels = ic_labels["labels"]

    artifact_tags = [
        "eye blink",
        "muscle artifact",
        "heart beat",
        "line noise",
        "channel noise",
        "other"
    ]

    auto_excluded = []
    low_eigen_excluded = []

    mixing_matrix = ica.mixing_matrix_
    eigenvalues = np.linalg.svd(mixing_matrix, compute_uv=False) ** 2
    threshold = np.percentile(eigenvalues, threshold_percentile)

    if autoReject:
        for i, label in enumerate(labels):
            probs = np.array(ic_labels["y_pred_proba"][i], ndmin=1)
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
        fig1 = ica.plot_components(
            picks=initial_excluded,
            inst=epochs_ica,
            show_names=False,
            show=False
        )
        fig1.savefig(save_dir / f"{sub}_AUTO_excluded_ICAs.png")
        plt.close(fig1)

    if initial_remaining:
        fig2 = ica.plot_components(
            picks=initial_remaining,
            inst=epochs_ica,
            show_names=False,
            show=False
        )
        fig2.savefig(save_dir / f"{sub}_AUTO_included_ICAs.png")
        plt.close(fig2)

    fig, ax = plt.subplots(figsize=(10, 5))

    above_threshold = np.where(eigenvalues >= threshold)[0]
    below_threshold = np.where(eigenvalues < threshold)[0]

    ax.plot(
        below_threshold,
        eigenvalues[below_threshold],
        marker="o",
        linestyle="-",
        color="black",
        label="Eigenvalues"
    )

    ax.scatter(
        above_threshold,
        eigenvalues[above_threshold],
        color="red",
        label="Above threshold",
        zorder=3
    )

    ax.axhline(
        threshold,
        color="r",
        linestyle="--",
        label=f"Threshold ({threshold_percentile}° percentile)"
    )

    ax.set_xlabel("ICA component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Eigenvalues of ICA components")
    ax.legend()

    fig.savefig(save_dir / f"{sub}_eigenvalueDist.png")
    plt.close(fig)

    components_dir = save_dir / "components"
    components_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(ica.n_components_):
        tag = labels[idx] if labels is not None else "Unknown"
        tag_clean = tag.replace("/", "_").replace(" ", "")

        fig = ica.plot_components(
            picks=idx,
            inst=epochs_ica,
            show=False
        )

        if isinstance(fig, list):
            for j, f in enumerate(fig):
                fname = components_dir / f"component_{idx}_{tag_clean}_view{j}.png"
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
            ica = tmspath_utils_adj.ICApp(
                ica,
                epochs_ica.copy()
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

    postICA_clean.info["bads"] = bad_channels

    final_remaining = sorted(list(all_components - set(final_excluded)))

    if final_excluded:
        fig3 = ica.plot_components(
            picks=final_excluded,
            inst=epochs_ica,
            show_names=False,
            show=False
        )
        fig3.savefig(save_dir / f"{sub}_FINAL_excluded_ICAs.png")
        plt.close(fig3)

    if final_remaining:
        fig4 = ica.plot_components(
            picks=final_remaining,
            inst=epochs_ica,
            show_names=False,
            show=False
        )
        fig4.savefig(save_dir / f"{sub}_FINAL_included_ICAs.png")
        plt.close(fig4)

    with open(save_dir / f"{sub}_ICA_selection_summary.json", "w") as f:
        json.dump({
            "bad_channels_excluded_from_fit": json_data["ICA_badChannelsExcludedFromFit"],
            "good_channels_for_fit": json_data["ICA_goodChannelsForFit"],
            "n_good_channels_for_fit": json_data["ICA_nGoodChannelsForFit"],
            "n_components_requested": json_data["ICA_nComponentsRequested"],
            "auto_excluded": json_data["ICA_autoExcludedComponents"],
            "low_eigen_excluded": json_data["ICA_lowEigenExcludedComponents"],
            "initial_excluded": json_data["ICA_initialExcludedComponents"],
            "final_excluded": json_data["ICA_finalExcludedComponents"],
            "manual_added": json_data["ICA_manualAddedComponents"],
            "manual_recovered": json_data["ICA_manualRecoveredComponents"]
        }, f, indent=4)

    return postICA_clean, ica

def postICAsteps(postICA_raw, json_data, experiment_dir, sub):
    import pickle
    import json
    from pathlib import Path

    postICA_final = postICA_raw.copy().filter(
        l_freq=json_data["l_freq"],
        h_freq=json_data["h_freq"],
        method="fir",
        verbose=True
    )

    newrate = json_data["sfreq"]
    postICA_final = postICA_final.resample(sfreq=newrate)

    postICA_final = postICA_final.pick("eeg")

    bad_channels = json_data.get("bad_channels", [])
    bad_channels = [
        ch for ch in bad_channels
        if ch in postICA_final.ch_names
    ]

    if len(bad_channels) > 0:
        print(f"🧩 Interpolating bad channels after ICA: {bad_channels}")
        postICA_final.info["bads"] = bad_channels
        postICA_final.interpolate_bads(reset_bads=True)
    else:
        print("✅ No bad channels available for interpolation after ICA")
        postICA_final.info["bads"] = []

    postICA_final.set_eeg_reference("average")

    times = postICA_final.times
    ch_names = postICA_final.ch_names

    json_data["postICA_final_l_freq"] = float(json_data["l_freq"])
    json_data["postICA_final_h_freq"] = float(json_data["h_freq"])
    json_data["postICA_final_sfreq"] = float(postICA_final.info["sfreq"])
    json_data["channels_interpolated_after_ica"] = bad_channels
    json_data["channel_interpolation_timing"] = "after_ica"
    json_data["reference_after_ica"] = "average"
    json_data["postICA_final_channels"] = list(ch_names)

    pkl_dir = Path(experiment_dir) / "7.pkls"
    pkl_dir.mkdir(parents=True, exist_ok=True)

    with open(pkl_dir / f"{sub}_postICA_final.pkl", "wb") as f:
        pickle.dump(postICA_final, f)

    json_data["mneEpochArrayFinal"] = postICA_final

    json_data_clean = make_json_serializable(json_data)

    with open(Path(experiment_dir) / f"{sub}_pars.json", "w") as json_file:
        json.dump(json_data_clean, json_file, indent=4)

    return postICA_final



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
        fig.savefig(f"{experiment_dir}/{subDir}/{sub}_ERSP_{saveNote}.png")

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
        
        fig.savefig(f"{experiment_dir}/{subDir}/{sub}_scalpmaptime_{saveNote}.png")
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


def run_ica_artist_ext_only(EPOCHS, n_components=None, ext_threshold_uv=30, manualCheck=True, subPath='4.postICA', saveNote='postICA'):
    import os
    os.makedirs(f"{experiment_dir}/{subPath}", exist_ok=True)

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

    os.makedirs(f"{experiment_dir}/{subPath}", exist_ok=True)

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
    ica_random_state=int(
        json_data.get(
            "ICA_seed",
            42
        )
    )
    json_data["ica_random_state"]=ica_random_state
    
    ica=mne.preprocessing.ICA(
        n_components=n_components,
        method="fastica",
        random_state=ica_random_state,
        max_iter="auto"
    )
   
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
    ax.set_title(f"{sub} - {key}")
    ax.legend(loc='upper right', frameon=True, shadow=True)
    #fig.tight_layout()
    fig.savefig(f"{experiment_dir}/{subPath}/butterflyPaper_{key}.png")
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

def get_epoch_plot_limits(epochs,json_data):
    data_min=float(epochs.times.min())
    data_max=float(epochs.times.max())

    plot_min=float(
        json_data.get(
            "epochs_plot_timewindow_min",
            data_min
        )
    )

    plot_max=float(
        json_data.get(
            "epochs_plot_timewindow_max",
            data_max
        )
    )

    plot_min=max(
        plot_min,
        data_min
    )

    plot_max=min(
        plot_max,
        data_max
    )

    if plot_min>=plot_max:
        raise ValueError(
            f"Limiti grafici non validi: "
            f"[{plot_min},{plot_max}] s; "
            f"dati disponibili [{data_min},{data_max}] s."
        )

    return plot_min,plot_max
    
def basicPlots(
    EPOCHS,
    json_data,
    experiment_dir,
    sub,
    key="epochs",
    subPath="1.basic",
    figsize=FIGSIZE,
    show=False,
    do_psdtopomap=False
):
    from pathlib import Path
    import matplotlib.pyplot as plt

    output_dir=Path(experiment_dir)/subPath
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plot_tmin,plot_tmax=get_epoch_plot_limits(
        EPOCHS,
        json_data
    )

    evoked=EPOCHS.average()

    fig=evoked.plot(
        show=show,
        spatial_colors=True,
        time_unit="s"
    )

    fig.set_size_inches(
        figsize[0],
        figsize[1]
    )

    for ax in fig.axes:
        if ax.get_xlabel() or ax.lines:
            ax.set_xlim(
                plot_tmin,
                plot_tmax
            )

    fig.savefig(
        output_dir/f"tep_{key}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    evoked_plot=evoked.copy().crop(
        tmin=plot_tmin,
        tmax=plot_tmax,
        include_tmax=True
    )

    fig=evoked_plot.plot_topo(
        show=show
    )

    fig.set_size_inches(
        figsize[0],
        figsize[1]
    )

    fig.savefig(
        output_dir/f"topo_{key}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    fig=EPOCHS.compute_psd(
        method="welch",
        fmin=max(
            0.0,
            float(
                EPOCHS.info.get(
                    "highpass",
                    0.0
                )
            )
        ),
        fmax=float(
            EPOCHS.info.get(
                "lowpass",
                EPOCHS.info["sfreq"]/2
            )
        )
    ).plot(
        xscale="log",
        show=show
    )

    fig.set_size_inches(
        figsize[0],
        figsize[1]
    )

    fig.savefig(
        output_dir/f"PSD_{key}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    if do_psdtopomap:
        fig=EPOCHS.compute_psd(
            method="welch"
        ).plot_topomap(
            cmap="turbo",
            show=show,
            normalize=True
        )

        fig.set_size_inches(
            figsize[0],
            figsize[1]
        )

        fig.savefig(
            output_dir/f"PSD_topomap_{key}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

    json_data["epochs_plot_timewindow_effective"]=[
        float(plot_tmin),
        float(plot_tmax)
    ]


def create_butterfly_topomap_gif(
    epochs,
    json_data,
    experiment_dir,
    sub,
    saveNote="postICA_final",
    subPath="5.Extra",
    tmin=None,
    tmax=None,
    step=0.001,
    xlim=(-0.1,0.45),
    vlim="p98",
    sphere=0.095,
    duration=50,
    transparency=False,
    dpi=120,
    save_static=True
):
    import os
    import imageio
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_dir=Path(experiment_dir)/subPath
    out_dir.mkdir(parents=True,exist_ok=True)

    if hasattr(epochs,"average"):
        evoked=epochs.average()
        data_epochs=epochs.get_data()
        tep=np.mean(data_epochs,axis=0).T
        times_data=epochs.times
    elif hasattr(epochs,"data") and hasattr(epochs,"times"):
        evoked=epochs
        tep=evoked.data.T
        times_data=evoked.times
    else:
        raise TypeError("epochs deve essere un oggetto mne.Epochs oppure mne.Evoked.")

    if tmin is None:
        tmin=float(times_data.min())

    if tmax is None:
        tmax=float(times_data.max())

    tmin=max(tmin,float(times_data.min()))
    tmax=min(tmax,float(times_data.max()))

    times=np.round(np.arange(tmin,tmax+step,step),4)
    times=[float(t) for t in times if tmin<=t<=tmax]

    if vlim=="p98":
        p=float(np.nanpercentile(np.abs(evoked.data),98))
        vmin,vmax=-p,p
    elif vlim=="p99":
        p=float(np.nanpercentile(np.abs(evoked.data),99))
        vmin,vmax=-p,p
    elif isinstance(vlim,(tuple,list)) and len(vlim)==2:
        vmin,vmax=float(vlim[0]),float(vlim[1])
    else:
        vmax=float(np.nanmax(np.abs(evoked.data)))
        vmin=-vmax

    fig_static,ax_static=plt.subplots(figsize=(12,6))
    ax_static.plot(times_data,tep,c="b",alpha=0.5,linewidth=0.8)
    ax_static.axvline(0,linewidth=2,c="k",alpha=0.5)
    ax_static.set_xlim(xlim)
    ax_static.set_xlabel("Time (s)")
    ax_static.set_ylabel("Amplitude (V)")
    ax_static.set_title(f"{sub} - {saveNote} butterfly")
    ax_static.grid(True,alpha=0.3)
    fig_static.tight_layout()

    if save_static:
        fig_static.savefig(out_dir/f"{sub}_{saveNote}_butterfly.png",dpi=300,bbox_inches="tight")
        fig_static.savefig(out_dir/f"{sub}_{saveNote}_butterfly.svg",dpi=300,bbox_inches="tight")

    plt.close(fig_static)

    frames=[]

    fig=plt.figure(figsize=(20,6),dpi=dpi)

    if transparency:
        fig.patch.set_alpha(0.0)

    gs=fig.add_gridspec(1,3,width_ratios=[5,1,0.12])
    ax_butterfly=fig.add_subplot(gs[0])
    ax_topo=fig.add_subplot(gs[1])
    ax_cbar=fig.add_subplot(gs[2])

    ax_butterfly.plot(times_data,tep,c="b",alpha=0.5,linewidth=0.8)
    ax_butterfly.axvline(0,linewidth=4,c="k",alpha=0.4)
    ax_butterfly.set_xlim(xlim)
    ax_butterfly.set_xlabel("Time (s)")
    ax_butterfly.set_ylabel("Amplitude (V)")
    ax_butterfly.set_title(f"{sub} - {saveNote}")
    ax_butterfly.grid(True,alpha=0.3)

    time_indicator=ax_butterfly.axvline(times[0],color="g",linestyle="--",linewidth=2)

    for t in times:
        time_indicator.set_xdata([t,t])

        evoked.plot_topomap(
            times=t,
            ch_type="eeg",
            contours=10,
            show=False,
            time_unit="s",
            vlim=(vmin,vmax),
            outlines="head",
            extrapolate="head",
            sphere=sphere,
            sensors=False,
            axes=[ax_topo,ax_cbar]
        )

        ax_topo.set_title(f"{t*1000:.0f} ms")

        fig.canvas.draw()

        image=np.frombuffer(fig.canvas.tostring_argb(),dtype=np.uint8)
        image=image.reshape(fig.canvas.get_width_height()[::-1]+(4,))
        image=np.roll(image,3,axis=2)

        frames.append(image.copy())

        ax_topo.clear()
        ax_cbar.clear()

    fig.tight_layout()

    alpha_note="transparent" if transparency else "opaque"
    gif_path=out_dir/f"{sub}_{saveNote}_butterfly_topomap_{alpha_note}.gif"

    imageio.mimsave(
        gif_path,
        frames,
        duration=duration,
        loop=0
    )

    plt.close(fig)

    json_data[f"gif_{saveNote}_butterfly_topomap"]=str(gif_path)

    print(f"✅ GIF salvata: {gif_path}")

    return gif_path,json_data
    

def runICA(detrendedEpochs):

    # find the maximum number of independent components
    # as the number of good channels - 1 because of average referencing
    
    n_components = len(detrendedEpochs.ch_names) - len(detrendedEpochs.info['bads']) - 1
    print(n_components)
    json_data['n_components'] = n_components
    # Salva parametri
    with open(Path(experiment_dir) / f"{sub}_pars.json", 'w') as txt_file:
        for key, value in sorted(json_data.items()):
            txt_file.write(f"{key}: {value}\n")
    
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
    save_dir = Path(experiment_dir) / '7.FOOOF' / note
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
                file_name=f"chan_{ch_name}_{note}",
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
    with open(Path(experiment_dir) / f"{sub}_pars.json", 'w') as json_file:
            json.dump(json_data, json_file, indent=4, sort_keys=True)

def plotTrialTepVariability(
    epochs,
    json_data,
    experiment_dir,
    sub,
    chanNAME="AF3",
    operator=np.mean,
    save=False,
    figsize=FIGSIZE,
    parDir="preDetrend"
):
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    plot_tmin,plot_tmax=get_epoch_plot_limits(
        epochs,
        json_data
    )

    channel_index=epochs.ch_names.index(
        chanNAME
    )

    channel_data=epochs.get_data()[
        :,
        channel_index,
        :
    ]

    fig,ax=plt.subplots(
        figsize=figsize
    )

    ax.plot(
        epochs.times,
        channel_data.T,
        color="b",
        linewidth=1,
        alpha=0.5
    )

    ax.plot(
        epochs.times,
        operator(
            channel_data,
            axis=0
        ),
        color="r",
        linewidth=5,
        label=operator.__name__
    )

    ax.axvline(
        0,
        color="k",
        linestyle="--"
    )

    ax.set_xlim(
        plot_tmin,
        plot_tmax
    )

    ax.set_title(
        chanNAME
    )

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        "Amplitude [V]"
    )

    ax.legend()

    fig.tight_layout()

    if save:
        output_dir=(
            Path(experiment_dir)
            /"2.trials"
            /parDir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        fig.savefig(
            output_dir/f"{sub}_tepVar_{chanNAME}.png",
            dpi=300,
            bbox_inches="tight"
        )

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
        plt.scatter(t_peak,sigAll[i_peak],s=60,label=f"peak @ {t_peak:.4f}s")
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
        plt.scatter(t_peak, sig[tMaxOffsetIndex], s=60, label=f"peak @ {t_peak:.4f}s")
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
        plt.title(f"{sub}-{epochs.ch_names[chan]}-trial{trial}")
        plt.scatter(epochs.times[tempMaskOffset][tMaxOffsetIndex], sig[tMaxOffsetIndex], c='r', 
                    label=f"peak point at {epochs.times[tempMaskOffset][tMaxOffsetIndex]}")
        plt.axvline(x=offset, label=f"maxTimeWindowOffset={json_data['detrend_maxTimeWindowOffset']}")
        plt.legend(loc='upper right')
        plt.savefig(f"{experiment_dir}/3.detrend/test_maskTest_{sub}_{chan}_{trial}.png")
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
        plt.title(f"{sub}-{epochs.ch_names[chan]}-trial{trial}")
        plt.scatter(epochs.times[tempMaskOffset][tMaxOffsetIndex], sig[tMaxOffsetIndex], c='r', 
                    label=f"peak point at {epochs.times[tempMaskOffset][tMaxOffsetIndex]}")
        plt.axvline(x=offset, label=f"maxTimeWindowOffset={json_data['detrend_maxTimeWindowOffset']}")
        plt.legend(loc='upper right')
        plt.savefig(f"{experiment_dir}/3.detrend/test/maskTest_{sub}_{chan}_{trial}.png")
        plt.close()
        
    maskOffset = np.logical_and(epochs.times>=json_data['detrend_minTimeWindowOffset'], 
                                epochs.times<=epochs.times[tempMaskOffset][tMaxOffsetIndex])   
    
    maskPostOffset = np.logical_and(epochs.times>epochs.times[tempMaskOffset][tMaxOffsetIndex], 
                                    epochs.times<=epochs.times.max())

    return maskPreOffset, maskOffset, maskPostOffset

def computeSlopes_v4(epochs, json_data, experiment_dir, sub, saveNote=f"plotTrialTepVariability"):
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

    
def computeSlopes_v4_old(epochs, json_data, experiment_dir, sub, saveNote=f"plotTrialTepVariability"):
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
                         saveNote='ALL-TRIALS', sharex=True, subPath='3.detrend',
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
    fig.suptitle(f"{saveNote} - N° Trials: {ntrial}")

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
    out_path = os.path.join(experiment_dir, subPath, f"{VAR}_{saveNote}.png")
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
    plt.title(f"{saveNote} - N° Trials: {ntrial} \n ANOVA: F={anova_stat:.3f}, p={p_value:.3f} \n seedChans={seed_list}")
    histo_path = os.path.join(experiment_dir, subPath, f"histo_{VAR}_{saveNote}.png")
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
                         saveNote='ALL-TRIALS', sharex=True, subPath='3.detrend', zvalue=True, json_data=None, experiment_dir='.'):
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
    fig.suptitle(f"{saveNote} - N° Trials: {ntrial}")

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
    out_path = os.path.join(experiment_dir, subPath, f"{VAR}_{saveNote}.png")
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
    plt.title(f"{saveNote} - N° Trials: {ntrial} \n ANOVA: F={anova_stat:.3f}, p={p_value:.3f} \n seedChans={seed_list}")

    histo_path = os.path.join(experiment_dir, subPath, f"histo_{VAR}_{saveNote}.png")
    plt.savefig(histo_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {'p_value': p_value, 'F': anova_stat, 'outlier_channels': outlier_channels}


def computeSlopesPlot(df_slopes, 
                      json_data, experiment_dir, sub,
                      saveNote='ALL-TRIALS', sharex=True, subPath='3.detrend', zvalue=True):
    
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
    fig.suptitle(f"{saveNote} - N° Trials: {ntrial}")

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
        out_path = os.path.join(experiment_dir, subPath, f"{VAR}_{saveNote}_{label}.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        data_subset.to_csv(out_path, index=False)


    # Salvataggio del primo plot
    fig.savefig(f"{experiment_dir}/{subPath}/{VAR}_{saveNote}.png")
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
    plt.title(f"{saveNote} - N° Trials: {ntrial} \n ANOVA: F={anova_stat:.3f}, p={p_value:.3f} \n seedChans={json_data['seedChans']}")
    plt.savefig(f"{experiment_dir}/{subPath}/histo_{VAR}_{saveNote}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print('p_value', p_value, 'F', anova_stat)

    return df_anova_results


def generate_noise_from_distribution(
    time_series,
    model="Gaussian",
    n_samples=1000,
    rng=None
):
    import numpy as np

    if rng is None:
        rng=np.random.default_rng(42)

    x=np.asarray(
        time_series,
        dtype=float
    ).ravel()

    x=x[
        np.isfinite(x)
    ]

    if x.size==0:
        return np.zeros(
            n_samples,
            dtype=float
        )

    mean=float(
        np.mean(x)
    )

    std=float(
        np.std(x)*0.5
    )

    median=float(
        np.median(x)
    )

    minimum=float(
        np.min(x)
    )

    maximum=float(
        np.max(x)
    )

    epsilon=1e-8
    std=max(
        std,
        epsilon
    )

    if not np.isfinite(minimum) or not np.isfinite(maximum):
        minimum=mean-std
        maximum=mean+std

    if minimum==maximum:
        minimum-=std
        maximum+=std

    if minimum>maximum:
        minimum,maximum=maximum,minimum

    model_name=str(
        model
    ).strip().lower()

    if model_name=="gaussian":
        return rng.normal(
            loc=mean,
            scale=std,
            size=n_samples
        )

    if model_name=="exponential":
        return rng.exponential(
            scale=max(std,epsilon),
            size=n_samples
        )

    if model_name=="laplace":
        return rng.laplace(
            loc=median,
            scale=std,
            size=n_samples
        )

    if model_name=="poisson":
        return rng.poisson(
            lam=max(mean,0.0),
            size=n_samples
        )

    if model_name=="rayleigh":
        return rng.rayleigh(
            scale=std,
            size=n_samples
        )

    if model_name=="gamma":
        shape=(
            mean**2/std**2
            if std>epsilon
            else 1.0
        )

        shape=max(
            shape,
            epsilon
        )

        scale=(
            std**2
            /max(
                abs(mean),
                epsilon
            )
        )

        return rng.gamma(
            shape=shape,
            scale=scale,
            size=n_samples
        )

    if model_name in (
        "studentt",
        "student_t",
        "t"
    ):
        return (
            rng.standard_t(
                df=2.0,
                size=n_samples
            )
            *std
            +mean
        )

    if model_name=="uniform":
        return rng.uniform(
            low=minimum,
            high=maximum,
            size=n_samples
        )

    raise ValueError(
        f"Modello non supportato: {model}"
    ) 

def generate_noise_from_distribution_old01082026(time_series, model='Gaussian', n_samples=1000):
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


    base_seed=int(
        json_data.get(
            "detrend_noise_seed",
            42
        )
    )
    
    json_data[
        "detrend_noise_seed"
    ]=base_seed
    
    json_data[
        "detrend_noise_generator"
    ]="numpy.random.default_rng"
    
    json_data[
        "detrend_noise_seed_strategy"
    ]="SeedSequence(base_seed,channel_index,epoch_index)"

    
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


    for chan in tqdm(
        EPOCHS.ch_names
    ):
        if (
            doDetrendOnlyOffsetChans
            and chan not in offsetChans
        ):
            continue  # salta questo canale, mantiene i dati originali
    
        id_chan=int(
            np.where(
                np.asarray(
                    EPOCHS.ch_names
                )==chan
            )[0][0]
        )
    
        for epoch_idx in range(
            data_detrended.shape[0]
        ):
            rng=np.random.default_rng(
                np.random.SeedSequence([
                    base_seed,
                    int(id_chan),
                    int(epoch_idx)
                ])
            )
    
            tep=data_detrended[
                epoch_idx,
                id_chan,
                :
            ].reshape(
                -1,
                1
            )

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
            OPTPARS_A = f"wind_poly_{json_data['detrend_polOrder_preOffset']}"

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
                    tep_agg = apply_offset_correction(tep_agg, tep, timeMask, correctMode, oddSamples, EPOCHS, supported_models, rng=rng)
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
                tep_agg = apply_offset_correction(tep_agg, tep, timeMask, correctMode, oddSamples, EPOCHS, supported_models, rng=rng)
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
    tabStat_path = os.path.join(experiment_dir, '3.detrend', f"tabStatDetrend_{typeOffsetDecay}.csv")
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
        f"{sub} | {chan} | epoch={epoch_idx} \n fitConstrain={fitConstraint} \n "
#        f'orderPre={orderPreOffset} | orderArt={orderOffset} | orderPost={orderPostOffset_temp}\n'
        f"Min Offset Time: {min_offset_time:.4f} | Max Offset Time: {max_offset_time:.4f} \n"
        f"typeOffsetRise={typeOffsetRise} with pars={OPTPARS_B} \n"
        f"typeOffsetDecay={typeOffsetDecay} with pars={OPTPARS_C} \n"     
    )
    filename = f"{sub}_{chan}_{epoch_idx}_{fitConstraint}.png"

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
    out_path = os.path.join(experiment_dir, '3.detrend', 'examples', filename)
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
            f"{sub} | {chan} | epoch={epoch_idx} \n fitConstrain={fitConstraint} \n "
            f"orderPre={orderPreOffset} | orderArt={orderOffset} | orderPost={orderPostOffset_temp}\n"
            f"Min Offset Time: {min_offset_time:.4f} | Max Offset Time: {max_offset_time:.4f}"
        )
        filename = f"{sub}_{chan}_{epoch_idx}_{fitConstraint}_{orderPostOffset_temp}.png"
        
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
            exp_info = f"Art exp={orderOffset} with pars={popt_B}\n"
            postart_info = f"postArt exp={detrendMode} with pars={popt_C}"
        else:
            exp_info = f"noExp but Lagrange with order={orderOffset}\n"
            postart_info = 'no postArt exp fit'
        
        title = (
            f"{sub} | {chan} | epoch={epoch_idx} \n"
            f"fitConstrain={fitConstraint} \n"
            f"orderPre={orderPreOffset} | {exp_info}"
            f"Min Offset Time: {min_offset_time:.4f} | Max Offset Time: {max_offset_time:.4f} \n"
            f"{postart_info}"
        )
        filename = f"{sub}_detrendExample_{chan}_{epoch_idx}_{fitConstraint}_{detrendMode}.png"

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
    out_path = os.path.join(experiment_dir, '3.detrend', 'examples', filename)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_tabStat(df_tabStat, experiment_dir):
    # Crea la cartella per i plot
    output_dir = os.path.join(experiment_dir, '3.detrend', 'statDetrend')
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


def apply_offset_correction(
    tep_agg,
    tep,
    timeMask,
    correctMode,
    oddSamples,
    EPOCHS,
    supported_models,
    rng=None
):
    import numpy as np
    from scipy.signal import resample

    if rng is None:
        rng=np.random.default_rng(42)

    tep_agg=np.asarray(
        tep_agg,
        dtype=float
    ).copy()

    tep=np.asarray(
        tep,
        dtype=float
    ).reshape(-1)

    times=np.asarray(
        EPOCHS.times,
        dtype=float
    )

    odd_samples_ms=(
        0.0
        if oddSamples is None or oddSamples is False
        else float(oddSamples)
    )

    margin_seconds=(
        odd_samples_ms
        /1000.0
    )

    pre_indices=np.flatnonzero(
        timeMask[0]
    )

    offset_indices=np.flatnonzero(
        timeMask[1]
    )

    if (
        pre_indices.size==0
        or offset_indices.size==0
    ):
        return tep_agg

    pre_end_time=float(
        times[
            pre_indices[-1]
        ]
    )

    offset_start_time=float(
        times[
            offset_indices[0]
        ]
    )

    offset_end_time=float(
        times[
            offset_indices[-1]
        ]
    )

    precorrectionMask=(
        (times>=float(times[pre_indices[0]]))
        &(times<=pre_end_time-margin_seconds)
    )

    if not np.any(
        precorrectionMask
    ):
        precorrectionMask=timeMask[
            0
        ].copy()

    correctionMask=(
        (times>=offset_start_time-margin_seconds)
        &(times<=offset_end_time+margin_seconds)
    )

    num_samples=int(
        np.sum(
            correctionMask
        )
    )

    pre_series=tep[
        precorrectionMask
    ]

    pre_series=pre_series[
        np.isfinite(
            pre_series
        )
    ]

    if (
        num_samples==0
        or pre_series.size==0
    ):
        return tep_agg

    mode=(
        str(correctMode).strip()
        if correctMode is not False
        else ""
    )

    mode_lower=mode.lower()

    supported_lower={
        str(item).lower()
        for item in supported_models
    }

    if mode_lower=="moving_average":
        window_size=max(
            1,
            int(
                round(
                    odd_samples_ms
                )
            )
        )

        if pre_series.size>=window_size:
            value=float(
                np.mean(
                    pre_series[
                        -window_size:
                    ]
                )
            )
        else:
            value=float(
                np.mean(
                    pre_series
                )
            )

        new_samples=np.full(
            num_samples,
            value,
            dtype=float
        )

    elif mode_lower=="median":
        new_samples=np.full(
            num_samples,
            float(
                np.median(
                    pre_series
                )
            ),
            dtype=float
        )

    elif mode_lower=="zeros":
        new_samples=np.zeros(
            num_samples,
            dtype=float
        )

    elif mode_lower=="resample":
        new_samples=resample(
            pre_series,
            num=num_samples
        )

    elif mode_lower in supported_lower:
        new_samples=generate_noise_from_distribution(
            time_series=pre_series,
            model=mode,
            n_samples=num_samples,
            rng=rng
        )

    else:
        raise ValueError(
            f"Metodo di correzione offset "
            f"non riconosciuto: {correctMode}"
        )

    tep_agg[
        correctionMask
    ]=new_samples

    return tep_agg


def apply_offset_correction_oldB01082026(tep_agg, tep, timeMask, correctMode, oddSamples, EPOCHS, supported_models):
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


def apply_offset_correction_oldA01082026(tep_agg, tep, timeMask, correctMode, oddSamples, EPOCHS, supported_models):
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



def plot_slope_resonances(PSTATS, PSTATS2, saveNote='pol_degree_estimate', subPath='3.detrend'):

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
    plt.savefig(f"{experiment_dir}/{subPath}/{saveNote}_slope.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(13, 6))
    #sns.swarmplot(data=df_neural_params, x='pol_degree', y='n_resonances', ax=ax2, color='r', alpha=.25)
    sns.pointplot(data=df_neural_params, x='pol_degree', y='n_resonances', ax=ax2, color='r')
    ax2.set_title(f"Estimated Pol Degree (min res)={pol_degree_min_resonances}")
    ax2.set_ylabel('Number of Resonances', color='r')
    ax2.set_xlabel('Polynomial Degree')
    plt.savefig(f"{experiment_dir}/{subPath}/{saveNote}_resonances.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(f"{experiment_dir}/{subPath}/{saveNote}_fp.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(13, 6))
    #sns.swarmplot(data=df_neural_params, x='pol_degree', y='n_resonances', ax=ax2, color='r', alpha=.25)
    sns.pointplot(data=df_neural_params, x='pol_degree', y='fiterror', ax=ax4, color='r')
    ax2.set_title(f"Estimated Pol Degree (min res)={pol_degree_min_resonances}")
    ax2.set_ylabel('Fit Error', color='r')
    ax2.set_xlabel('Polynomial Degree')
    plt.savefig(f"{experiment_dir}/{subPath}/{saveNote}_fitError.png", dpi=300, bbox_inches='tight')
    plt.close(fig4)

    fig5, ax5 = plt.subplots(figsize=(13, 6))
    #sns.swarmplot(data=df_neural_params, x='pol_degree', y='n_resonances', ax=ax2, color='r', alpha=.25)
    sns.pointplot(data=df_neural_params, x='pol_degree', y='r2', ax=ax4, color='r')
    ax2.set_title(f"Estimated Pol Degree (min res)={pol_degree_min_resonances}")
    ax2.set_ylabel('Fit Error', color='r')
    ax2.set_xlabel('Polynomial Degree')
    plt.savefig(f"{experiment_dir}/{subPath}/{saveNote}_r2.png", dpi=300, bbox_inches='tight')
    plt.close(fig4)

    return df_neural_params, pol_degree_min_resonances, pol_degree_min_F


def init_notes(json_data,experiment_dir,sub,pipeline="TEP"):
    from pathlib import Path
    from datetime import datetime
    import platform
    import sys

    notes_path=Path(experiment_dir)/"log.txt"
    notes_path.parent.mkdir(parents=True,exist_ok=True)

    parameter_descriptions={
        "date":"Identificativo temporale assegnato all’analisi.",
        "start_time":"Timestamp numerico usato per calcolare la durata totale della pipeline.",
        "analysis_id":"Identificativo univoco dell’analisi.",
        "eeg_type":"Tipo di registrazione analizzata. Per questa pipeline è normalmente tep.",
        "subject":"Identificativo del soggetto.",
        "mainDir":"Directory principale contenente i dati del soggetto.",
        "experiment_dir":"Directory specifica in cui vengono salvati i risultati dell’analisi.",
        "sourceData":"Dataset o centro di provenienza dei dati.",
        "dataType":"Formato del file EEG caricato, per esempio ASCII o VHDR.",
        "emispheric_stimulation":"Emisfero o lato della stimolazione TMS.",
        "seedChans":"Canali EEG usati come regione di interesse per alcune feature TEP.",

        "r_sfreq":"Frequenza di ricampionamento usata nelle fasi intermedie della pipeline.",
        "sfreq":"Frequenza di campionamento originale del segnale.",
        "l_freq":"Frequenza minima del filtro passa-banda.",
        "h_freq":"Frequenza massima del filtro finale.",
        "broad_band_h_freq":"Frequenza massima del filtro preliminare broad-band.",
        "powerline_freq":"Frequenza fondamentale della rete elettrica rimossa tramite notch filter.",
        "do_filter_and_plot_raw":"Attiva il filtraggio del Raw e il salvataggio dei relativi grafici PSD.",

        "do_pulseArtifactRej":"Attiva la rimozione o sostituzione dell’artefatto immediato da impulso TMS.",
        "pulse_artifact_rej_timewindow_min":"Inizio, in secondi, della finestra contenente l’artefatto TMS.",
        "pulse_artifact_rej_timewindow_max":"Fine, in secondi, della finestra contenente l’artefatto TMS.",
        "pulse_artifact_rej_smoothingvalue":"Ampiezza della finestra usata per raccordare i bordi della correzione dell’impulso.",

        "do_clean_trials_channels":"Attiva il rilevamento e la gestione di trial e canali artefattuali.",
        "do_chan_trials_selection_automatic":"Se True usa il rilevamento automatico; se False apre la selezione manuale.",
        "bad_trials":"Indici dei trial esclusi dall’analisi.",
        "bad_channels":"Canali marcati come artefattuali.",
        "channels_marked_bad":"Canali mantenuti nell’oggetto ma marcati come bad.",
        "channels_dropped":"Canali fisicamente eliminati dall’oggetto.",
        "channel_rejection_policy":"Strategia adottata per la gestione dei canali bad.",

        "do_prepare_epochs":"Attiva la creazione delle epoche TEP finali.",
        "epochs_timewindow_min":"Inizio dell’epoca rispetto all’impulso TMS, espresso in secondi.",
        "epochs_timewindow_max":"Fine dell’epoca rispetto all’impulso TMS, espressa in secondi.",
        "baseline_cor_tmin":"Inizio della finestra usata per la baseline correction.",
        "baseline_cor_tmax":"Fine della finestra usata per la baseline correction.",
        "TEP_ID_events":"Origine o identificativo degli eventi usati per costruire le epoche.",

        "trials_wise":"Se True il detrending viene applicato separatamente a ogni trial.",
        "do_detrend":"Attiva il detrending specifico dell’artefatto post-TMS.",
        "detrend_type":"Nome generale del modello di detrending selezionato.",
        "detrend_typeOffsetRise":"Modello usato per descrivere la fase iniziale dell’offset.",
        "detrend_typeOffsetDecay":"Modello usato per descrivere il decadimento dell’offset.",
        "detrend_fitConstraint":"Indica se il modello è vincolato a un punto iniziale.",
        "detrend_polOrder_preOffset":"Ordine polinomiale usato nella finestra precedente all’offset.",
        "detrend_minTimeWindowOffset":"Inizio della finestra di modellizzazione dell’offset.",
        "detrend_maxTimeWindowOffset":"Limite massimo della finestra di ricerca dell’offset.",
        "detrendExtremeTechinque":"Criterio usato per identificare l’estremo dell’artefatto.",
        "detrend_slopeThr":"Soglia usata per identificare canali con pendenza anomala.",
        "do_detrend_onlyOffsetChans":"Se True applica il detrending solo ai canali identificati come offset.",
        "detrend_offsetCorrectionType":"Metodo usato per sostituire o correggere i campioni dell’artefatto.",
        "detrend_offsetOddSamples":"Ampiezza della regione di raccordo intorno all’offset.",
        "detrend_lag_correction":"Attiva l’eventuale correzione temporale associata alla finestra dell’offset.",
        "detrend_overall":"Attiva un detrending globale quando il detrending finestrato non viene eseguito.",
        "detrend_noWindowedOrder":"Ordine del detrending globale non finestrato.",

        "do_artifact":"Attiva l’iniezione di un artefatto sintetico per test della pipeline.",
        "do_artifact_rise":"Costante temporale di salita dell’artefatto sintetico.",
        "do_artifact_decay":"Costante temporale di decadimento dell’artefatto sintetico.",
        "do_artifact_gain":"Ampiezza dell’artefatto sintetico.",
        "do_artifact_chans":"Canali sui quali viene aggiunto l’artefatto sintetico.",

        "do_ica":"Attiva la decomposizione ICA.",
        "do_ica_continuum":"Attiva un’eventuale ICA preliminare sul continuo.",
        "do_ica_manualCheck":"Attiva la revisione manuale delle componenti ICA.",
        "do_ica_automaticRej":"Attiva l’esclusione automatica delle componenti tramite ICLabel.",
        "do_label_prob_threshold":"Probabilità minima ICLabel richiesta per escludere automaticamente una componente.",
        "do_ica_eigThresh":"Percentile impiegato dal criterio basato sugli autovalori ICA.",
        "ICA_components_tot":"Numero totale di componenti ICA stimate.",
        "ICA_excludedComponents":"Componenti ICA escluse definitivamente.",
        "ICA_includedComponents_tot":"Numero di componenti ICA mantenute.",
        "ICA_autoExcludedComponents":"Componenti proposte automaticamente per l’esclusione.",
        "ICA_manualAddedComponents":"Componenti aggiunte manualmente alla lista di esclusione.",
        "ICA_manualRecoveredComponents":"Componenti inizialmente escluse e successivamente recuperate.",

        "rest_features_fmin":"Frequenza minima delle feature spettrali, se usata anche nella pipeline TEP.",
        "rest_features_fmax":"Frequenza massima delle feature spettrali, se usata anche nella pipeline TEP.",
        "pcist_baseline_window_ms":"Finestra pre-stimolo usata da PCIst per stimare la dinamica di baseline.",
        "pcist_response_window_ms":"Finestra post-stimolo usata per calcolare PCIst.",
        "pcist_k":"Parametro di soglia PCIst che regola il criterio di significatività delle transizioni.",
        "pcist_min_snr":"Rapporto segnale-rumore minimo richiesto per mantenere una componente.",
        "pcist_max_var":"Percentuale massima di varianza cumulativa mantenuta nella decomposizione.",
        "pcist_embed":"Attiva l’embedding temporale previsto dall’implementazione PCIst.",
        "pcist_n_steps":"Numero di soglie testate durante il calcolo PCIst.",
        "pcist_baseline_corr":"Attiva la baseline correction interna alla funzione PCIst.",
        "do_pcist":"Attiva il calcolo della Perturbational Complexity Index state-transition.",
        "PCIst":"Valore finale PCIst calcolato sul TEP medio.",
        "PCIst_n_dims":"Numero di componenti SVD significative mantenute da PCIst.",
        "PCIst_dNST":"Contributo alla complessità di ogni componente SVD.",

        "feature_extraction_dir":"Directory contenente le feature post-ICA.",
        "feature_extraction_completed":"Indica se l’estrazione delle feature è stata completata.",
        "feature_extraction_sections":"Elenco delle sezioni di feature effettivamente calcolate."
    }

    with open(notes_path,"w",encoding="utf-8") as file:
        file.write("="*80+"\n")
        file.write("TMSPATH ANALYSIS NOTES\n")
        file.write("="*80+"\n\n")
        file.write(f"Subject: {sub}\n")
        file.write(f"Pipeline: {pipeline}\n")
        file.write(f"Analysis started: {datetime.now().isoformat(timespec='seconds')}\n")
        file.write(f"Experiment directory: {Path(experiment_dir).resolve()}\n")
        file.write(f"Python: {sys.version.split()[0]}\n")
        file.write(f"Operating system: {platform.platform()}\n\n")

        file.write("-"*80+"\n")
        file.write("CONFIGURATION PARAMETERS\n")
        file.write("-"*80+"\n\n")

        for key,value in json_data.items():
            description=parameter_descriptions.get(
                key,
                "Parametro della pipeline senza descrizione specifica."
            )
            file.write(f"{key}\n")
            file.write(f"  Value: {value}\n")
            file.write(f"  Meaning: {description}\n\n")

        file.write("-"*80+"\n")
        file.write("PROCEDURAL LOG\n")
        file.write("-"*80+"\n\n")

    json_data["notes_path"]=str(notes_path)

    return notes_path,json_data

def log_note(experiment_dir,message,level="INFO",details=None):
    from pathlib import Path
    from datetime import datetime

    notes_path=Path(experiment_dir)/"log.txt"
    notes_path.parent.mkdir(parents=True,exist_ok=True)

    timestamp=datetime.now().isoformat(timespec="seconds")

    with open(notes_path,"a",encoding="utf-8") as file:
        file.write(f"[{timestamp}] [{level}] {message}\n")

        if details is not None:
            if isinstance(details,dict):
                for key,value in details.items():
                    file.write(f"    {key}: {value}\n")
            else:
                file.write(f"    {details}\n")

        file.write("\n")

def start_log_step(experiment_dir,step_name,details=None):
    import time

    log_note(
        experiment_dir,
        step_name,
        level="START",
        details=details
    )

    return time.time()

def end_log_step(experiment_dir,step_name,start_time,details=None):
    import time

    elapsed=time.time()-start_time

    final_details={} if details is None else dict(details)
    final_details["elapsed_sec"]=round(elapsed,3)

    log_note(
        experiment_dir,
        step_name,
        level="DONE",
        details=final_details
    )

    return elapsed

def finalize_notes(json_data,experiment_dir,sub,status="completed"):
    from pathlib import Path
    from datetime import datetime
    import time

    try:
        elapsed=time.time()-float(json_data["start_time"])
    except (KeyError,TypeError,ValueError):
        elapsed=None

    details={
        "subject":sub,
        "status":status,
        "finished":datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec":round(elapsed,2) if elapsed is not None else "unknown",
        "source_data":json_data.get("sourceData"),
        "data_type":json_data.get("dataType"),
        "stimulation_side":json_data.get("emispheric_stimulation"),
        "trials_total":json_data.get("trials_tot"),
        "trials_selected":json_data.get("trials_selected"),
        "bad_trials":json_data.get("bad_trials",[]),
        "bad_channels":json_data.get("bad_channels",[]),
        "offset_channels":json_data.get("offsetChans",[]),
        "detrend_performed":json_data.get("do_detrend"),
        "detrend_model_rise":json_data.get("detrend_typeOffsetRise"),
        "detrend_model_decay":json_data.get("detrend_typeOffsetDecay"),
        "detrend_mse":json_data.get("detrend_MSE"),
        "ICA_components_total":json_data.get("ICA_components_tot"),
        "ICA_components_excluded":json_data.get("ICA_excludedComponents",[]),
        "ICA_components_included":json_data.get("ICA_includedComponents_tot"),
        "PCIst":json_data.get("PCIst"),
        "PCIst_n_dims":json_data.get("PCIst_n_dims"),
        "feature_directory":json_data.get("feature_extraction_dir"),
        "postICA_directory":json_data.get("postICA_dir")
    }

    log_note(
        experiment_dir,
        "TEP analysis finished",
        level="SUMMARY",
        details=details
    )

    notes_path=Path(experiment_dir)/"log.txt"

    with open(notes_path,"a",encoding="utf-8") as file:
        file.write("="*80+"\n")
        file.write("END OF TEP ANALYSIS\n")
        file.write("="*80+"\n")

    return notes_path



def compute_tep_fingerprint(
    postICA_final,
    json_data,
    experiment_dir,
    sub,
    ntop=None,
    p1_window_ms=None,
    search_end_ms=None,
    min_peak_distance_ms=None,
    exclude_lateral_channels=None,
    save=True
):
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import mne
    from pathlib import Path
    from scipy.signal import find_peaks

    sub=str(sub).strip()

    ntop=int(
        json_data.get(
            "tep_fingerprint_ntop",
            4 if ntop is None else ntop
        )
    )

    p1_window_ms=tuple(
        json_data.get(
            "tep_fingerprint_p1_window_ms",
            (10,40) if p1_window_ms is None else p1_window_ms
        )
    )

    search_end_ms=float(
        json_data.get(
            "tep_fingerprint_search_end_ms",
            200 if search_end_ms is None else search_end_ms
        )
    )

    min_peak_distance_ms=float(
        json_data.get(
            "tep_fingerprint_min_peak_distance_ms",
            14 if min_peak_distance_ms is None else min_peak_distance_ms
        )
    )

    if ntop<1:
        raise ValueError("ntop deve essere almeno 1.")

    if exclude_lateral_channels is None:
        exclude_lateral_channels=json_data.get(
            "tep_fingerprint_exclude_channels",
            [
                "Fp1","Fpz","Fp2","F7","F8",
                "FT7","FT8","T7","T8",
                "TP7","TP8","TP9","TP10",
                "P7","P8","Iz"
            ]
        )

    epochs=postICA_final.copy().pick("eeg")
    evoked=epochs.average()

    data=np.asarray(
        evoked.get_data(),
        dtype=float
    )*1e6

    times_ms=np.asarray(
        evoked.times,
        dtype=float
    )*1000.0

    sfreq=float(evoked.info["sfreq"])

    excluded_channels=[
        channel
        for channel in exclude_lateral_channels
        if channel in evoked.ch_names
    ]

    allowed_channels=[
        channel
        for channel in evoked.ch_names
        if channel not in excluded_channels
    ]

    if len(allowed_channels)<ntop:
        raise ValueError(
            f"Canali analizzabili={len(allowed_channels)}, "
            f"ma ntop={ntop}."
        )

    p1_start=float(p1_window_ms[0])
    p1_stop=float(p1_window_ms[1])

    if p1_stop<=p1_start:
        raise ValueError(
            f"Finestra P1 non valida: {p1_window_ms}"
        )

    if times_ms.min()>p1_start or times_ms.max()<p1_stop:
        raise ValueError(
            f"Finestra P1 {p1_window_ms} ms non contenuta nei dati "
            f"[{times_ms.min():.3f},{times_ms.max():.3f}] ms."
        )

    search_end_ms=min(
        search_end_ms,
        float(times_ms.max())
    )

    if search_end_ms<=p1_stop:
        raise ValueError(
            "tep_fingerprint_search_end_ms deve essere "
            "successivo alla finestra P1."
        )

    p1_mask=(
        (times_ms>=p1_start)
        &(times_ms<=p1_stop)
    )

    if np.sum(p1_mask)<3:
        raise ValueError(
            "Numero insufficiente di campioni nella finestra P1."
        )

    min_peak_distance_samples=max(
        1,
        int(
            round(
                min_peak_distance_ms
                *sfreq
                /1000.0
            )
        )
    )

    def first_valid_peak(
        signal,
        candidate_indices,
        positive
    ):
        if candidate_indices.size<3:
            return None

        candidate_signal=signal[
            candidate_indices
        ]

        peaks,_=find_peaks(
            candidate_signal if positive else -candidate_signal,
            distance=min_peak_distance_samples
        )

        if len(peaks)==0:
            return None

        return int(
            candidate_indices[
                int(peaks[0])
            ]
        )

    def detect_channel_peaks(
        signal,
        channel
    ):
        p1_indices=np.where(
            p1_mask
        )[0]

        first_index=int(
            p1_indices[0]
        )

        last_index=int(
            p1_indices[-1]
        )

        midpoint_ms=float(
            np.mean(
                p1_window_ms
            )
        )

        midpoint_index=int(
            np.argmin(
                np.abs(
                    times_ms-midpoint_ms
                )
            )
        )

        edge_mean=float(
            np.mean([
                signal[first_index],
                signal[last_index]
            ])
        )

        midpoint_value=float(
            signal[midpoint_index]
        )

        p1_is_positive=bool(
            midpoint_value>edge_mean
        )

        p1_segment=signal[
            p1_indices
        ]

        if p1_is_positive:
            p1_index=int(
                p1_indices[
                    int(np.argmax(p1_segment))
                ]
            )
        else:
            p1_index=int(
                p1_indices[
                    int(np.argmin(p1_segment))
                ]
            )

        p2_indices=np.where(
            (
                times_ms
                >=times_ms[p1_index]
                +min_peak_distance_ms
            )
            &(times_ms<=search_end_ms)
        )[0]

        p2_index=first_valid_peak(
            signal=signal,
            candidate_indices=p2_indices,
            positive=not p1_is_positive
        )

        if p2_index is None:
            return None

        p3_indices=np.where(
            (
                times_ms
                >=times_ms[p2_index]
                +min_peak_distance_ms
            )
            &(times_ms<=search_end_ms)
        )[0]

        p3_index=first_valid_peak(
            signal=signal,
            candidate_indices=p3_indices,
            positive=p1_is_positive
        )

        if p3_index is None:
            return None

        lp1=float(
            times_ms[p1_index]
        )

        lp2=float(
            times_ms[p2_index]
        )

        lp3=float(
            times_ms[p3_index]
        )

        vp1=float(
            signal[p1_index]
        )

        vp2=float(
            signal[p2_index]
        )

        vp3=float(
            signal[p3_index]
        )

        dt1=float(
            lp2-lp1
        )

        dt2=float(
            lp3-lp2
        )

        ipi=float(
            lp3-lp1
        )

        if dt1<=0 or dt2<=0 or ipi<=0:
            return None

        ap1_p2=float(
            abs(vp1-vp2)
        )

        ap2_p3=float(
            abs(vp2-vp3)
        )

        sp1_p2=float(
            (vp2-vp1)/dt1
        )

        sp2_p3=float(
            (vp3-vp2)/dt2
        )

        ipi_hz=float(
            1000.0/ipi
        )

        return {
            "subject":sub,
            "subject_id":sub,
            "channel":str(channel),
            "P1_polarity":(
                "positive"
                if p1_is_positive
                else "negative"
            ),
            "P1_amplitude_uV":vp1,
            "P2_amplitude_uV":vp2,
            "P3_amplitude_uV":vp3,
            "LP1_ms":lp1,
            "LP2_ms":lp2,
            "LP3_ms":lp3,
            "AP1_P2_uV":ap1_p2,
            "AP2_P3_uV":ap2_p3,
            "SP1_P2_uV_per_ms":sp1_p2,
            "SP2_P3_uV_per_ms":sp2_p3,
            "abs_SP1_P2_uV_per_ms":abs(sp1_p2),
            "abs_SP2_P3_uV_per_ms":abs(sp2_p3),
            "IPI_ms":ipi,
            "IPI_Hz":ipi_hz,
            "P1_sample":int(p1_index),
            "P2_sample":int(p2_index),
            "P3_sample":int(p3_index)
        }

    rows=[]

    for channel in allowed_channels:
        channel_index=evoked.ch_names.index(
            channel
        )

        channel_result=detect_channel_peaks(
            signal=data[channel_index],
            channel=channel
        )

        if channel_result is not None:
            rows.append(
                channel_result
            )

    df_channels=pd.DataFrame(
        rows
    )

    if df_channels.empty:
        raise RuntimeError(
            "P1, P2 e P3 non sono stati identificati "
            "in alcun canale."
        )

    df_channels=df_channels.sort_values(
        "AP1_P2_uV",
        ascending=False
    ).reset_index(
        drop=True
    )

    adjacency,adjacency_channels=(
        mne.channels.find_ch_adjacency(
            evoked.info,
            ch_type="eeg"
        )
    )

    adjacency_channels=[
        str(channel)
        for channel in adjacency_channels
    ]

    channel_to_adjacency_index={
        channel:index
        for index,channel in enumerate(
            adjacency_channels
        )
    }

    ranked_channels=[
        str(channel)
        for channel in df_channels["channel"]
        if channel in channel_to_adjacency_index
    ]

    if not ranked_channels:
        raise RuntimeError(
            "Nessun canale rilevato è presente "
            "nella matrice di adiacenza."
        )

    seed_channel=ranked_channels[0]
    roi_channels=[seed_channel]

    while len(roi_channels)<ntop:
        candidates=[]

        for channel in ranked_channels:
            if channel in roi_channels:
                continue

            candidate_index=channel_to_adjacency_index[
                channel
            ]

            connected=any(
                bool(
                    adjacency[
                        candidate_index,
                        channel_to_adjacency_index[
                            selected_channel
                        ]
                    ]
                )
                for selected_channel in roi_channels
                if selected_channel
                in channel_to_adjacency_index
            )

            if connected:
                candidates.append(
                    channel
                )

        if not candidates:
            print(
                "⚠️ Impossibile espandere il cluster "
                f"connesso oltre {len(roi_channels)} canali."
            )
            break

        roi_channels.append(
            candidates[0]
        )

    if len(roi_channels)<ntop:
        print(
            f"⚠️ ROI selezionata con {len(roi_channels)} "
            f"canali invece di ntop={ntop}."
        )

    df_channels["ROI_selected"]=(
        df_channels["channel"].isin(
            roi_channels
        )
    )

    df_channels["AP1_P2_rank"]=(
        np.arange(
            len(df_channels)
        )+1
    )

    df_roi=df_channels[
        df_channels["ROI_selected"]
    ].copy()

    if df_roi.empty:
        raise RuntimeError(
            "La ROI del fingerprint è vuota."
        )

    numeric_features=[
        "LP1_ms",
        "LP2_ms",
        "LP3_ms",
        "AP1_P2_uV",
        "AP2_P3_uV",
        "SP1_P2_uV_per_ms",
        "SP2_P3_uV_per_ms",
        "abs_SP1_P2_uV_per_ms",
        "abs_SP2_P3_uV_per_ms",
        "IPI_ms",
        "IPI_Hz"
    ]

    fingerprint={
        feature:float(
            df_roi[feature].mean()
        )
        for feature in numeric_features
    }

    reference={
        "citation":(
            "Hassan G, Gaglioti G, Furregoni G, et al. "
            "Temporal fingerprints of TMS-evoked potentials "
            "across thalamocortical circuits. bioRxiv. 2026."
        ),
        "title":(
            "Temporal fingerprints of TMS-evoked "
            "potentials across thalamocortical circuits"
        ),
        "doi":"10.64898/2026.06.29.734769",
        "publication_status":"bioRxiv preprint",
        "license":"CC-BY 4.0"
    }

    fingerprint.update({
        "subject":sub,
        "subject_id":sub,
        "seed_channel":seed_channel,
        "roi_channels":list(
            roi_channels
        ),
        "ntop_requested":int(
            ntop
        ),
        "ntop_selected":int(
            len(roi_channels)
        ),
        "n_channels_available":int(
            len(allowed_channels)
        ),
        "n_channels_with_valid_peaks":int(
            len(df_channels)
        ),
        "excluded_channels":list(
            excluded_channels
        ),
        "P1_window_ms":[
            p1_start,
            p1_stop
        ],
        "search_end_ms":float(
            search_end_ms
        ),
        "min_peak_distance_ms":float(
            min_peak_distance_ms
        ),
        "ROI_selection":(
            "Maximum AP1-P2 seed followed by highest-ranked "
            "spatially adjacent channels"
        ),
        "reference":reference
    })

    out_dir=(
        Path(experiment_dir)
        /"5.Extra"
        /"FE"
        /"Fingerprint"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    channel_csv=(
        out_dir
        /f"{sub}_TEP_fingerprint_channels.csv"
    )

    roi_csv=(
        out_dir
        /f"{sub}_TEP_fingerprint_ROI.csv"
    )

    summary_json=(
        out_dir
        /f"{sub}_TEP_fingerprint_summary.json"
    )

    summary_csv=(
        out_dir
        /f"{sub}_TEP_fingerprint_summary.csv"
    )

    waveform_png=(
        out_dir
        /f"{sub}_TEP_fingerprint_ROI_waveforms.png"
    )

    topomap_png=(
        out_dir
        /f"{sub}_TEP_fingerprint_AP1_P2_topomap.png"
    )

    if save:
        df_channels.to_csv(
            channel_csv,
            index=False
        )

        df_roi.to_csv(
            roi_csv,
            index=False
        )

        flat_summary={
            key:value
            for key,value in fingerprint.items()
            if not isinstance(
                value,
                (list,dict)
            )
        }

        flat_summary["roi_channels"]=";".join(
            roi_channels
        )

        flat_summary["excluded_channels"]=";".join(
            excluded_channels
        )

        flat_summary["reference_doi"]=reference[
            "doi"
        ]

        flat_summary["reference_citation"]=reference[
            "citation"
        ]

        pd.DataFrame([
            flat_summary
        ]).to_csv(
            summary_csv,
            index=False
        )

        with open(
            summary_json,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                fingerprint,
                file,
                indent=4,
                ensure_ascii=False
            )

        fig,ax=plt.subplots(
            figsize=(11,6)
        )

        for channel in roi_channels:
            channel_index=evoked.ch_names.index(
                channel
            )

            row=df_roi[
                df_roi["channel"]==channel
            ].iloc[0]

            ax.plot(
                times_ms,
                data[channel_index],
                linewidth=1.5,
                label=channel
            )

            ax.scatter(
                [
                    row["LP1_ms"],
                    row["LP2_ms"],
                    row["LP3_ms"]
                ],
                [
                    row["P1_amplitude_uV"],
                    row["P2_amplitude_uV"],
                    row["P3_amplitude_uV"]
                ],
                s=50
            )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1
        )

        ax.axvspan(
            p1_start,
            p1_stop,
            alpha=0.15
        )

        ax.set_xlim(
            max(
                float(times_ms.min()),
                -100
            ),
            search_end_ms
        )

        ax.set_xlabel(
            "Time [ms]"
        )

        ax.set_ylabel(
            "Amplitude [µV]"
        )

        ax.set_title(
            f"{sub} TEP temporal fingerprint\n"
            f"ROI={roi_channels}"
        )

        ax.legend(
            loc="best"
        )

        fig.tight_layout()

        fig.savefig(
            waveform_png,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        topomap_values=np.zeros(
            len(evoked.ch_names),
            dtype=float
        )

        valid_topomap_mask=np.zeros(
            len(evoked.ch_names),
            dtype=bool
        )

        for _,row in df_channels.iterrows():
            channel_index=evoked.ch_names.index(
                row["channel"]
            )

            topomap_values[
                channel_index
            ]=float(
                row["AP1_P2_uV"]
            )

            valid_topomap_mask[
                channel_index
            ]=True

        roi_mask=np.asarray(
            [
                channel in roi_channels
                for channel in evoked.ch_names
            ],
            dtype=bool
        )

        if np.any(valid_topomap_mask):
            fig,ax=plt.subplots(
                figsize=(7,6)
            )

            mne.viz.plot_topomap(
                topomap_values,
                evoked.info,
                axes=ax,
                show=False,
                contours=8,
                mask=roi_mask,
                mask_params={
                    "marker":"o",
                    "markerfacecolor":"none",
                    "markeredgecolor":"black",
                    "linewidth":2,
                    "markersize":10
                }
            )

            ax.set_title(
                f"{sub} AP1-P2 [µV]\n"
                f"ROI={roi_channels}"
            )

            fig.tight_layout()

            fig.savefig(
                topomap_png,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

    json_data["subject"]=sub
    json_data["subject_id"]=sub
    json_data["TEP_fingerprint_computed"]=True
    json_data["TEP_fingerprint_subject"]=sub
    json_data["TEP_fingerprint_ntop_requested"]=int(
        ntop
    )
    json_data["TEP_fingerprint_ntop_selected"]=int(
        len(roi_channels)
    )
    json_data["TEP_fingerprint_seed_channel"]=(
        seed_channel
    )
    json_data["TEP_fingerprint_ROI_channels"]=list(
        roi_channels
    )
    json_data["TEP_fingerprint_excluded_channels"]=list(
        excluded_channels
    )
    json_data["TEP_fingerprint_P1_window_ms"]=[
        p1_start,
        p1_stop
    ]
    json_data["TEP_fingerprint_search_end_ms"]=float(
        search_end_ms
    )
    json_data[
        "TEP_fingerprint_min_peak_distance_ms"
    ]=float(
        min_peak_distance_ms
    )
    json_data["TEP_fingerprint_summary"]=fingerprint
    json_data["TEP_fingerprint_reference"]=reference
    json_data["TEP_fingerprint_reference_doi"]=(
        reference["doi"]
    )
    json_data["TEP_fingerprint_output_dir"]=str(
        out_dir
    )
    json_data["TEP_fingerprint_channels_csv"]=str(
        channel_csv
    )
    json_data["TEP_fingerprint_ROI_csv"]=str(
        roi_csv
    )
    json_data["TEP_fingerprint_summary_csv"]=str(
        summary_csv
    )
    json_data["TEP_fingerprint_summary_json"]=str(
        summary_json
    )
    json_data["TEP_fingerprint_waveform_png"]=str(
        waveform_png
    )
    json_data["TEP_fingerprint_topomap_png"]=str(
        topomap_png
    )

    with open(
        Path(experiment_dir)/f"{sub}_pars.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            make_json_serializable(
                json_data
            ),
            file,
            indent=4,
            sort_keys=True
        )

    print("✅ TEP temporal fingerprint completato")
    print(f"   Subject: {sub}")
    print(f"   Seed channel: {seed_channel}")
    print(f"   ROI channels: {roi_channels}")
    print(
        f"   ROI size: {len(roi_channels)} / "
        f"{ntop} richiesta"
    )
    print(
        f"   LP1: {fingerprint['LP1_ms']:.3f} ms"
    )
    print(
        f"   LP2: {fingerprint['LP2_ms']:.3f} ms"
    )
    print(
        f"   LP3: {fingerprint['LP3_ms']:.3f} ms"
    )
    print(
        f"   AP1-P2: "
        f"{fingerprint['AP1_P2_uV']:.3f} µV"
    )
    print(
        f"   AP2-P3: "
        f"{fingerprint['AP2_P3_uV']:.3f} µV"
    )
    print(
        f"   |SP1-P2|: "
        f"{fingerprint['abs_SP1_P2_uV_per_ms']:.3f} "
        "µV/ms"
    )
    print(
        f"   |SP2-P3|: "
        f"{fingerprint['abs_SP2_P3_uV_per_ms']:.3f} "
        "µV/ms"
    )
    print(
        f"   IPI: {fingerprint['IPI_ms']:.3f} ms"
    )
    print(
        f"   IPIHz: {fingerprint['IPI_Hz']:.3f} Hz"
    )
    print(
        "📚 Reference: Hassan G, Gaglioti G, "
        "Furregoni G, et al. Temporal fingerprints "
        "of TMS-evoked potentials across "
        "thalamocortical circuits. bioRxiv, 2026."
    )
    print(
        "🔗 DOI: 10.64898/2026.06.29.734769"
    )
    print(
        f"   Results: {out_dir}"
    )

    return (
        fingerprint,
        df_channels,
        df_roi,
        json_data
    )




def compute_tep_random_trigger_pcist_null(
    raw_continuous,
    real_events,
    reference_epochs,
    json_data,
    experiment_dir,
    sub,
    n_replicates=None,
    random_seed=None,
    exclusion_ms=None,
    min_interval_ms=None,
    save=True
):
    import json
    import tempfile
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import mne
    from pathlib import Path

    n_replicates=int(
        json_data.get(
            "tep_random_pcist_replicates",
            50 if n_replicates is None else n_replicates
        )
    )

    random_seed=int(
        json_data.get(
            "tep_random_pcist_seed",
            42 if random_seed is None else random_seed
        )
    )

    exclusion_ms=float(
        json_data.get(
            "tep_random_pcist_exclusion_ms",
            500.0 if exclusion_ms is None else exclusion_ms
        )
    )

    min_interval_ms=float(
        json_data.get(
            "tep_random_pcist_min_interval_ms",
            500.0 if min_interval_ms is None else min_interval_ms
        )
    )

    raw=raw_continuous.copy().pick("eeg").load_data()

    sfreq=float(raw.info["sfreq"])
    n_times=int(raw.n_times)
    first_samp=int(raw.first_samp)

    tmin=float(reference_epochs.tmin)
    tmax=float(reference_epochs.tmax)
    n_events=int(len(reference_epochs))

    if n_events<1:
        raise ValueError(
            "reference_epochs non contiene epoche."
        )

    start_margin=int(
        np.ceil(
            abs(tmin)*sfreq
        )
    )

    stop_margin=int(
        np.ceil(
            tmax*sfreq
        )
    )

    exclusion_samples=int(
        round(
            exclusion_ms*sfreq/1000.0
        )
    )

    min_interval_samples=int(
        round(
            min_interval_ms*sfreq/1000.0
        )
    )

    valid=np.ones(
        n_times,
        dtype=bool
    )

    valid[:start_margin]=False
    valid[max(0,n_times-stop_margin):]=False

    for annotation in raw.annotations:
        if not str(
            annotation["description"]
        ).startswith("BAD"):
            continue

        start=int(
            np.floor(
                float(annotation["onset"])*sfreq
            )
        )-stop_margin

        stop=int(
            np.ceil(
                (
                    float(annotation["onset"])
                    +float(annotation["duration"])
                )*sfreq
            )
        )+start_margin

        start=max(
            0,
            start
        )

        stop=min(
            n_times,
            stop
        )

        valid[start:stop]=False

    real_event_samples=np.asarray(
        real_events[:,0],
        dtype=int
    )-first_samp

    for event_sample in real_event_samples:
        start=max(
            0,
            event_sample-exclusion_samples
        )

        stop=min(
            n_times,
            event_sample+exclusion_samples+1
        )

        valid[start:stop]=False

    candidate_samples=np.flatnonzero(
        valid
    )

    if candidate_samples.size<n_events:
        raise ValueError(
            "Campioni validi insufficienti per generare "
            f"{n_events} trigger casuali."
        )

    rng=np.random.default_rng(
        random_seed
    )

    experiment_dir=Path(
        experiment_dir
    ).expanduser().resolve()

    out_dir=(
        experiment_dir
        /"5.Extra"
        /"FE"
        /"PCIst"
        /"RandomTriggers"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    def sample_random_triggers():
        shuffled=rng.permutation(
            candidate_samples
        )

        selected=[]

        for sample in shuffled:
            if all(
                abs(sample-existing)
                >=min_interval_samples
                for existing in selected
            ):
                selected.append(
                    int(sample)
                )

                if len(selected)==n_events:
                    break

        if len(selected)<n_events:
            raise RuntimeError(
                "Impossibile generare abbastanza trigger casuali "
                "rispettando la distanza minima."
            )

        return np.sort(
            np.asarray(
                selected,
                dtype=int
            )
        )

    rows=[]

    for replicate in range(
        n_replicates
    ):
        random_samples=sample_random_triggers()

        random_events=np.column_stack([
            random_samples+first_samp,
            np.zeros(
                n_events,
                dtype=int
            ),
            np.full(
                n_events,
                999,
                dtype=int
            )
        ])

        random_epochs=mne.Epochs(
            raw,
            random_events,
            event_id={"RANDOM_TRIGGER":999},
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            detrend=None,
            preload=True,
            reject_by_annotation=True,
            event_repeated="drop",
            verbose=False
        )

        random_epochs=random_epochs.pick(
            "eeg"
        )

        if len(random_epochs)==0:
            rows.append({
                "subject":str(sub),
                "replicate":int(replicate+1),
                "n_requested":int(n_events),
                "n_kept":0,
                "PCIst":np.nan,
                "n_dims":np.nan,
                "status":"no_epochs"
            })

            continue

        with tempfile.TemporaryDirectory(
            prefix=(
                f"tep_random_pcist_"
                f"{replicate+1:03d}_"
            )
        ) as temporary_directory:

            try:
                (
                    pci_value,
                    pci_result,
                    _
                )=compute_pcist(
                    postICA_final=random_epochs,
                    json_data=json_data.copy(),
                    experiment_dir=temporary_directory,
                    sub=(
                        f"{sub}_RANDOM_TRIGGER_"
                        f"{replicate+1:03d}"
                    )
                )

                rows.append({
                    "subject":str(sub),
                    "replicate":int(replicate+1),
                    "n_requested":int(n_events),
                    "n_kept":int(len(random_epochs)),
                    "PCIst":float(pci_value),
                    "n_dims":int(
                        pci_result["n_dims"]
                    ),
                    "first_trigger_sec":float(
                        random_samples[0]/sfreq
                    ),
                    "last_trigger_sec":float(
                        random_samples[-1]/sfreq
                    ),
                    "status":"ok"
                })

            except Exception as error:
                rows.append({
                    "subject":str(sub),
                    "replicate":int(replicate+1),
                    "n_requested":int(n_events),
                    "n_kept":int(len(random_epochs)),
                    "PCIst":np.nan,
                    "n_dims":np.nan,
                    "first_trigger_sec":float(
                        random_samples[0]/sfreq
                    ),
                    "last_trigger_sec":float(
                        random_samples[-1]/sfreq
                    ),
                    "status":repr(error)
                })

    df=pd.DataFrame(
        rows
    )

    values=(
        df["PCIst"]
        .replace(
            [np.inf,-np.inf],
            np.nan
        )
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if values.size==0:
        raise RuntimeError(
            "Nessuna replica random-trigger PCIst valida."
        )

    observed_value=None

    try:
        with tempfile.TemporaryDirectory(
            prefix="tep_observed_pcist_"
        ) as temporary_directory:

            (
                observed_value,
                observed_result,
                _
            )=compute_pcist(
                postICA_final=reference_epochs,
                json_data=json_data.copy(),
                experiment_dir=temporary_directory,
                sub=f"{sub}_OBSERVED"
            )

            observed_value=float(
                observed_value
            )

    except Exception as error:
        print(
            f"⚠️ PCIst osservata non ricalcolata: {error}"
        )

    null_mean=float(
        np.mean(values)
    )

    null_std=float(
        np.std(
            values,
            ddof=1
        )
    ) if values.size>1 else 0.0

    null_p95=float(
        np.percentile(
            values,
            95
        )
    )

    empirical_p=(
        float(
            (
                1+np.sum(
                    values>=observed_value
                )
            )/(
                values.size+1
            )
        )
        if observed_value is not None
        else np.nan
    )

    summary={
        "subject":str(sub),
        "null_model":(
            "Random trigger relocation on the unchanged "
            "continuous TEP signal"
        ),
        "n_replicates_requested":int(
            n_replicates
        ),
        "n_replicates_valid":int(
            values.size
        ),
        "n_events_per_replicate":int(
            n_events
        ),
        "random_seed":int(
            random_seed
        ),
        "exclusion_from_real_triggers_ms":float(
            exclusion_ms
        ),
        "minimum_random_trigger_interval_ms":float(
            min_interval_ms
        ),
        "observed_pcist":observed_value,
        "null_mean":null_mean,
        "null_std":null_std,
        "null_median":float(
            np.median(values)
        ),
        "null_min":float(
            np.min(values)
        ),
        "null_max":float(
            np.max(values)
        ),
        "null_p95":null_p95,
        "empirical_p_upper_tail":empirical_p
    }

    replicates_csv=(
        out_dir
        /f"{sub}_TEP_random_trigger_PCIst_replicates.csv"
    )

    summary_csv=(
        out_dir
        /f"{sub}_TEP_random_trigger_PCIst_summary.csv"
    )

    summary_json=(
        out_dir
        /f"{sub}_TEP_random_trigger_PCIst_summary.json"
    )

    values_npy=(
        out_dir
        /f"{sub}_TEP_random_trigger_PCIst_values.npy"
    )

    figure_png=(
        out_dir
        /f"{sub}_TEP_random_trigger_PCIst_distribution.png"
    )

    if save:
        df.to_csv(
            replicates_csv,
            index=False
        )

        pd.DataFrame([
            summary
        ]).to_csv(
            summary_csv,
            index=False
        )

        with open(
            summary_json,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                summary,
                file,
                indent=4,
                ensure_ascii=False
            )

        np.save(
            values_npy,
            values
        )

        fig,ax=plt.subplots(
            figsize=(9,6)
        )

        ax.hist(
            values,
            bins=min(
                20,
                max(
                    5,
                    int(
                        np.sqrt(
                            values.size
                        )
                    )
                )
            )
        )

        ax.axvline(
            null_mean,
            linestyle="--",
            linewidth=2,
            label=f"Null mean={null_mean:.3f}"
        )

        ax.axvline(
            null_p95,
            linestyle=":",
            linewidth=2,
            label=f"Null p95={null_p95:.3f}"
        )

        if observed_value is not None:
            ax.axvline(
                observed_value,
                linewidth=3,
                label=f"Observed={observed_value:.3f}"
            )

        ax.set_xlabel(
            "PCIst"
        )

        ax.set_ylabel(
            "Random-trigger replicates"
        )

        ax.set_title(
            f"{sub} TEP random-trigger PCIst null"
        )

        ax.legend()
        fig.tight_layout()

        fig.savefig(
            figure_png,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

    json_data[
        "TEP_random_trigger_PCIst_computed"
    ]=True

    json_data[
        "TEP_random_trigger_PCIst_replicates"
    ]=int(
        n_replicates
    )

    json_data[
        "TEP_random_trigger_PCIst_seed"
    ]=int(
        random_seed
    )

    json_data[
        "TEP_random_trigger_PCIst_null_mean"
    ]=null_mean

    json_data[
        "TEP_random_trigger_PCIst_null_std"
    ]=null_std

    json_data[
        "TEP_random_trigger_PCIst_null_p95"
    ]=null_p95

    json_data[
        "TEP_random_trigger_PCIst_observed"
    ]=observed_value

    json_data[
        "TEP_random_trigger_PCIst_empirical_p"
    ]=empirical_p

    json_data[
        "TEP_random_trigger_PCIst_output_dir"
    ]=str(
        out_dir
    )

    print(
        "✅ TEP random-trigger PCIst null completata"
    )
    print(
        f"   Segnale continuo invariato"
    )
    print(
        f"   Trigger reali esclusi ±{exclusion_ms:.1f} ms"
    )
    print(
        f"   Eventi per replica: {n_events}"
    )
    print(
        f"   Repliche valide: {values.size}"
    )
    print(
        f"   Null PCIst: {null_mean:.3f} ± {null_std:.3f}"
    )
    print(
        f"   Null p95: {null_p95:.3f}"
    )
    print(
        f"   Observed: {observed_value}"
    )
    print(
        f"   Empirical p: {empirical_p}"
    )

    return df,summary,values,json_data

from pathlib import Path
import numpy as np
import mne

try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks=None

def _auto_joint_times(evoked,tmin=0.02,tmax=0.30,n_peaks=4,min_distance_ms=20,picks="eeg"):
    evo=evoked.copy().pick(picks)
    times=evo.times
    mask=(times>=tmin)&(times<=tmax)
    if not np.any(mask):
        raise ValueError("Nessun campione nella finestra richiesta per selezionare i picchi.")
    data=evo.data[:,mask]
    gfp=np.std(data,axis=0)
    sfreq=float(evo.info["sfreq"])
    min_distance=max(1,int(round(min_distance_ms*sfreq/1000.0)))
    if find_peaks is not None:
        peaks,_=find_peaks(gfp,distance=min_distance)
    else:
        peaks=np.array([],dtype=int)
    if len(peaks)==0:
        idx_sorted=np.argsort(gfp)[::-1]
        chosen=[]
        for idx in idx_sorted:
            if all(abs(idx-c)>=min_distance for c in chosen):
                chosen.append(idx)
            if len(chosen)>=n_peaks:
                break
        peaks=np.array(sorted(chosen),dtype=int)
    else:
        peaks=peaks[np.argsort(gfp[peaks])[::-1]]
        chosen=[]
        for idx in peaks:
            if all(abs(idx-c)>=min_distance for c in chosen):
                chosen.append(idx)
            if len(chosen)>=n_peaks:
                break
        peaks=np.array(sorted(chosen),dtype=int)
    if len(peaks)<n_peaks:
        idx_sorted=np.argsort(gfp)[::-1]
        chosen=list(peaks)
        for idx in idx_sorted:
            if all(abs(idx-c)>=min_distance for c in chosen):
                chosen.append(idx)
            if len(chosen)>=n_peaks:
                break
        peaks=np.array(sorted(chosen[:n_peaks]),dtype=int)
    sel_times=times[mask][peaks]
    return [float(x) for x in sel_times[:n_peaks]]


def plot_postica_joint(
    postICA_final,
    json_data,
    experiment_dir,
    sub,
    subPath,
    saveNote="postICA_final",
    joint_times=None,
    figsize=(16,10),
    dpi=300
):
    from pathlib import Path
    import matplotlib.pyplot as plt

    output_dir=Path(experiment_dir)/subPath
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plot_tmin,plot_tmax=get_epoch_plot_limits(
        postICA_final,
        json_data
    )

    evoked_joint=(
        postICA_final
        .average()
        .copy()
        .crop(
            tmin=plot_tmin,
            tmax=plot_tmax,
            include_tmax=True
        )
    )

    if joint_times is None:
        joint_times=json_data.get(
            "postICA_joint_plot_times_s",
            [
                0.025,
                0.076,
                0.127,
                0.216
            ]
        )

    joint_times=[
        float(time)
        for time in joint_times
        if plot_tmin<=float(time)<=plot_tmax
    ]

    if len(joint_times)==0:
        response_times=[
            time
            for time in evoked_joint.times
            if max(0.0,plot_tmin)<=time<=plot_tmax
        ]

        if len(response_times)==0:
            response_times=list(
                evoked_joint.times
            )

        n_maps=min(
            4,
            len(response_times)
        )

        indices=[
            int(index)
            for index in __import__("numpy").linspace(
                0,
                len(response_times)-1,
                n_maps
            )
        ]

        joint_times=[
            float(response_times[index])
            for index in indices
        ]

    fig=evoked_joint.plot_joint(
        times=joint_times,
        picks="eeg",
        show=False,
        ts_args={
            "spatial_colors":True,
            "time_unit":"s"
        },
        topomap_args={
            "time_unit":"s",
            "contours":6,
            "sensors":True
        }
    )

    fig.set_size_inches(
        figsize[0],
        figsize[1]
    )

    for axis in fig.axes:
        xlabel=str(
            axis.get_xlabel()
        ).lower()

        if (
            "time" in xlabel
            or len(axis.lines)>1
        ):
            try:
                axis.set_xlim(
                    plot_tmin,
                    plot_tmax
                )
            except Exception:
                pass

    output_path=(
        output_dir
        /f"{sub}_{saveNote}_joint_plot.png"
    )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    json_data[
        "postICA_joint_plot"
    ]=str(output_path)

    json_data[
        "postICA_joint_plot_tmin_s"
    ]=float(plot_tmin)

    json_data[
        "postICA_joint_plot_tmax_s"
    ]=float(plot_tmax)

    json_data[
        "postICA_joint_plot_times_s"
    ]=[
        float(time)
        for time in joint_times
    ]

    print(
        "✅ Post-ICA joint plot: "
        f"{output_path}"
    )

    print(
        "   Plot window: "
        f"{plot_tmin*1000:.1f}–"
        f"{plot_tmax*1000:.1f} ms"
    )

    print(
        "   Topomap times:",
        joint_times
    )

    return str(output_path),json_data




def save_final_joint_plot(
    epochs,
    output_dir,
    subject,
    json_data,
    times=None,
    n_peaks=4,
    peak_tmin=0.020,
    peak_tmax=0.300,
    min_distance_ms=30,
    dpi=300
):
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import mne

    output_dir=Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if isinstance(epochs,mne.Evoked):
        evoked=epochs.copy()

    elif isinstance(epochs,mne.BaseEpochs):
        evoked=epochs.copy().average()

    else:
        raise TypeError(
            "epochs deve essere mne.Epochs oppure mne.Evoked."
        )

    evoked.pick("eeg")

    data_tmin=float(
        evoked.times.min()
    )

    data_tmax=float(
        evoked.times.max()
    )

    plot_tmin=max(
        float(
            json_data.get(
                "epochs_plot_timewindow_min",
                data_tmin
            )
        ),
        data_tmin
    )

    plot_tmax=min(
        float(
            json_data.get(
                "epochs_plot_timewindow_max",
                data_tmax
            )
        ),
        data_tmax
    )

    if plot_tmin>=plot_tmax:
        raise ValueError(
            f"Finestra grafica non valida: "
            f"[{plot_tmin},{plot_tmax}] s."
        )

    evoked_plot=evoked.copy().crop(
        tmin=plot_tmin,
        tmax=plot_tmax,
        include_tmax=True
    )

    valid_peak_tmin=max(
        float(peak_tmin),
        plot_tmin
    )

    valid_peak_tmax=min(
        float(peak_tmax),
        plot_tmax
    )

    if valid_peak_tmin>=valid_peak_tmax:
        valid_peak_tmin=max(
            0.0,
            plot_tmin
        )

        valid_peak_tmax=plot_tmax

    if times is None:
        mask=(
            (evoked_plot.times>=valid_peak_tmin)
            &(evoked_plot.times<=valid_peak_tmax)
        )

        candidate_times=evoked_plot.times[
            mask
        ]

        if candidate_times.size==0:
            raise ValueError(
                "Nessun campione nella finestra di selezione topomap."
            )

        gfp=np.std(
            evoked_plot.data[:,mask],
            axis=0
        )

        min_distance_samples=max(
            1,
            int(
                round(
                    min_distance_ms
                    *float(evoked_plot.info["sfreq"])
                    /1000.0
                )
            )
        )

        order=np.argsort(
            gfp
        )[::-1]

        selected_indices=[]

        for index in order:
            if all(
                abs(
                    int(index)-int(previous)
                )>=min_distance_samples
                for previous in selected_indices
            ):
                selected_indices.append(
                    int(index)
                )

            if len(selected_indices)>=n_peaks:
                break

        selected_indices=sorted(
            selected_indices
        )

        times=[
            float(
                candidate_times[index]
            )
            for index in selected_indices
        ]

    else:
        times=[
            float(time)
            for time in times
            if plot_tmin<=float(time)<=plot_tmax
        ]

    if not times:
        raise ValueError(
            "Nessun tempo valido disponibile per plot_joint."
        )

    fig=evoked_plot.plot_joint(
        times=times,
        title=(
            f"{subject} — EEG "
            f"({len(evoked_plot.ch_names)} channels)"
        ),
        show=False,
        ts_args={
            "spatial_colors":True,
            "gfp":False,
            "time_unit":"s"
        },
        topomap_args={
            "sensors":True,
            "contours":6,
            "time_unit":"s"
        }
    )

    if isinstance(fig,(list,tuple)):
        fig=fig[0]

    fig.set_size_inches(
        16,
        10
    )

    for axis in fig.axes:
        xlabel=str(
            axis.get_xlabel()
        ).lower()

        if "time" in xlabel:
            axis.set_xlim(
                plot_tmin,
                plot_tmax
            )

    output_path=(
        output_dir
        /f"{subject}_postICA_final_joint_plot.png"
    )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    return (
        output_path,
        times,
        plot_tmin,
        plot_tmax
    )


def update_pkl_hashes(
    postICA_final,
    json_data,
    experiment_dir
):
    from pathlib import Path
    from datetime import datetime
    import hashlib
    import numpy as np
    import mne

    root=Path(
        experiment_dir
    ).expanduser().resolve()

    def sha256_file(
        file_path,
        chunk_size=1024*1024
    ):
        digest=hashlib.sha256()

        with open(
            file_path,
            "rb"
        ) as file:
            while True:
                chunk=file.read(
                    chunk_size
                )

                if not chunk:
                    break

                digest.update(
                    chunk
                )

        return digest.hexdigest()

    def sha256_array(
        array,
        dtype=np.float64
    ):
        array=np.ascontiguousarray(
            array,
            dtype=dtype
        )

        return hashlib.sha256(
            array.tobytes()
        ).hexdigest()

    pkl_files=sorted(
        path
        for path in root.rglob(
            "*.pkl"
        )
        if path.is_file()
    )

    pkl_hashes={}

    for pkl_path in pkl_files:
        relative_path=str(
            pkl_path.relative_to(
                root
            )
        )

        try:
            stat=pkl_path.stat()

            pkl_hashes[
                relative_path
            ]={
                "sha256":sha256_file(
                    pkl_path
                ),
                "size_bytes":int(
                    stat.st_size
                ),
                "modified_time":datetime.fromtimestamp(
                    stat.st_mtime
                ).isoformat(),
                "absolute_path":str(
                    pkl_path
                )
            }

        except Exception as error:
            pkl_hashes[
                relative_path
            ]={
                "sha256":None,
                "size_bytes":None,
                "modified_time":None,
                "absolute_path":str(
                    pkl_path
                ),
                "error":str(
                    error
                )
            }

    collection_digest=hashlib.sha256()

    for relative_path in sorted(
        pkl_hashes
    ):
        file_hash=pkl_hashes[
            relative_path
        ].get(
            "sha256"
        )

        if file_hash is None:
            continue

        collection_digest.update(
            relative_path.encode(
                "utf-8"
            )
        )

        collection_digest.update(
            file_hash.encode(
                "ascii"
            )
        )

    json_data[
        "pkl_hash_algorithm"
    ]="SHA-256"

    json_data[
        "pkl_hashes"
    ]=pkl_hashes

    json_data[
        "pkl_hashes_count"
    ]=int(
        len(pkl_hashes)
    )

    json_data[
        "pkl_collection_sha256"
    ]=collection_digest.hexdigest()

    json_data[
        "pkl_hashes_updated_at"
    ]=datetime.now().isoformat()

    json_data.pop(
        "pkl_hash_manifest",
        None
    )

    if isinstance(
        postICA_final,
        mne.BaseEpochs
    ):
        final_data=postICA_final.get_data()

        selection=getattr(
            postICA_final,
            "selection",
            np.arange(
                len(postICA_final)
            )
        )

        json_data[
            "postICA_final_object_type"
        ]=type(
            postICA_final
        ).__name__

        json_data[
            "postICA_final_data_sha256"
        ]=sha256_array(
            final_data
        )

        json_data[
            "postICA_final_selection_sha256"
        ]=sha256_array(
            selection,
            dtype=np.int64
        )

        json_data[
            "postICA_final_data_shape"
        ]=[
            int(value)
            for value in final_data.shape
        ]

        json_data[
            "postICA_final_n_epochs"
        ]=int(
            len(postICA_final)
        )

    elif isinstance(
        postICA_final,
        mne.Evoked
    ):
        final_data=postICA_final.data

        json_data[
            "postICA_final_object_type"
        ]=type(
            postICA_final
        ).__name__

        json_data[
            "postICA_final_data_sha256"
        ]=sha256_array(
            final_data
        )

        json_data[
            "postICA_final_selection_sha256"
        ]=None

        json_data[
            "postICA_final_data_shape"
        ]=[
            int(value)
            for value in final_data.shape
        ]

    else:
        raise TypeError(
            "postICA_final deve essere "
            "mne.Epochs oppure mne.Evoked."
        )

    json_data[
        "postICA_final_sfreq"
    ]=float(
        postICA_final.info["sfreq"]
    )

    json_data[
        "postICA_final_tmin"
    ]=float(
        postICA_final.times[0]
    )

    json_data[
        "postICA_final_tmax"
    ]=float(
        postICA_final.times[-1]
    )

    json_data[
        "postICA_final_channels"
    ]=list(
        postICA_final.ch_names
    )

    json_data[
        "postICA_final_bads"
    ]=list(
        postICA_final.info.get(
            "bads",
            []
        )
    )

    print(
        f"✅ PKL hashes aggiornati: "
        f"{len(pkl_hashes)} file"
    )

    for relative_path,information in pkl_hashes.items():
        file_hash=information.get(
            "sha256"
        )

        if file_hash is None:
            print(
                f"   {relative_path}: ERROR"
            )
        else:
            print(
                f"   {relative_path}: "
                f"{file_hash[:16]}..."
            )

    print(
        "✅ PKL collection SHA-256:",
        json_data[
            "pkl_collection_sha256"
        ]
    )

    print(
        "✅ postICA_final data SHA-256:",
        json_data[
            "postICA_final_data_sha256"
        ]
    )

    return json_data


def saveLoadTestFinal(
    postICA_final,
    json_data,
    experiment_dir,
    sub,
    start_time
):
    from pathlib import Path
    from datetime import datetime
    import json

    if postICA_final is None:
        raise ValueError(
            "saveLoadTestFinal ha ricevuto postICA_final=None."
        )

    experiment_dir=Path(
        experiment_dir
    ).expanduser().resolve()

    json_data.pop(
        "postICA_final_joint_plot",
        None
    )

    json_data.pop(
        "postICA_final_joint_plot_tmin_s",
        None
    )

    json_data.pop(
        "postICA_final_joint_plot_tmax_s",
        None
    )

    json_data.pop(
        "postICA_final_joint_times_s",
        None
    )

    json_data.pop(
        "postICA_final_joint_times_ms",
        None
    )

    json_data.pop(
        "postICA_final_joint_plot_error",
        None
    )

    old_joint_paths=[
        experiment_dir
        /f"{sub}_postICA_final_joint_plot.png",
        experiment_dir
        /f"{sub}_final_joint_plot.png"
    ]

    for old_joint_path in old_joint_paths:
        try:
            if old_joint_path.exists():
                old_joint_path.unlink()

                print(
                    "🗑️ Vecchio joint plot eliminato:",
                    old_joint_path
                )

        except Exception as error:
            print(
                "⚠️ Impossibile eliminare il vecchio joint plot:",
                error
            )

    try:
        json_data=plot_final_tep_summary(
            postICA_final=postICA_final,
            json_data=json_data,
            experiment_dir=experiment_dir,
            sub=sub,
            dpi=300
        )

        json_data.pop(
            "TEP_final_summary_error",
            None
        )

        print(
            "✅ Final TEP summary:",
            json_data.get(
                "TEP_final_summary_plot"
            )
        )

    except Exception as error:
        json_data[
            "TEP_final_summary_plot"
        ]=None

        json_data[
            "TEP_final_summary_error"
        ]=str(
            error
        )

        print(
            "⚠️ Impossibile generare il final TEP summary:",
            error
        )

    try:
        json_data=update_pkl_hashes(
            postICA_final=postICA_final,
            json_data=json_data,
            experiment_dir=experiment_dir
        )

        json_data.pop(
            "pkl_hashes_error",
            None
        )

    except Exception as error:
        json_data[
            "pkl_hashes_error"
        ]=str(
            error
        )

        print(
            "⚠️ Impossibile calcolare gli hash dei PKL:",
            error
        )

    try:
        json_data[
            "saveLoadTestFinal_runtime_seconds"
        ]=float(
            datetime.now().timestamp()
            -float(start_time)
        )

    except Exception:
        json_data[
            "saveLoadTestFinal_runtime_seconds"
        ]=None

    json_data[
        "final_data_object_type"
    ]=type(
        postICA_final
    ).__name__

    json_data[
        "final_data_sfreq"
    ]=float(
        postICA_final.info[
            "sfreq"
        ]
    )

    json_data[
        "final_data_channels"
    ]=list(
        postICA_final.ch_names
    )

    json_data[
        "final_data_bads"
    ]=list(
        postICA_final.info.get(
            "bads",
            []
        )
    )

    json_data[
        "final_data_tmin"
    ]=float(
        postICA_final.times[
            0
        ]
    )

    json_data[
        "final_data_tmax"
    ]=float(
        postICA_final.times[
            -1
        ]
    )

    json_data[
        "final_data_n_epochs"
    ]=int(
        len(
            postICA_final
        )
    )

    json_data[
        "final_data_n_channels"
    ]=int(
        len(
            postICA_final.ch_names
        )
    )

    json_data[
        "final_stimulation_side"
    ]=str(
        json_data.get(
            "emispheric_stimulation",
            "UNK"
        )
    ).upper()

    json_data[
        "TEP_final_summary_joint_plot_disabled"
    ]=True

    pars_path=(
        experiment_dir
        /f"{sub}_pars.json"
    )

    try:
        with open(
            pars_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                json_data,
                file,
                indent=4,
                sort_keys=True,
                default=str
            )

        print(
            "✅ JSON globale aggiornato:",
            pars_path
        )

    except Exception as error:
        print(
            "⚠️ Impossibile aggiornare il JSON globale:",
            error
        )

    return json_data


def plot_final_tep_summary(
    postICA_final,
    json_data,
    experiment_dir,
    sub,
    dpi=300
):
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import mne

    from pathlib import Path
    from scipy.signal import find_peaks
    from matplotlib.patches import ConnectionPatch

    if postICA_final is None:
        raise ValueError(
            "plot_final_tep_summary ha ricevuto postICA_final=None."
        )

    experiment_dir=Path(
        experiment_dir
    ).expanduser().resolve()

    def resolve_path(value):
        if value is None:
            return None

        try:
            path=Path(
                str(value)
            ).expanduser()

            if not path.is_absolute():
                path=experiment_dir/path

            return path.resolve()

        except Exception:
            return None

    def optional_float(value):
        if value is None:
            return None

        try:
            value=float(value)
        except Exception:
            return None

        if not np.isfinite(value):
            return None

        return value

    def optional_int(value):
        if value is None:
            return None

        try:
            return int(
                float(value)
            )
        except Exception:
            return None

    def load_tfr_csv(csv_path):
        dataframe=pd.read_csv(
            csv_path,
            index_col=0
        )

        frequencies=pd.to_numeric(
            dataframe.index,
            errors="coerce"
        ).to_numpy(
            dtype=float
        )

        times_raw=np.asarray(
            [
                float(column)
                for column in dataframe.columns
            ],
            dtype=float
        )

        tfr=dataframe.apply(
            pd.to_numeric,
            errors="coerce"
        ).to_numpy(
            dtype=float
        )

        valid_frequencies=np.isfinite(
            frequencies
        )

        frequencies=frequencies[
            valid_frequencies
        ]

        tfr=tfr[
            valid_frequencies,
            :
        ]

        if (
            times_raw.size
            and np.nanmax(
                np.abs(
                    times_raw
                )
            )<=10
        ):
            times_tfr_ms=times_raw*1000.0
        else:
            times_tfr_ms=times_raw

        time_order=np.argsort(
            times_tfr_ms
        )

        frequency_order=np.argsort(
            frequencies
        )

        times_tfr_ms=times_tfr_ms[
            time_order
        ]

        frequencies=frequencies[
            frequency_order
        ]

        tfr=tfr[
            frequency_order,
            :
        ][
            :,
            time_order
        ]

        return (
            times_tfr_ms,
            frequencies,
            tfr
        )

    def select_topomap_indices(
        times_ms,
        gmfp,
        sfreq,
        x_min,
        x_max,
        n_maps,
        tmin_ms,
        tmax_ms,
        minimum_distance_ms
    ):
        search_mask=(
            (times_ms>=max(x_min,tmin_ms))
            &(times_ms<=min(x_max,tmax_ms))
        )

        if np.sum(search_mask)<3:
            search_mask=(
                (times_ms>=x_min)
                &(times_ms<=x_max)
            )

        candidate_indices=np.where(
            search_mask
        )[0]

        if candidate_indices.size==0:
            return []

        candidate_gmfp=gmfp[
            candidate_indices
        ]

        minimum_distance_samples=max(
            1,
            int(
                round(
                    minimum_distance_ms
                    *sfreq
                    /1000.0
                )
            )
        )

        peaks,_=find_peaks(
            candidate_gmfp,
            distance=minimum_distance_samples
        )

        selected=[]

        if peaks.size:
            ranked_peaks=peaks[
                np.argsort(
                    candidate_gmfp[
                        peaks
                    ]
                )[::-1]
            ]

            for local_index in ranked_peaks:
                global_index=int(
                    candidate_indices[
                        local_index
                    ]
                )

                if all(
                    abs(
                        global_index-existing_index
                    )>=minimum_distance_samples
                    for existing_index in selected
                ):
                    selected.append(
                        global_index
                    )

                if len(selected)>=n_maps:
                    break

        if len(selected)<n_maps:
            ranked_samples=np.argsort(
                candidate_gmfp
            )[::-1]

            for local_index in ranked_samples:
                global_index=int(
                    candidate_indices[
                        local_index
                    ]
                )

                if all(
                    abs(
                        global_index-existing_index
                    )>=minimum_distance_samples
                    for existing_index in selected
                ):
                    selected.append(
                        global_index
                    )

                if len(selected)>=n_maps:
                    break

        if len(selected)<n_maps:
            fallback_positions=np.linspace(
                0,
                len(candidate_indices)-1,
                min(
                    n_maps,
                    len(candidate_indices)
                )
            ).astype(int)

            for local_index in fallback_positions:
                global_index=int(
                    candidate_indices[
                        local_index
                    ]
                )

                if global_index not in selected:
                    selected.append(
                        global_index
                    )

                if len(selected)>=n_maps:
                    break

        return sorted(
            selected[:n_maps]
        )

    side=str(
        json_data.get(
            "emispheric_stimulation",
            json_data.get(
                "hemispheric_stimulation",
                "UNK"
            )
        )
    ).strip().upper()

    evoked=postICA_final.average()

    data_uv=np.asarray(
        evoked.get_data(),
        dtype=float
    )*1e6

    times_ms=np.asarray(
        evoked.times,
        dtype=float
    )*1000.0

    sfreq=float(
        evoked.info["sfreq"]
    )

    n_epochs=int(
        len(
            postICA_final
        )
    )

    n_channels=int(
        len(
            evoked.ch_names
        )
    )

    data_min_ms=float(
        np.nanmin(
            times_ms
        )
    )

    data_max_ms=float(
        np.nanmax(
            times_ms
        )
    )

    requested_min_ms=1000.0*float(
        json_data.get(
            "epochs_plot_timewindow_min",
            evoked.times.min()
        )
    )

    requested_max_ms=1000.0*float(
        json_data.get(
            "epochs_plot_timewindow_max",
            evoked.times.max()
        )
    )

    x_min=max(
        requested_min_ms,
        data_min_ms
    )

    x_max=min(
        requested_max_ms,
        data_max_ms
    )

    if x_min>=x_max:
        raise ValueError(
            "Finestra grafica non valida: "
            f"{x_min:.1f}–{x_max:.1f} ms."
        )

    plot_mask=(
        (times_ms>=x_min)
        &(times_ms<=x_max)
    )

    if not np.any(plot_mask):
        raise ValueError(
            "La finestra grafica non contiene campioni."
        )

    channel_colors=plt.cm.hsv(
        np.linspace(
            0,
            1,
            n_channels,
            endpoint=False
        )
    )

    gmfp=np.sqrt(
        np.nanmean(
            data_uv**2,
            axis=0
        )
    )

    response_window=json_data.get(
        "pcist_response_window_ms",
        (10.0,300.0)
    )

    if (
        isinstance(
            response_window,
            (list,tuple)
        )
        and len(response_window)==2
    ):
        response_start_ms=float(
            response_window[0]
        )

        response_end_ms=float(
            response_window[1]
        )
    else:
        response_start_ms=10.0
        response_end_ms=300.0

    gmfp_peak_mask=(
        (times_ms>=response_start_ms)
        &(times_ms<=response_end_ms)
        &plot_mask
    )

    if not np.any(gmfp_peak_mask):
        gmfp_peak_mask=(
            (times_ms>=0)
            &plot_mask
        )

    if not np.any(gmfp_peak_mask):
        gmfp_peak_mask=plot_mask.copy()

    gmfp_peak_indices=np.where(
        gmfp_peak_mask
    )[0]

    gmfp_peak_index=int(
        gmfp_peak_indices[
            np.nanargmax(
                gmfp[
                    gmfp_peak_mask
                ]
            )
        ]
    )

    gmfp_peak_uv=float(
        gmfp[
            gmfp_peak_index
        ]
    )

    gmfp_peak_latency_ms=float(
        times_ms[
            gmfp_peak_index
        ]
    )

    n_topomaps=int(
        json_data.get(
            "TEP_final_summary_n_topomaps",
            4
        )
    )

    n_topomaps=max(
        1,
        min(
            n_topomaps,
            6
        )
    )

    topomap_tmin_ms=float(
        json_data.get(
            "TEP_final_summary_topomap_tmin_ms",
            max(
                20.0,
                response_start_ms
            )
        )
    )

    topomap_tmax_ms=float(
        json_data.get(
            "TEP_final_summary_topomap_tmax_ms",
            min(
                300.0,
                response_end_ms
            )
        )
    )

    topomap_min_distance_ms=float(
        json_data.get(
            "TEP_final_summary_topomap_min_distance_ms",
            30.0
        )
    )

    configured_times=json_data.get(
        "TEP_final_summary_topomap_times_ms",
        None
    )

    if (
        isinstance(
            configured_times,
            (list,tuple)
        )
        and len(configured_times)>0
    ):
        topomap_indices=[]

        for time_value in configured_times:
            try:
                time_value=float(
                    time_value
                )
            except Exception:
                continue

            if not x_min<=time_value<=x_max:
                continue

            nearest_index=int(
                np.argmin(
                    np.abs(
                        times_ms-time_value
                    )
                )
            )

            if nearest_index not in topomap_indices:
                topomap_indices.append(
                    nearest_index
                )

        topomap_indices=sorted(
            topomap_indices
        )[:n_topomaps]

    else:
        topomap_indices=select_topomap_indices(
            times_ms=times_ms,
            gmfp=gmfp,
            sfreq=sfreq,
            x_min=x_min,
            x_max=x_max,
            n_maps=n_topomaps,
            tmin_ms=topomap_tmin_ms,
            tmax_ms=topomap_tmax_ms,
            minimum_distance_ms=topomap_min_distance_ms
        )

    pci_value=optional_float(
        json_data.get(
            "PCIst"
        )
    )

    pci_n_dims=optional_int(
        json_data.get(
            "PCIst_n_dims"
        )
    )

    energy_value=optional_float(
        json_data.get(
            "feat_step_energy"
        )
    )

    integral_value=optional_float(
        json_data.get(
            "feat_step_absolute_integral",
            json_data.get(
                "feat_step_integral"
            )
        )
    )

    nf_value=optional_float(
        json_data.get(
            "TEP_natural_frequency_hz"
        )
    )

    nf_score=optional_float(
        json_data.get(
            "TEP_natural_frequency_score",
            json_data.get(
                "TEP_natural_frequency_power_db"
            )
        )
    )

    nf_boundary=bool(
        json_data.get(
            "TEP_natural_frequency_at_boundary",
            False
        )
    )

    nf_seed_channels=json_data.get(
        "TEP_natural_frequency_seed_channels",
        json_data.get(
            "seedChans",
            []
        )
    )

    if isinstance(
        nf_seed_channels,
        (list,tuple)
    ):
        nf_seed_text=", ".join(
            str(channel)
            for channel in nf_seed_channels
        )
    else:
        nf_seed_text=str(
            nf_seed_channels
        )

    nf_tfr_csv=resolve_path(
        json_data.get(
            "TEP_natural_frequency_tfr_csv"
        )
    )

    nf_spectrum_csv=resolve_path(
        json_data.get(
            "TEP_natural_frequency_spectrum_csv"
        )
    )

    feature_labels=[]
    feature_values=[]
    feature_units=[]

    if energy_value is not None:
        feature_labels.append(
            "Energy"
        )

        feature_values.append(
            energy_value
        )

        feature_units.append(
            "µV²"
        )

    if integral_value is not None:
        feature_labels.append(
            "Absolute integral"
        )

        feature_values.append(
            integral_value
        )

        feature_units.append(
            "µV·ms"
        )

    if pci_value is not None:
        feature_labels.append(
            "PCIst"
        )

        feature_values.append(
            pci_value
        )

        feature_units.append(
            ""
        )

    if pci_n_dims is not None:
        feature_labels.append(
            "PCIst n_dims"
        )

        feature_values.append(
            float(
                pci_n_dims
            )
        )

        feature_units.append(
            ""
        )

    figure=plt.figure(
        figsize=(16,23),
        constrained_layout=True
    )

    grid=figure.add_gridspec(
        nrows=5,
        ncols=3,
        width_ratios=[
            10.0,
            2.0,
            0.40
        ],
        height_ratios=[
            2.10,
            3.10,
            1.60,
            3.60,
            2.50
        ],
        hspace=0.16,
        wspace=0.10
    )

    topomap_grid=grid[0,:].subgridspec(
        1,
        n_topomaps+1,
        width_ratios=[
            *([1.0]*n_topomaps),
            0.10
        ],
        wspace=0.08
    )

    topomap_axes=[
        figure.add_subplot(
            topomap_grid[
                0,
                index
            ]
        )
        for index in range(
            n_topomaps
        )
    ]

    ax_topomap_colorbar=figure.add_subplot(
        topomap_grid[
            0,
            n_topomaps
        ]
    )

    ax_butterfly=figure.add_subplot(
        grid[1,0]
    )

    ax_scalp=figure.add_subplot(
        grid[1,1]
    )

    ax_butterfly_pad=figure.add_subplot(
        grid[1,2]
    )

    ax_gmfp=figure.add_subplot(
        grid[2,0],
        sharex=ax_butterfly
    )

    ax_gmfp_side=figure.add_subplot(
        grid[2,1:]
    )

    ax_ersp=figure.add_subplot(
        grid[3,0],
        sharex=ax_butterfly
    )

    ax_ersp_profile=figure.add_subplot(
        grid[3,1],
        sharey=ax_ersp
    )

    ax_ersp_colorbar=figure.add_subplot(
        grid[3,2]
    )

    ax_features=figure.add_subplot(
        grid[4,:]
    )

    ax_butterfly_pad.axis(
        "off"
    )

    ax_gmfp_side.axis(
        "off"
    )

    for channel_index in range(
        n_channels
    ):
        ax_butterfly.plot(
            times_ms[
                plot_mask
            ],
            data_uv[
                channel_index,
                plot_mask
            ],
            color=channel_colors[
                channel_index
            ],
            linewidth=1.0,
            alpha=0.75
        )

    if x_min<=0<=x_max:
        ax_butterfly.axvline(
            0,
            color="black",
            linestyle="--",
            linewidth=1
        )

    ax_butterfly.axhline(
        0,
        color="black",
        linewidth=0.8
    )

    ax_butterfly.set_xlim(
        x_min,
        x_max
    )

    ax_butterfly.set_ylabel(
        "Amplitude [µV]"
    )

    ax_butterfly.set_title(
        (
            f"Final TEP butterfly — "
            f"{side} stimulation\n"
            f"epochs={n_epochs} | "
            f"channels={n_channels}"
        ),
        pad=8
    )

    ax_butterfly.tick_params(
        axis="x",
        labelbottom=False
    )

    ax_butterfly.grid(
        True,
        alpha=0.18
    )

    for axis in topomap_axes:
        axis.axis(
            "off"
        )

    topomap_image=None

    if topomap_indices:
        topomap_vmax=float(
            np.nanpercentile(
                np.abs(
                    data_uv[
                        :,
                        topomap_indices
                    ]
                ),
                98
            )
        )

        if (
            not np.isfinite(topomap_vmax)
            or topomap_vmax<=0
        ):
            topomap_vmax=1.0

        for axis,index in zip(
            topomap_axes,
            topomap_indices
        ):
            try:
                topomap_image,_=mne.viz.plot_topomap(
                    data_uv[
                        :,
                        index
                    ],
                    evoked.info,
                    axes=axis,
                    show=False,
                    cmap="RdBu_r",
                    vlim=(
                        -topomap_vmax,
                        topomap_vmax
                    ),
                    contours=6,
                    sensors=True,
                    outlines="head",
                    extrapolate="head",
                    image_interp="cubic"
                )

                axis.set_title(
                    f"{times_ms[index]/1000.0:.3f} s",
                    fontsize=14,
                    pad=7
                )

            except Exception as error:
                axis.text(
                    0.5,
                    0.5,
                    "Topomap\nnot available",
                    ha="center",
                    va="center",
                    transform=axis.transAxes
                )

                axis.axis(
                    "off"
                )

                print(
                    "⚠️ Final topomap:",
                    error
                )

        if topomap_image is not None:
            topomap_colorbar=figure.colorbar(
                topomap_image,
                cax=ax_topomap_colorbar
            )

            topomap_colorbar.set_label(
                "Amplitude [µV]"
            )

        else:
            ax_topomap_colorbar.axis(
                "off"
            )

        topomap_axes[0].text(
            -0.10,
            1.20,
            (
                f"TEP scalp maps"
            ),
            transform=topomap_axes[0].transAxes,
            ha="left",
            va="bottom",
            fontsize=13,
            fontweight="bold"
        )

    else:
        for axis in topomap_axes:
            axis.text(
                0.5,
                0.5,
                "Topomap not available",
                ha="center",
                va="center",
                transform=axis.transAxes
            )

            axis.axis(
                "off"
            )

        ax_topomap_colorbar.axis(
            "off"
        )

    try:
        positions=np.asarray(
            [
                channel["loc"][:2]
                for channel in evoked.info["chs"]
            ],
            dtype=float
        )

        valid_positions=(
            np.all(
                np.isfinite(
                    positions
                ),
                axis=1
            )
            &(
                np.linalg.norm(
                    positions,
                    axis=1
                )>0
            )
        )

        scalp_positions=positions[
            valid_positions
        ]

        scalp_names=[
            channel_name
            for channel_name,valid
            in zip(
                evoked.ch_names,
                valid_positions
            )
            if valid
        ]

        scalp_colors=channel_colors[
            valid_positions
        ]

        if len(scalp_positions)==0:
            raise ValueError(
                "Nessuna posizione EEG valida."
            )

        scalp_x=scalp_positions[:,0]
        scalp_y=scalp_positions[:,1]

        scalp_x-=np.mean(
            scalp_x
        )

        scalp_y-=np.mean(
            scalp_y
        )

        scalp_radius=float(
            np.max(
                np.sqrt(
                    scalp_x**2
                    +scalp_y**2
                )
            )
        )

        if scalp_radius>0:
            scalp_x/=scalp_radius
            scalp_y/=scalp_radius

        head=plt.Circle(
            (0,0),
            1.0,
            edgecolor="black",
            facecolor="none",
            linewidth=1.2
        )

        ax_scalp.add_patch(
            head
        )

        ax_scalp.plot(
            [-0.08,0.00,0.08],
            [1.00,1.12,1.00],
            color="black",
            linewidth=1
        )

        ax_scalp.plot(
            [-1.02,-1.08,-1.08,-1.02],
            [0.15,0.10,-0.10,-0.15],
            color="black",
            linewidth=1
        )

        ax_scalp.plot(
            [1.02,1.08,1.08,1.02],
            [0.15,0.10,-0.10,-0.15],
            color="black",
            linewidth=1
        )

        for (
            x_value,
            y_value,
            channel_name,
            channel_color
        ) in zip(
            scalp_x,
            scalp_y,
            scalp_names,
            scalp_colors
        ):
            ax_scalp.scatter(
                x_value,
                y_value,
                s=45,
                color=channel_color,
                edgecolor="black",
                linewidth=0.3,
                zorder=3
            )

            ax_scalp.text(
                x_value,
                y_value+0.055,
                channel_name,
                ha="center",
                va="bottom",
                fontsize=6
            )

        ax_scalp.set_xlim(
            -1.25,
            1.25
        )

        ax_scalp.set_ylim(
            -1.20,
            1.25
        )

        ax_scalp.set_aspect(
            "equal"
        )

        ax_scalp.set_title(
            "Butterfly channel colors",
            pad=10
        )

        ax_scalp.axis(
            "off"
        )

    except Exception as error:
        ax_scalp.text(
            0.5,
            0.5,
            "Scalp channel colors not available",
            ha="center",
            va="center",
            transform=ax_scalp.transAxes
        )

        ax_scalp.set_title(
            "Butterfly channel colors"
        )

        ax_scalp.axis(
            "off"
        )

        print(
            "⚠️ Scalp channel-color plot:",
            error
        )

    ax_gmfp.plot(
        times_ms[
            plot_mask
        ],
        gmfp[
            plot_mask
        ],
        linewidth=2
    )

    if x_min<=0<=x_max:
        ax_gmfp.axvline(
            0,
            color="black",
            linestyle="--",
            linewidth=1
        )

    ax_gmfp.axvline(
        gmfp_peak_latency_ms,
        color="black",
        linestyle=":",
        linewidth=1.4
    )

    ax_gmfp.scatter(
        gmfp_peak_latency_ms,
        gmfp_peak_uv,
        color="black",
        s=35,
        zorder=4
    )

    ax_gmfp.set_xlim(
        x_min,
        x_max
    )

    ax_gmfp.set_ylabel(
        "GMFP [µV]"
    )

    ax_gmfp.set_title(
        (
            f"GMFP — {side} stimulation\n"
            f"peak={gmfp_peak_uv:.3f} µV "
            f"at {gmfp_peak_latency_ms:.1f} ms"
        ),
        pad=8
    )

    ax_gmfp.tick_params(
        axis="x",
        labelbottom=False
    )

    ax_gmfp.grid(
        True,
        alpha=0.18
    )

    if topomap_indices:
        figure.canvas.draw()

        butterfly_ymin,butterfly_ymax=(
            ax_butterfly.get_ylim()
        )

        connection_y=butterfly_ymax

        for axis,index in zip(
            topomap_axes,
            topomap_indices
        ):
            selected_time=float(
                times_ms[
                    index
                ]
            )

            connector=ConnectionPatch(
                xyA=(
                    0.5,
                    0.02
                ),
                coordsA=axis.transAxes,
                xyB=(
                    selected_time,
                    connection_y
                ),
                coordsB=ax_butterfly.transData,
                color="0.65",
                linewidth=1.0,
                clip_on=False,
                zorder=1
            )

            figure.add_artist(
                connector
            )

            ax_butterfly.axvline(
                selected_time,
                color="0.70",
                linewidth=0.8,
                alpha=0.75
            )

    ersp_available=False

    if (
        nf_tfr_csv is not None
        and nf_tfr_csv.exists()
    ):
        try:
            (
                ersp_times_ms,
                ersp_frequencies,
                ersp_db
            )=load_tfr_csv(
                nf_tfr_csv
            )

            valid_time_mask=(
                (ersp_times_ms>=x_min)
                &(ersp_times_ms<=x_max)
            )

            if not np.any(
                valid_time_mask
            ):
                raise ValueError(
                    "Il TFR non contiene campioni "
                    "nella finestra richiesta."
                )

            ersp_times_plot=ersp_times_ms[
                valid_time_mask
            ]

            ersp_db_plot=ersp_db[
                :,
                valid_time_mask
            ]

            ersp_vmax=float(
                np.nanpercentile(
                    np.abs(
                        ersp_db_plot
                    ),
                    98
                )
            )

            if (
                not np.isfinite(ersp_vmax)
                or ersp_vmax<=0
            ):
                ersp_vmax=1.0

            ersp_image=ax_ersp.pcolormesh(
                ersp_times_plot,
                ersp_frequencies,
                ersp_db_plot,
                shading="auto",
                cmap="jet",
                vmin=-ersp_vmax,
                vmax=ersp_vmax
            )

            if x_min<=0<=x_max:
                ax_ersp.axvline(
                    0,
                    color="black",
                    linestyle=":",
                    linewidth=1.5
                )

            if nf_value is not None:
                ax_ersp.axhline(
                    nf_value,
                    color="black",
                    linestyle=":",
                    linewidth=1.2
                )

                ax_ersp.text(
                    x_max
                    -0.01*(
                        x_max-x_min
                    ),
                    nf_value+0.4,
                    f"{nf_value:.1f} Hz",
                    ha="right",
                    va="bottom",
                    fontsize=9
                )

            ax_ersp.set_xlim(
                x_min,
                x_max
            )

            ax_ersp.set_xlabel(
                "Time [ms]"
            )

            ax_ersp.set_ylabel(
                "Frequency [Hz]"
            )

            title=(
                f"Morlet ERSP — "
                f"{side} stimulation"
            )

            title_details=[]

            if nf_value is not None:
                title_details.append(
                    f"NF={nf_value:.1f} Hz"
                )

            if nf_score is not None:
                title_details.append(
                    f"score={nf_score:.3f}"
                )

            if nf_seed_text:
                title_details.append(
                    f"seed={nf_seed_text}"
                )

            if nf_boundary:
                title_details.append(
                    "boundary estimate"
                )

            if title_details:
                title+=(
                    "\n"
                    +" | ".join(
                        title_details
                    )
                )

            ax_ersp.set_title(
                title,
                pad=8
            )

            colorbar=figure.colorbar(
                ersp_image,
                cax=ax_ersp_colorbar
            )

            colorbar.set_label(
                "ERSP [dB]"
            )

            profile_loaded=False

            if (
                nf_spectrum_csv is not None
                and nf_spectrum_csv.exists()
            ):
                spectrum_dataframe=pd.read_csv(
                    nf_spectrum_csv
                )

                required_columns={
                    "frequency_hz",
                    "cumulative_positive_ersp"
                }

                if required_columns.issubset(
                    spectrum_dataframe.columns
                ):
                    profile_frequencies=pd.to_numeric(
                        spectrum_dataframe[
                            "frequency_hz"
                        ],
                        errors="coerce"
                    ).to_numpy(
                        dtype=float
                    )

                    profile_values=pd.to_numeric(
                        spectrum_dataframe[
                            "cumulative_positive_ersp"
                        ],
                        errors="coerce"
                    ).to_numpy(
                        dtype=float
                    )

                    valid_profile=(
                        np.isfinite(
                            profile_frequencies
                        )
                        &np.isfinite(
                            profile_values
                        )
                    )

                    profile_frequencies=profile_frequencies[
                        valid_profile
                    ]

                    profile_values=profile_values[
                        valid_profile
                    ]

                    if profile_values.size:
                        ax_ersp_profile.plot(
                            profile_values,
                            profile_frequencies,
                            color="black",
                            linewidth=2
                        )

                        profile_loaded=True

            if not profile_loaded:
                positive_ersp=np.maximum(
                    ersp_db_plot,
                    0
                )

                if hasattr(
                    np,
                    "trapezoid"
                ):
                    profile_values=np.trapezoid(
                        positive_ersp,
                        x=ersp_times_plot,
                        axis=1
                    )
                else:
                    profile_values=np.trapz(
                        positive_ersp,
                        x=ersp_times_plot,
                        axis=1
                    )

                ax_ersp_profile.plot(
                    profile_values,
                    ersp_frequencies,
                    color="black",
                    linewidth=2
                )

            ax_ersp_profile.set_xlabel(
                "Cumulative\npositive ERSP"
            )

            ax_ersp_profile.tick_params(
                axis="y",
                labelleft=False
            )

            ax_ersp_profile.grid(
                True,
                alpha=0.15
            )

            visible_fmin=float(
                np.nanmin(
                    ersp_frequencies
                )
            )

            visible_fmax=float(
                np.nanmax(
                    ersp_frequencies
                )
            )

            frequency_bands=[
                ("α",8.0,12.0),
                ("β1",13.0,20.0),
                ("β2",21.0,29.0),
                ("γ",30.0,visible_fmax)
            ]

            for (
                band_label,
                band_min,
                band_max
            ) in frequency_bands:
                visible_start=max(
                    band_min,
                    visible_fmin
                )

                visible_stop=min(
                    band_max,
                    visible_fmax
                )

                if visible_start>visible_stop:
                    continue

                band_center=(
                    visible_start
                    +visible_stop
                )/2.0

                if visible_fmin<=band_min<=visible_fmax:
                    ax_ersp_profile.axhline(
                        band_min,
                        color="0.7",
                        linewidth=0.7
                    )

                ax_ersp_profile.text(
                    0.96,
                    band_center,
                    band_label,
                    transform=(
                        ax_ersp_profile
                        .get_yaxis_transform()
                    ),
                    ha="right",
                    va="center",
                    fontsize=9
                )

            ersp_available=True

        except Exception as error:
            print(
                "⚠️ Morlet ERSP summary:",
                error
            )

    if not ersp_available:
        ax_ersp.text(
            0.5,
            0.5,
            "Morlet ERSP not available",
            ha="center",
            va="center",
            transform=ax_ersp.transAxes
        )

        ax_ersp.set_xlim(
            x_min,
            x_max
        )

        ax_ersp.set_xlabel(
            "Time [ms]"
        )

        ax_ersp.set_ylabel(
            "Frequency [Hz]"
        )

        ax_ersp_profile.axis(
            "off"
        )

        ax_ersp_colorbar.axis(
            "off"
        )

    if feature_values:
        values=np.asarray(
            feature_values,
            dtype=float
        )

        positions=np.arange(
            len(values)
        )

        bars=ax_features.barh(
            positions,
            values
        )

        ax_features.set_yticks(
            positions
        )

        ax_features.set_yticklabels(
            feature_labels
        )

        ax_features.invert_yaxis()

        finite_nonzero=np.abs(
            values[
                np.isfinite(values)
                &(values!=0)
            ]
        )

        use_symlog=False

        if finite_nonzero.size>=2:
            minimum_value=max(
                float(
                    np.nanmin(
                        finite_nonzero
                    )
                ),
                np.finfo(float).tiny
            )

            maximum_value=float(
                np.nanmax(
                    finite_nonzero
                )
            )

            use_symlog=bool(
                maximum_value/minimum_value>1e4
            )

        if use_symlog:
            linear_threshold=max(
                float(
                    np.nanmin(
                        finite_nonzero
                    )
                )*10.0,
                1e-15
            )

            ax_features.set_xscale(
                "symlog",
                linthresh=linear_threshold
            )

            scale_label="symlog scale"
        else:
            scale_label="linear scale"

        for (
            bar,
            value,
            unit
        ) in zip(
            bars,
            values,
            feature_units
        ):
            text=(
                f"{value:.4g} {unit}"
                if unit
                else f"{value:.4g}"
            )

            ax_features.annotate(
                text,
                xy=(
                    bar.get_width(),
                    bar.get_y()
                    +bar.get_height()/2
                ),
                xytext=(
                    5 if value>=0 else -5,
                    0
                ),
                textcoords="offset points",
                ha=(
                    "left"
                    if value>=0
                    else "right"
                ),
                va="center",
                fontsize=9
            )

        ax_features.set_xlabel(
            f"Feature value ({scale_label})"
        )

        ax_features.set_title(
            (
                f"Scalar TEP features — "
                f"{side} stimulation\n"
                "Energy, absolute integral and PCIst summary"
            ),
            pad=8
        )

        ax_features.grid(
            True,
            axis="x",
            alpha=0.18
        )

    else:
        ax_features.text(
            0.5,
            0.5,
            "Scalar TEP features not available",
            ha="center",
            va="center",
            transform=ax_features.transAxes
        )

        ax_features.axis(
            "off"
        )

    ax_butterfly.set_xlim(
        x_min,
        x_max
    )

    ax_gmfp.set_xlim(
        x_min,
        x_max
    )

    ax_ersp.set_xlim(
        x_min,
        x_max
    )

    figure.suptitle(
        (
            f"{sub} — Final TEP summary — "
            f"STIMULATION SIDE: {side}"
        ),
        fontsize=21
    )

    output_path=(
        experiment_dir
        /f"{sub}_{side}_TEP_final_summary.png"
    )

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(
        figure
    )

    selected_topomap_times_ms=[
        float(
            times_ms[
                index
            ]
        )
        for index in topomap_indices
    ]

    json_data[
        "TEP_final_summary_plot"
    ]=str(
        output_path
    )

    json_data[
        "TEP_final_summary_side"
    ]=side

    json_data[
        "TEP_final_summary_requested_time_axis_ms"
    ]=[
        float(requested_min_ms),
        float(requested_max_ms)
    ]

    json_data[
        "TEP_final_summary_effective_time_axis_ms"
    ]=[
        float(x_min),
        float(x_max)
    ]

    json_data[
        "TEP_final_summary_topomap_times_ms"
    ]=selected_topomap_times_ms

    json_data[
        "TEP_final_summary_topomap_times_s"
    ]=[
        float(
            time_ms/1000.0
        )
        for time_ms in selected_topomap_times_ms
    ]

    json_data[
        "TEP_final_summary_topomap_selection"
    ]=(
        "Local GMFP maxima ranked by amplitude "
        "with minimum temporal separation"
    )

    json_data[
        "TEP_final_summary_topomap_min_distance_ms"
    ]=float(
        topomap_min_distance_ms
    )

    json_data[
        "TEP_final_GMFP_peak_uV"
    ]=gmfp_peak_uv

    json_data[
        "TEP_final_GMFP_peak_latency_ms"
    ]=gmfp_peak_latency_ms

    json_data[
        "TEP_final_summary_PCIst"
    ]=pci_value

    json_data[
        "TEP_final_summary_PCIst_n_dims"
    ]=pci_n_dims

    json_data[
        "TEP_final_summary_natural_frequency_hz"
    ]=nf_value

    json_data[
        "TEP_final_summary_natural_frequency_at_boundary"
    ]=nf_boundary

    json_data[
        "TEP_final_summary_scalar_feature_labels"
    ]=list(
        feature_labels
    )

    json_data[
        "TEP_final_summary_scalar_feature_values"
    ]=[
        float(value)
        for value in feature_values
    ]

    json_data[
        "TEP_final_summary_scalar_feature_units"
    ]=list(
        feature_units
    )

    json_data[
        "TEP_final_summary_temporal_panels_aligned"
    ]=True

    json_data[
        "TEP_final_summary_topomap_connectors"
    ]=True

    json_data[
        "TEP_final_summary_channel_color_scalp"
    ]=True

    pars_path=(
        experiment_dir
        /f"{sub}_pars.json"
    )

    with open(
        pars_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            json_data,
            file,
            indent=4,
            sort_keys=True,
            default=str
        )

    print(
        "✅ Final TEP summary:",
        output_path
    )

    print(
        "   Plot window:",
        f"{x_min:.1f}–{x_max:.1f} ms"
    )

    print(
        "   Topomap times:",
        selected_topomap_times_ms
    )

    return json_data

def select_topomap_times_from_gmfp(
    times_ms,
    gmfp,
    sfreq,
    x_min,
    x_max,
    n_maps=4,
    peak_tmin_ms=20.0,
    peak_tmax_ms=300.0,
    min_distance_ms=30.0
):
    search_start=max(
        x_min,
        peak_tmin_ms
    )

    search_end=min(
        x_max,
        peak_tmax_ms
    )

    mask=(
        (times_ms>=search_start)
        &(times_ms<=search_end)
    )

    if np.sum(mask)<5:
        valid_times=times_ms[
            (times_ms>=x_min)
            &(times_ms<=x_max)
        ]

        if len(valid_times)==0:
            return []

        if len(valid_times)<=n_maps:
            return list(valid_times)

        idx=np.linspace(
            0,
            len(valid_times)-1,
            n_maps
        ).astype(int)

        return list(valid_times[idx])

    gmfp_window=gmfp[mask]
    times_window=times_ms[mask]

    min_distance_samples=max(
        1,
        int(
            round(
                min_distance_ms
                *sfreq
                /1000.0
            )
        )
    )

    peaks,properties=find_peaks(
        gmfp_window,
        distance=min_distance_samples
    )

    if len(peaks)==0:
        idx=np.linspace(
            0,
            len(times_window)-1,
            min(
                n_maps,
                len(times_window)
            )
        ).astype(int)

        return list(times_window[idx])

    peak_heights=gmfp_window[peaks]

    order=np.argsort(
        peak_heights
    )[::-1]

    selected_peaks=peaks[
        order[:n_maps]
    ]

    selected_times=np.sort(
        times_window[
            selected_peaks
        ]
    )

    if len(selected_times)<n_maps:
        fallback_idx=np.linspace(
            0,
            len(times_window)-1,
            n_maps
        ).astype(int)

        fallback_times=list(
            times_window[
                fallback_idx
            ]
        )

        combined=list(
            selected_times
        )

        for time in fallback_times:
            if len(combined)>=n_maps:
                break

            if not np.any(
                np.isclose(
                    combined,
                    time,
                    atol=min_distance_ms/2
                )
            ):
                combined.append(time)

        selected_times=np.sort(
            np.asarray(combined)
        )

    return list(
        selected_times[:n_maps]
    )