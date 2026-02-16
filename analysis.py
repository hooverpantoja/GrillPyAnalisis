# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:50:30 2023

@author: Esteban
"""

import numpy as np
import pandas as pd
import seaborn as sns
from maad import sound, util, rois, features, spl
import matplotlib.pyplot as plt
try:
    from skimage.transform import downscale_local_mean
except ImportError:
    from skimage.measure import block_reduce as downscale_local_mean
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances 
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import cm


class Analyser:
    def __init__(self, X, XX, sites_ind, meta_rows):
        self.X = X
        self.XX = XX
        self.sites_ind = sites_ind
        self.meta_rows = meta_rows

    @classmethod
    def longwave(cls, df, path, samp_len, flims, db, ch):
        long_wav = []
        L = str(len(df.file))
        for idx, fname in enumerate(df.file):
            s, fs = sound.load(path + '/' + fname)
            s = sound.trim(s, fs, 0, samp_len)
            if ch == 1:
                s = sound.resample(s, fs, flims[1] * 2 + 4000, res_type='kaiser_fast')
                fs = flims[1] * 2 + 4000
            long_wav.append(s)
            print('progress: ' + str(idx + 1) + ' of ' + L)
        long_wav = util.crossfade_list(long_wav, fs, fade_len=0.5)
        Sxx, tn, fn, _ = sound.spectrogram(
            long_wav, fs, window='hann', nperseg=1024, noverlap=512, flims=flims)
        Sxx_db = 20 * np.log10(Sxx / Sxx.min())
        Sxx_db[Sxx_db < db] = db
        fig, ax = plt.subplots()
        c = ax.pcolorfast(tn, fn / 1000, Sxx_db, cmap='gray', vmin=Sxx_db.min(), vmax=Sxx_db.max())
        plt.ylabel('Frequency [kHz]'); plt.xlabel('Time (s)')
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('dB in ref to min')
        plt.show(block=False)
        return (long_wav, fs, Sxx, tn, fn)

    @classmethod
    def shortwave(cls, s, fs, flims, db):
        Sxx, tn, fn, _ = sound.spectrogram(
            s, fs, window='hann', nperseg=1024, noverlap=512, flims=flims)
        Sxx_db = 20 * np.log10(Sxx / Sxx.min())
        Sxx_db[Sxx_db < db] = db
        fig, ax = plt.subplots()
        c = ax.pcolorfast(tn, fn / 1000, Sxx_db, cmap='gray', vmin=Sxx_db.min(), vmax=Sxx_db.max())
        plt.ylabel('Frequency [kHz]'); plt.xlabel('Time (s)')
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('dB in ref to min')
        plt.show(block=False)

    @classmethod
    def rois_spec(cls, data, flims, _ch_rec, db, bstd, bp):
        s_filt = data.wav
        Sxx, tn, fn, _ = sound.spectrogram(s_filt, data.wavfs, nperseg=1024, noverlap=512, flims=flims)
        Sxx_db = 20 * np.log(Sxx / Sxx.min())
        Sxx_db[Sxx_db < db] = 0
        Sxx_db = sound.smooth(Sxx_db, std=0.05)
        Sxxf = Sxx_db / Sxx_db.max()
        im_mask = rois.create_mask(im=Sxxf, mode_bin='absolute', bin_h=bstd, bin_l=bp)
        im_rois, df_rois = rois.select_rois(im_mask, min_roi=1, max_roi=5000)
        df_rois = util.format_features(df_rois, tn, fn)
        fig_kwargs = {
            'vmin': Sxxf.min(), 'vmax': Sxxf.max(),
            'extent': (tn[0], tn[-1], fn[0] / 1000, fn[-1] / 1000),
            'figsize': (4, 13), 'title': 'spectrogram',
            'xlabel': 'Time [s]', 'ylabel': 'Frequency [kHz]',
        }
        util.overlay_rois(Sxxf, df_rois, **fig_kwargs)
        plt.show(block=False)
        return df_rois, im_rois

    @classmethod
    def ind_batch(cls, df):
        print('in progress')
        df.reset_index()
        L = str(len(df.route))
        for i, file_name in enumerate(df.route):
            temp, f_ok = cls.compute_acoustic_indices(file_name)
            if f_ok == 'error':
                print('error calculating index')
            else:
                if isinstance(temp, pd.Series):
                    df.loc[df.index[i], 'ADI'] = temp.get('ADI')
                    df.loc[df.index[i], 'ACI'] = temp.get('ACI')
                    df.loc[df.index[i], 'NDSI'] = temp.get('NDSI')
                    df.loc[df.index[i], 'BI'] = temp.get('BI')
                    df.loc[df.index[i], 'Hf'] = temp.get('Hf')
                    df.loc[df.index[i], 'Ht'] = temp.get('Ht')
                    df.loc[df.index[i], 'H'] = temp.get('H')
                    df.loc[df.index[i], 'SC'] = temp.get('SC')
                    df.loc[df.index[i], 'NP'] = temp.get('NP')
                else:
                    print('warning: compute_acoustic_indices returned non-Series')
                print('progress: ' + str(i + 1) + ' of ' + L)
        print('indices calculated')
        return df

    @classmethod
    def ind_per_day(cls, df):
        print('in progress')
        group_site = df.groupby(['site'])
        Ls = len(group_site)
        k = 0
        ind_sites = []

        for site, data_site in group_site:
            print('site: ' + str(site) + ' # ' + str(k + 1) + ' of ' + str(Ls) + ' sites')
            group_day = data_site.groupby(['day'])
            Ld = len(group_day)
            i = 0
            s_long = []
            for _, rows in group_day:
                L = len(rows)
                for _, row in rows.iterrows():
                    file = row['route']
                    s, fs = sound.load(file)
                    s = sound.resample(s, fs, 48000, res_type='kaiser_fast')
                    fs = 48000
                    s_long.append(s)
                sound.spectrum(s_long, fs, window='hann', nperseg=1024, noverlap=512)
                i = i + 1
                print('progress: ' + str(i) + ' of ' + str(Ld))
            ind_sites.append(site)
            k = k + 1
            print('progress: ' + str(k) + ' of ' + str(Ls))

        df.reset_index()
        L = str(len(df.route))
        for i, file_name in enumerate(df.route):
            temp, f_ok = cls.compute_acoustic_indices(file_name)
            if f_ok == 'error':
                print('error calculating index')
            else:
                if isinstance(temp, pd.Series):
                    df.loc[df.index[i], 'ADI'] = temp.get('ADI')
                    df.loc[df.index[i], 'ACI'] = temp.get('ACI')
                    df.loc[df.index[i], 'NDSI'] = temp.get('NDSI')
                    df.loc[df.index[i], 'BI'] = temp.get('BI')
                    df.loc[df.index[i], 'Hf'] = temp.get('Hf')
                    df.loc[df.index[i], 'Ht'] = temp.get('Ht')
                    df.loc[df.index[i], 'H'] = temp.get('H')
                    df.loc[df.index[i], 'SC'] = temp.get('SC')
                    df.loc[df.index[i], 'NP'] = temp.get('NP')
                else:
                    print('warning: compute_acoustic_indices returned non-Series')
                print('progress: ' + str(i + 1) + ' of ' + L)
        print('indices calculated')
        return df

    @staticmethod
    def saturation(S, fn):
        S = S[fn > 2000]
        values, counts = np.unique(S, return_counts=True)
        mc = values[counts.argmax()]
        Sxx_db = 20 * np.log(S / mc)
        Sxx_b = Sxx_db > 100
        x, y = Sxx_b.shape
        sat = Sxx_b.sum() / (x * y)
        return sat

    @classmethod
    def compute_acoustic_indices(cls, path_audio):
        s, fs = sound.load(path_audio)
        if 's' in locals():
            target_fs = 48000
            s = sound.resample(s, fs, target_fs, res_type='kaiser_fast')
            Sxx, _tn, fn, _ext = sound.spectrogram(
                s, target_fs, nperseg=1024, noverlap=0, mode='amplitude')
            Sxx_power = Sxx ** 2
            # Sxx_dB = util.amplitude2dB(Sxx)
            ADI = features.acoustic_diversity_index(Sxx, fn, fmin=2000, fmax=24000, bin_step=1000, index='shannon', dB_threshold=-70)
            _, _, ACI = features.acoustic_complexity_index(Sxx)
            NDSI, _xBA, _xA, _xB = features.soundscape_index(Sxx_power, fn, flim_bioPh=(2000, 20000), flim_antroPh=(100, 2000))
            Ht = features.temporal_entropy(s)
            Hf, _ = features.frequency_entropy(Sxx_power)
            H = Hf * Ht
            BI = features.bioacoustics_index(Sxx, fn, flim=(2000, 24000))
            NP = features.number_of_peaks(Sxx_power, fn, mode='linear', min_peak_val=0, min_freq_dist=100, slopes=None, prominence=1e-6)
            SC = cls.saturation(Sxx, fn)
            df_indices = pd.Series({
                'ADI': ADI,
                'ACI': ACI,
                'NDSI': NDSI,
                'BI': BI,
                'Hf': Hf,
                'Ht': Ht,
                'H': H,
                'NP': int(NP),
                'SC': SC})
            flag = 'ok'
            return df_indices, flag
        else:
            df_indices = 1
            flag = 'error'
            return df_indices, flag

    @classmethod
    def plot_acoustic_indices(cls, df):
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
        sns.scatterplot(data=df, x='hour', y='ACI', ax=ax[0, 0], hue='site')
        sns.scatterplot(data=df, x='hour', y='ADI', ax=ax[0, 1], hue='site')
        sns.scatterplot(data=df, x='hour', y='BI', ax=ax[0, 2], hue='site')
        sns.scatterplot(data=df, x='hour', y='H', ax=ax[1, 0], hue='site')
        sns.scatterplot(data=df, x='hour', y='Ht', ax=ax[1, 1], hue='site')
        sns.scatterplot(data=df, x='hour', y='Hf', ax=ax[1, 2], hue='site')
        sns.scatterplot(data=df, x='hour', y='NDSI', ax=ax[2, 0], hue='site')
        sns.scatterplot(data=df, x='hour', y='NP', ax=ax[2, 1], hue='site')
        sns.scatterplot(data=df, x='hour', y='SC', ax=ax[2, 2], hue='site')
        fig.set_tight_layout('tight')
        plt.show(block=False)

    @classmethod
    def spl_batch(cls, df):
        S = -18
        G = 15
        VADC = 3.3
        df.reset_index()
        L = str(len(df.file))
        for i, file_name in enumerate(df.route):
            w, fs = sound.load(file_name)
            tmp = spl.wav2dBSPL(wave=w, gain=G, Vadc=VADC, sensitivity=S)
            df.loc[df.index[i], 'spl'] = tmp.mean()
            Sxx_tmp, _tn, fn, _ext = sound.spectrogram(
                w, fs, window='hann', nperseg=1024, noverlap=1024 // 2, verbose=False, display=False, savefig=None)
            mean_PSD = np.mean(Sxx_tmp, axis=1)
            spl_ant_tmp = [spl.psd2leq(mean_PSD[util.index_bw(fn, (0, 3000))], gain=G, sensitivity=S, Vadc=VADC)]
            df.loc[df.index[i], 'spl_ant'] = spl_ant_tmp
            spl_bio_tmp = [spl.psd2leq(mean_PSD[util.index_bw(fn, (3000, 24000))], gain=G, sensitivity=S, Vadc=VADC)]
            df.loc[df.index[i], 'spl_bio'] = spl_bio_tmp
            print('progress: ' + str(i + 1) + ' of ' + L)

        df_summary = {}
        for i, df_site in df.groupby('site'):
            night = df_site.query("hour>= 0 & hour <= 4.5 | hour>= 19.5 & hour <= 24")
            day = df_site.query("hour>= 6.5 & hour <= 17.5")
            site_summary = {
                'site': df_site.site[df_site.index[0]],
                'date_ini': str(df_site.date_fmt.min()),
                'date_end': str(df_site.date_fmt.max()),
                'spl_mean_night': str(night.spl.mean()),
                'spl_mean_day': str(day.spl.mean()),
                'spl_std_night': str(night.spl.std()),
                'spl_std_day': str(day.spl.std()),
                'spl_ant_mean_night': str(night.spl_ant.mean()),
                'spl_ant_mean_day': str(day.spl_ant.mean()),
                'spl_ant_std_night': str(night.spl_ant.std()),
                'spl_ant_std_day': str(day.spl_ant.std()),
                'spl_bio_mean_night': str(night.spl_bio.mean()),
                'spl_bio_mean_day': str(day.spl_bio.mean()),
                'spl_bio_std_night': str(night.spl_bio.std()),
                'spl_bio_std_day': str(day.spl_bio.std()),
            }
            df_summary[i] = site_summary
        df_summary = pd.DataFrame(df_summary).T
        return df, df_summary

    @classmethod
    def plot_spl(cls, df):
        fig, _ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        sns.scatterplot(data=df, x='hour', y='spl', hue='day')
        fig.set_tight_layout('tight')
        plt.show(block=False)

    def ac_print(self, df, by_days: bool = False, save_dir=None):
        """Build acoustic prints and matrices.
        When by_days=False, aggregates per site (original ac_print).
        When by_days=True, aggregates per (site, day) (original ac_print_by_days).
        Populates self.X, self.XX, self.sites_ind, and self.meta_rows accordingly.
        """
        plt.ion()
        group_site = df.groupby(['site'])
        Ls = len(group_site)

        group_hours= df.groupby(['hour'])
        lh = len(group_hours)

        # Prepare allocation based on mode
        if not by_days:
            L_treatments = Ls
        else:
            treatments = []
            for site_tmp, data_site_tmp in group_site:
                for day_tmp, _ in data_site_tmp.groupby(['day']):
                    treatments.append((site_tmp, day_tmp))
            L_treatments = len(treatments)
            print("Número de combinaciones (sitio-día): " + str(L_treatments))

        self.XX = np.zeros([65, lh, L_treatments])
        self.X = np.zeros([L_treatments, 65 * lh])
        self.sites_ind = []
        self.meta_rows = []

        k = 0
        for site, data_site in group_site:
            if not by_days:
                # Per-site aggregation
                print('site: ' + str(site) + ' # ' + str(k + 1) + ' of ' + str(Ls) + ' sites')
                Sd, tn, fn = self.build_acoustic_print(data_site)
                self.XX[:, :, k] = Sd
                self.X[k, :] = np.ravel(Sd, order='C')
                self.plot_acoustic_print(k,save_dir)
                try:
                    group_label = data_site['group'].iloc[0]
                except Exception:
                    group_label = site
                self.meta_rows.append({'treatment_idx': k, 'site': site, 'group': group_label})
                self.sites_ind.append(k)
                #fig_site.canvas.draw_idle()
                plt.show(block=False)
                plt.pause(0.05)
                k = k + 1
                print('progress: sitio ' + str(k) + ' of ' + str(Ls))
            else:
                # Per (site, day) aggregation
                group_day_site = data_site.groupby(['day'])
                Ld = len(group_day_site)
                for day, data_day in group_day_site:
                    print('dia: ' + str(day) + ' # ' + ' of ' + str(Ld) + ' days')
                    Sd, tn, fn = self.build_acoustic_print(data_day)
                    self.XX[:, :, k] = Sd
                    self.X[k, :] = np.ravel(Sd, order='C')
                    self.plot_acoustic_print(k,save_dir)

                    try:
                        group_label = data_day['group'].iloc[0]
                    except Exception:
                        group_label = site
                    self.meta_rows.append({'treatment_idx': k, 'site': site, 'day': day, 'group': group_label})
                    self.sites_ind.append(k)
                    k = k + 1
                    print('progreso: combinación sitio-día ' + str(k) + ' de ' + str(L_treatments))

    def build_acoustic_print(self, data_subset):
        """Compute the acoustic print matrix (Sd) for a subset of metadata rows.

        The subset is expected to belong to a single site (and optionally a single day).
        Rows are grouped by 'hour', each hour's spectra are averaged, converted to dB,
        masked, and downscaled to produce a compact acoustic print.

        Returns:
            Sd: 2D numpy array [65 x 48] acoustic print
            tn: 1D numpy array of hour bins (for plotting extent)
            fn: 1D numpy array of frequency bins from last spectrum
        """
        group_hour = data_subset.groupby(['hour'])
        Lh = len(group_hour)
        i = 0
        Stt = np.zeros([513, Lh])
        tn = np.zeros([Lh])
        fn = None
        for hour, rows in group_hour:
            tn[i] = sum(hour)
            L = len(rows)
            St = np.zeros(513)
            for _, row in rows.iterrows():
                file = row['route']
                s, fs = sound.load(file)
                if fs != 48000:
                    s = sound.resample(s, fs, 48000, res_type='kaiser_fast')
                    fs = 48000
                Stmp, fn = sound.spectrum(s, fs, window='hann', nperseg=1024, noverlap=512)
                St = St + Stmp
            St = St / max(L, 1)
            Stt[:, i] = St
            i = i + 1
            print('progress: ' + str(i) + ' of ' + str(Lh) + ' hours')
        # Robust dB transform using a modal reference within the informative band
        values, counts = np.unique(Stt[128:384, :], return_counts=True)
        mc = values[counts.argmax()] if values.size > 0 else 1.0
        if mc == 0:
            mc = 1.0
        Stt_db = 20 * np.log10(Stt / mc)
        mask = Stt_db > 0
        Sm = Stt_db * mask
        Sd = downscale_local_mean(Sm, (8, 1))
        return Sd, tn, fn
    
    def plot_acoustic_print(self, idx, save_dir=None):
        """Plot the acoustic print for a given treatment index with metadata-derived labels when available."""
        if idx < 0 or idx >= self.XX.shape[2]:
            print(f"Index {idx} out of range for acoustic prints with length {self.XX.shape[2]}")
            return

        # Derive labels from metadata when available, otherwise use generic labels
        if idx < len(self.meta_rows):
            meta = self.meta_rows[idx]
            site_label = str(meta.get('site', f'Treatment {idx}'))
            day_label = meta.get('day')
        else:
            site_label = f'Treatment {idx}'
            day_label = None

        title = f"Acoustic Print - {site_label} (Treatment {idx})"
        if day_label is not None:
            title += f" - Day {day_label}"

        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel('Hour')
        ax.set_ylabel('Frequency (kHz)')
        im = ax.imshow(self.XX[:, :, idx], aspect='auto', origin='lower', extent=[0, 24, 0, 24])
        fig.tight_layout()
        safe_label = ''.join(c if (c.isalnum() or c in '._- ') else '_' for c in site_label)
        if save_dir is not None:
            fig.savefig(f"{save_dir}/site_print_{safe_label}.png", dpi=150, bbox_inches='tight')
        fig.colorbar(im, ax=ax, label='Intensity (dB)')
        plt.show(block=False)
        plt.pause(0.05)
        fig.canvas.draw_idle()

    
    def calculate_nmds(self):
        dist_euclid = euclidean_distances(self.X)
        metric = True
        dist_matrix = dist_euclid
        y = np.transpose(self.sites_ind)
        mds = MDS(metric=metric, dissimilarity='precomputed', random_state=0)
        pts = mds.fit_transform(dist_matrix)

        # Align metadata and point counts if they differ to avoid pandas length errors
        if len(self.meta_rows) != pts.shape[0]:
            print(f"Warning: metadata rows ({len(self.meta_rows)}) != points ({pts.shape[0]}). Aligning sizes.")
            n = min(len(self.meta_rows), pts.shape[0])
            pts = pts[:n, :]
            self.X = self.X[:n, :]
            self.sites_ind = self.sites_ind[:n]

        # Build metadata DataFrame aligned to the points and include NMDS coordinates
        npts = pts.shape[0]
        df_meta = pd.DataFrame(self.meta_rows[:npts])
        df_meta['NMDSx'] = pts[:, 0]
        df_meta['NMDSy'] = pts[:, 1]

        # Create a new figure for the MDS scatter (do not reuse figure 1)
        fig2 = plt.figure(figsize=(10, 10))
        ax2 = fig2.add_subplot()
        plt.scatter(pts[:, 0], pts[:, 1])
        for x, ind in zip(self.X, range(pts.shape[0])):
            im = x.reshape(65, 48)
            imagebox = OffsetImage(im, zoom=0.7, cmap=cm.get_cmap('viridis'))
            i = pts[ind, 0]
            j = pts[ind, 1]
            ab = AnnotationBbox(imagebox, (i, j), frameon=False)
            ax2.add_artist(ab)
            ax2.invert_yaxis()
        plt.show(block=False)

        fig3 = plt.figure(figsize=(10, 10))
        ax3 = fig3.add_subplot()
        # Use 'group' labels for hue if present; fallback to 'site'
        hue_labels = df_meta['group'].tolist() if 'group' in df_meta.columns else df_meta['site'].tolist()
        sns.scatterplot(x=pts[:, 0], y=pts[:, 1], hue=hue_labels, palette='pastel', ax=ax3)
        plt.title('Metric NMDS with Euclidean distances')
        plt.ylabel('NMDSy')
        plt.xlabel('NMDSx')
        plt.show(block=False)
        # Return matrices, labels, points, and enriched metadata
        print('Non parametric multidimensional scaling (NMDS) calculated successfully')
        return (self.X, y, pts, df_meta)