# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 11:07:09 2023

@author: Esteban
"""
import os
from tkinter.filedialog import askdirectory
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from resampy import resample
from scipy.io import wavfile
from datetime import time
from maad import sound


class Loader:
    def __init__(self):
        pass

    @staticmethod
    def find(s, ch):
        """Find all indices of character 'ch' in string 's'."""
        return [i for i, ltr in enumerate(s) if ltr == ch]

    @classmethod
    def create_df(cls, path, flist, recorder, fload):
        """Create a DataFrame with metadata extracted from audio file names."""
        d=pd.DataFrame()
        L=str(len(flist))
        for i,j in enumerate(flist):
            if fload=='file':
                file = os.path.basename(flist)
            else:
                file = os.path.basename(j)
                
            if recorder=='Grillo: nombre_aammdd_hhmmss':
                ind=cls.find(file,'_')
                ind.append(file.find('.'))
                d.loc[i,'route']=flist[i]
                d.loc[i,'file']=file
                d.loc[i,'site']=file[0:ind[0]]
                d.loc[i,'date_fmt']=pd.to_datetime(file[ind[0]+1:ind[1]]+' '+file[ind[1]+1:ind[2]], format='%Y%m%d %H%M%S')
                d.loc[i,'day']=file[ind[0]+1:ind[1]]
                d.loc[i,'hour']=d.loc[i,'date_fmt'].hour+d.loc[i,'date_fmt'].minute/60
                d.loc[i,'hourfmt']=pd.to_datetime(file[ind[1]+1:ind[2]], format='%H%M%S').time()

            elif recorder=='Audiomoth: aammdd_hhmmss':
                site = ''
                if fload=='file' or fload == 'folder':
                    # Use folder name as site
                    site = os.path.basename(path)
                elif fload == 'set':
                    # Parent folder of the file is the site
                    site = os.path.basename(os.path.dirname(j))
                    
                ind=cls.find(file,'_')
                ind.append(file.find('.'))
                d.loc[i,'route']=flist[i]
                d.loc[i,'file']=file
                d.loc[i,'site']=site
                d.loc[i,'date_fmt']=pd.to_datetime(file[0:ind[0]]+' '+file[ind[0]+1:ind[1]], format='%Y%m%d %H%M%S')
                d.loc[i,'day']=file[0:ind[0]]
                d.loc[i,'hour']=d.loc[i,'date_fmt'].hour+d.loc[i,'date_fmt'].minute/60
                d.loc[i,'hourfmt']=pd.to_datetime(file[ind[0]+1:ind[1]], format='%H%M%S').time()

            if fload == 'file': # Break for cycle so flist is assigned to route in the first iteration. This is only necessary for files
                d.loc[0,'route']=flist
                break
            
            print('progress: ' + str(i+1) +' of ' + L)    
        return(d)  
    
    @classmethod
    def plot_folder_stats(cls, df):
        """Plot the distribution of recordings by day and hour."""
        days, indd = np.unique(df.day, return_inverse=True)   
        hour=df.hour
        hours, indh = np.unique(hour, return_inverse=True)   

        countd=np.unique(indd, return_counts=True)
        counth=np.unique(indh, return_counts=True)
        
        plt.figure(figsize=(10,6))
        plt.subplot(1,2,1)
        plt.barh(days, countd[1], align='center', alpha=0.5)
        plt.ylabel('Day')
        plt.xlabel('File count')
        plt.subplot(1,2,2)
        plt.barh(hours, counth[1], align='center', alpha=0.5)
        plt.ylabel('Hour')
        plt.xlabel('File count')
        plt.show()

    @classmethod
    def plot_set_recorders(cls, df):
        """Plot the distribution of recordings by site and day."""
        d = df.groupby('site')
        dc = pd.DataFrame(d.day.value_counts())
        dc.rename(columns={'day': 'count'}, inplace=True)
        dc.sort_values(by='day', ascending=True, inplace=True)
        sns.scatterplot(y='day', x='site', size='count', hue='count',
                        size_norm=(10, 100), hue_norm=(10, 100), data=dc)
        plt.show()
        df_summary = {}
        for i, df_site in df.groupby('site'):
            site_summary = {
                'date_ini': str(df_site.date_fmt.min()),
                'date_end': str(df_site.date_fmt.max()),
                'n_recordings': len(df_site),
                'duration': str(df_site.date_fmt.max() - df_site.date_fmt.min()),
                'time_diff': df_site['date_fmt'].sort_values().diff().median(),
            }
            df_summary[i] = site_summary
        df_summary = pd.DataFrame(df_summary).T
        df_new = df_summary.reset_index().rename(columns={'index': 'site'})
        return df_new

class DataFrameEditor:
    """Class for editing and displaying DataFrames."""
    def __init__(self):
        pass

    @classmethod
    def update_df(cls, df, txt):
        """Update the DataFrame to include only recordings within a specified date range.

        Expects input like 'YYYYMMDD:YYYYMMDD'.
        """
        parts = str(txt).split(':')
        if len(parts) != 2:
            return pd.DataFrame()
        init = pd.to_datetime(parts[0], format='%Y%m%d')
        fin = pd.to_datetime(parts[1], format='%Y%m%d')
        df_day = df[(df['date_fmt'] >= init) & (df['date_fmt'] < fin)]
        return df_day
    
    @classmethod
    def show_dataframe(cls, df_to_show: pd.DataFrame, title: str = "DataFrame",root_window=None):
        """Open a new Tkinter window with a scrollable table (ttk.Treeview) to display a DataFrame.

        - Shows columns as headers
        - Adds vertical and horizontal scrollbars
        - Adapts column widths based on header length
        """
        if df_to_show is None or (isinstance(df_to_show, pd.DataFrame) and df_to_show.empty):
            print('No hay metadatos para mostrar')
            return

        top = tk.Toplevel(root_window.window if root_window else None)
        top.title(title)
        top.geometry("900x500")

        container = ttk.Frame(top)
        container.pack(fill='both', expand=True)

        # Create Treeview
        cols = list(df_to_show.columns)
        tree = ttk.Treeview(container, columns=cols, show='headings')

        # Scrollbars
        vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Grid placement
        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Configure headings and column widths
        for c in cols:
            tree.heading(c, text=str(c))
            # Set a reasonable default width based on header length
            width = max(100, min(300, 10 * len(str(c))))
            tree.column(c, width=width, stretch=True, anchor='w')

        # Insert data rows
        for _, row in df_to_show.iterrows():
            values = [row[c] for c in cols]
            # Convert complex objects to string for display
            values = [str(v) if not (isinstance(v, (int, float, str)) or pd.isna(v)) else ("" if pd.isna(v) else v) for v in values]
            tree.insert('', 'end', values=values)

    @classmethod
    def select_days_gui(cls, df_ref: pd.DataFrame, root_window=None) -> pd.DataFrame:
        """Open a simple GUI to select start/end days and return a filtered DataFrame.

        - Lists unique values from df_ref['day'] in two dropdowns (start/end)
        - Returns the filtered DataFrame using update_df(start:end)
        - Returns an empty DataFrame if cancelled or invalid
        """
        if not isinstance(df_ref, pd.DataFrame) or df_ref.empty or 'day' not in df_ref.columns:
            print('No hay metadatos cargados o falta la columna "day"')
            return pd.DataFrame()

        # Gather unique days, keep as strings, sorted
        try:
            day_values = pd.unique(df_ref['day'])
        except Exception:
            day_values = []
        day_values = [str(d) for d in day_values if pd.notna(d)]
        if not day_values:
            print('No hay valores de días disponibles')
            return pd.DataFrame()
        day_values = sorted(day_values)

        parent = root_window.window if root_window and hasattr(root_window, 'window') else None
        top = tk.Toplevel(parent)
        top.title('Seleccionar días a analizar')
        top.geometry('420x220')

        # Start day selector
        tk.Label(top, text='Día inicial (YYYYMMDD):').pack(padx=10, pady=(12, 2), anchor='w')
        start_var = tk.StringVar(value=day_values[0])
        start_combo = ttk.Combobox(top, textvariable=start_var, values=day_values, state='readonly')
        start_combo.pack(fill='x', padx=10, pady=(0, 8))

        # End day selector
        tk.Label(top, text='Día final (YYYYMMDD):').pack(padx=10, pady=(4, 2), anchor='w')
        end_var = tk.StringVar(value=day_values[-1])
        end_combo = ttk.Combobox(top, textvariable=end_var, values=day_values, state='readonly')
        end_combo.pack(fill='x', padx=10, pady=(0, 12))

        # Result holder
        result = {'df': None}

        def on_save():
            start = start_var.get().strip()
            end = end_var.get().strip()
            if not start or not end:
                print('Seleccione ambos días')
                return
            # Validate ordering using lexicographic since YYYYMMDD
            if start > end:
                print('El día inicial debe ser menor o igual al final')
                return
            try:
                df_new = cls.update_df(df_ref, f"{start}:{end}")
            except Exception as e:
                print(f'Error al filtrar por días: {e}')
                df_new = pd.DataFrame()
            result['df'] = df_new
            top.destroy()

        def on_cancel():
            top.destroy()

        btns = tk.Frame(top)
        btns.pack(fill='x', padx=10, pady=(0, 10))
        tk.Button(btns, text='Guardar', command=on_save).pack(side='left')
        tk.Button(btns, text='Cancelar', command=on_cancel).pack(side='left', padx=8)

        # Block until window closes so we can return synchronously
        top.wait_window()
        return result['df'] if isinstance(result.get('df'), pd.DataFrame) else pd.DataFrame()

    @classmethod
    def assign_groups_gui(cls, df_ref: pd.DataFrame, root_window=None):
        """GUI to assign a group label to each recorder and add 'group' to df_ref.

        Steps:
        - Select the recorder-identifying column from a dropdown.
        - Enter a group label for each unique recorder value.
        - Click Guardar to write df_ref['group'].
        """
        if not isinstance(df_ref, pd.DataFrame) or df_ref.empty:
            print('No hay metadatos cargados')
            return

        parent = root_window.window if root_window and hasattr(root_window, 'window') else None
        top = tk.Toplevel(parent)
        top.title('Asignar grupos de grabadoras')
        top.geometry('520x520')

        tk.Label(top, text='Seleccione la columna que identifica la grabadora:').pack(padx=10, pady=(10, 0), anchor='w')
        # Exclude 'route' and 'file' from selectable columns
        cols = [c for c in list(df_ref.columns) if c != 'route' and c != 'file']
        col_var = tk.StringVar(value=cols[0] if cols else '')
        col_combo = ttk.Combobox(top, textvariable=col_var, values=cols, state='readonly')
        col_combo.pack(fill='x', padx=10, pady=6)

        container = tk.Frame(top)
        container.pack(fill='both', expand=True, padx=10, pady=10)

        canvas = tk.Canvas(container, borderwidth=0)
        scrollbar = ttk.Scrollbar(container, orient='vertical', command=canvas.yview)
        form_frame = tk.Frame(canvas)

        form_frame.bind(
            '<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all'))
        )
        canvas.create_window((0, 0), window=form_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        mapping_entries = {}

        def build_form(selected_col):
            for wdg in list(form_frame.children.values()):
                wdg.destroy()
            mapping_entries.clear()

            try:
                unique_recorders = pd.unique(df_ref[selected_col])
            except Exception:
                unique_recorders = []
            if unique_recorders is None:
                unique_recorders = []

            tk.Label(form_frame, text=f'Asigne grupo a cada valor en "{selected_col}":').grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 6))
            for i, r in enumerate(unique_recorders, start=1):
                tk.Label(form_frame, text=str(r)).grid(row=i, column=0, sticky='w', padx=(0, 8), pady=3)
                entry = tk.Entry(form_frame)
                entry.grid(row=i, column=1, sticky='ew', pady=3)
                form_frame.grid_columnconfigure(1, weight=1)
                mapping_entries[r] = entry

        build_form(col_var.get())

        def on_col_change(_event=None):
            build_form(col_var.get())

        col_combo.bind('<<ComboboxSelected>>', on_col_change)

        btns = tk.Frame(top)
        btns.pack(fill='x', padx=10, pady=(0, 10))

        def on_save():
            selected_col = col_var.get()
            if selected_col not in df_ref.columns:
                print(f'La columna "{selected_col}" no existe en df_ref')
                return
            # Only save text explicitly entered by the user; blanks become NaN
            mapping = {}
            for r, entry in mapping_entries.items():
                val = entry.get().strip()
                mapping[r] = (val if val != '' else None)
            try:
                df_ref['group'] = df_ref[selected_col].map(mapping)
                df_ref['group'] = df_ref['group'].fillna('Sin grupo')
                print('Columna "group" agregada y actualizada en df.md')
            except Exception as e:
                print(f'Error al asignar grupos: {e}')
                return
            top.destroy()

        tk.Button(btns, text='Guardar', command=on_save).pack(side='left')
        tk.Button(btns, text='Cancelar', command=top.destroy).pack(side='left', padx=8)
    
    @classmethod
    def fix_hours_by_intervals_gui(cls, df_ref: pd.DataFrame, root_window=None) -> pd.DataFrame:
        """GUI to snap 'hour' values to user-defined intervals within each hour.

        - Asks for the number of recordings per hour (N)
        - Rounds `hour` to the nearest multiple of 1/N (e.g., N=10 -> 0.1 hour = 6 min)
        - Updates `hour` in-place and also refreshes `hourfmt` to match the snapped time
        - Filters out days whose total recordings are not exactly N*24
        - Returns the updated and filtered DataFrame (or the original if cancelled/invalid)
        """
        if not isinstance(df_ref, pd.DataFrame) or df_ref.empty or 'hour' not in df_ref.columns:
            print('No hay metadatos cargados o falta la columna "hour"')
            return df_ref if isinstance(df_ref, pd.DataFrame) else pd.DataFrame()

        parent = root_window.window if root_window and hasattr(root_window, 'window') else None
        top = tk.Toplevel(parent)
        top.title('Corregir hora por intervalos')
        top.geometry('420x180')

        tk.Label(top, text='Grabaciones por hora (N):').pack(padx=10, pady=(12, 4), anchor='w')
        n_var = tk.StringVar(value='10')
        n_entry = tk.Entry(top, textvariable=n_var)
        n_entry.pack(fill='x', padx=10, pady=(0, 12))

        result = {'df': df_ref}

        def _snap_hours(df: pd.DataFrame, n: int) -> pd.DataFrame:
            if n <= 0:
                raise ValueError('N debe ser un entero positivo')
            # Snap hour to nearest multiple of 1/n
            snapped = (df['hour'] * n).round() / n
            df['hour'] = snapped

            # Update hourfmt to match snapped time
            def _to_time(h: float) -> time:
                hr = int(np.floor(h))
                mins = int(round((h - hr) * 60))
                # Handle 60-minute rounding overflow
                if mins == 60:
                    hr = (hr + 1) % 24
                    mins = 0
                return time(hr, mins, 0)

            try:
                df['hourfmt'] = df['hour'].apply(_to_time)
            except Exception:
                # If hourfmt doesn't exist or fails, ignore silently
                pass

            # Filter days with exactly N*24 recordings per site (if available)
            expected_per_day = n * 24
            if 'day' in df.columns:
                if 'site' in df.columns:
                    counts_by_site_day = df.groupby(['site', 'day']).size()
                    valid_site_days = counts_by_site_day[counts_by_site_day == expected_per_day].index
                    df = df[df.set_index(['site', 'day']).index.isin(valid_site_days)]
                    print(f"Días válidos con {expected_per_day} grabaciones por sitio: {len(valid_site_days)} pares sitio-día")
                else:
                    counts_by_day = df.groupby('day').size()
                    valid_days = counts_by_day[counts_by_day == expected_per_day].index
                    df = df[df['day'].isin(valid_days)]
                    print(f"Días válidos con {expected_per_day} grabaciones: {len(valid_days)}")
            else:
                print('Advertencia: no se encontró la columna "day" para filtrar por días')
            return df

        def on_save():
            raw = n_var.get().strip()
            try:
                n = int(raw)
                if n <= 0:
                    raise ValueError
            except Exception:
                print('Ingrese un número entero positivo para N')
                return
            try:
                df_out = _snap_hours(df_ref, n)
            except Exception as e:
                print(f'Error al corregir horas: {e}')
                df_out = df_ref
            result['df'] = df_out
            print(f'Horas corregidas usando N={n} segmentos por hora')
            top.destroy()

        def on_cancel():
            top.destroy()

        btns = tk.Frame(top)
        btns.pack(fill='x', padx=10, pady=(0, 10))
        tk.Button(btns, text='Guardar', command=on_save).pack(side='left')
        tk.Button(btns, text='Cancelar', command=on_cancel).pack(side='left', padx=8)

        top.wait_window()
        Loader.plot_set_recorders(result['df'])
        return result['df']
    
class Sampler:
    """Class for resampling audio files in the dataset."""
    def __init__(self):
        pass

    def resample_dataset(self,df,ftarget=48000):
        """Resample audio files in the dataset to a target sampling frequency."""
        save_dir=askdirectory(title="Seleccione carpeta para guardar los archivos resampleados")   
        print('Resampleando archivos...')
        for i, row in df.iterrows():
            p=Path(row.route)
            folder=p.parent.name
            save_path = os.path.join(save_dir, folder)
            filename= save_dir + '/' + str(folder) + '/' + str(p.name)
            if os.path.isfile(filename):
                print("Archivo ya existe")  
            else:
            
                try :
                    fs, s = wavfile.read(row.route)
                    print('Archivo cargado: ' + str(row.route))
                except ValueError:
                    print('Error resampling file: ' + str(row.route))

                if fs != ftarget:
                    s_res = resample(s, fs, ftarget, res_type='kaiser_fast')
                else:
                    s_res = s

                s_res = sound.select_bandwidth(x=s_res, fs=ftarget, fcut=1000, forder=4,fname='butter', ftype='highpass') 
        
                try:
                    # Create the directory
                    os.mkdir(save_path)
                    print(f"Directory created: {save_path}")
                except FileExistsError:
                    print("Carpeta ya existe")

                wavfile.write(filename, ftarget, s_res.astype(np.int16)) # Save the resampled audio file format as 16-bit PCM
                print('Archivo creado:')
                print(filename)

            print('progreso: ' + str(i) + ' de ' + str(len(df)))
        