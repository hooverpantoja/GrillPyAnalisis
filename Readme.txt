This code is written in Python, and it is graphic user interface GUI for processing audio data of Passive Acoustic Monitoring (PAM). The code generates spectrograms, and saving audio files. The application allows the user to select a folder containing WAV files recorded with different types of devices (e.g., SongMeter, Snap, Audiomoth, etc.), and generates a dataframe to store information about each audio file (i.e., file name, site, date, and hour). Once the data has been loaded, the user can select one day of recordings to generate a spectrogram that shows the frequency content of the audio over time.

Installation: Follow the installation guide file

Run the following command in the terminal:python maad_gui.py

Usage
1. Before running the application, make sure to read the installation guide and this readme file.

2. Select the type of recorder that was used to create the recordings. Currently, the application supports SongMeter, SM3, SM4, and Snap recorders.

3. Click on the "Load folder" button to select a folder containing WAV files.

4. Once the folder has been loaded, click on the "Select one day" button to choose the day that you want to analyze.

5. The application will ask you to select the frequency range of interest and the length of the signal to build the long wave or on-day spectrogram. The default values are 0-Sample frequency Hz and 10 seconds, respectively.

6. After selecting the frequency range and signal length, the application will generate a spectrogram for the selected day of recordings.

7. Finally, the user can save the generated WAV file by clicking on the "Save WAV file" button.

The main functions are:

find(s, ch): A function to find all occurrences of a character ch in a string s.
get_data_files(): A function that prompts the user to select a folder with .wav files and generates a pandas dataframe with metadata (site, day, hour).
one_day_spec(): A function that prompts the user to select a day to analyze and generates a spectrogram for all the .wav files corresponding to that day.
save_wav(): A function that prompts the user to select a folder to save a new .wav file and generates a .wav file with the concatenated audio from all the .wav files in the pandas dataframe.
The script also contains several functions to handle the GUI, such as enter_func(event), which is called when the user hits the "Enter" key after inputting data into the GUI.

To use the script, the user should run it in a Python environment and follow the prompts on the GUI. The user should select a folder with .wav files and specify the recorder type, and then select a day to analyze or a folder to save the concatenated .wav file.

Note: The maad library is used to generate the spectrograms. If this library is not installed, the user should install it using the command !pip install maad in their Python environment.

