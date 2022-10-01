# Acordia
**Guitar chord chart generator**

Acordia is a Python 3 script intended to create guitar chord charts for any guitar tuning and any chord. The tuning, chord and some other options are specified in configuration file with YAML formatting style. The script contains a trained neural network that helps to determine which potential fret diagrams are actually playable.

You can check the example configuration file (`config.yaml`) to see the different options. As it uses the YAML formatting style, it is quite self-explicative. Acordia can be run directly from the terminal or also within a graphical editor like Spyder. To run it from the terminal, you can type `python3 acordia.py config.yaml`, using the corresponding path and name of your configuration file. To run it from a graphical editor like Spyder, you can change the path of the configuration file at the beginning of the script, changing the value of the variable `config_file`.

After running Acordia, several plots will be shown: first, an scheme of the guitar fretboard with the notes of the input chord marked with colors; then, a chord chart (or charts) with the resulting chord diagrams. You can export these images with the options of the plot window or also marking the corresponding option in the configuration file.

Acordia uses the libraries PyYAML, NumPy and Matplotlib, and its neural network was trained using the library Keras (built with TensorFlow).