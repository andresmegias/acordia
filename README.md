# Acordia

Python 3 script intended to create guitar chord charts for any guitar tuning and any chord. The tuning, chord and some other options are specified in a configuration file with YAML formatting style. The script contains an artificial neural network that helps to determine which potential fret diagrams are actually playable.

In order to run Acordia, you need the main script (`acordia.py`) and a configuration file (like `config.yaml`). Also, if you want to use the neural network (which is done by default) you need the file `weights.npy`, which contains the parameters of the trained neural network.

You can check the example configuration file (`config.yaml`) to see the different options - it is quite self-explicative. Acordia can be run directly from the terminal or also within a graphical editor like Spyder. After downloading all the files, enter to the Acordia folder. To run it from the terminal, you can type `python3 acordia.py config.yaml`, using the corresponding path and name of your configuration file. To run it from a graphical editor, you can change the path of the configuration file at the beginning of the script, changing the value of the variable `config_file`, and then run the script.

After running Acordia, several plots will be shown: first, an scheme of the guitar fretboard with the notes of the input chord marked with colors; then, a chord chart (or charts) with the resulting chord diagrams. You can export these images with the options of the plot window or also marking the corresponding option in the configuration file.

Acordia uses the libraries PyYAML, NumPy and Matplotlib, and its neural network was trained using the library Keras (built with TensorFlow).

**Some examples**

F chord, standard tuning:
![image1](example-images/F-standard.jpg?raw=true)

F chord, all-fourths tuning:
![image2](example-images/F-allfourths.jpg?raw=true)
