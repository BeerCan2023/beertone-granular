# BeerTone Granular

![BeerTone Granular Logo](BeerToneGranular_icon.png)

An advanced granular synthesizer developed in Python.

## Description

BeerTone Granular is a granular synthesizer offering advanced real-time sound manipulation capabilities. This project allows loading audio samples and applying various granular effects to create unique sound textures.

## Features

- Audio file loading (WAV, MP3, FLAC, etc.)
- Granular synthesis with precise parameter control
- Intuitive user interface
- Parameter presets
- MIDI support

## Prerequisites

- Python 3.7 or higher
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/BeerCan2023/beertone-granular.git
   cd beertone-granular
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application with:

```bash
python main.py
```

## Project Structure

- `main.py` - Application entry point
- `gui_main.py` - Main user interface
- `custom_dial.py` - Custom UI components
- `splash/` - Splash screens
- `splash_launcher.py` - Launcher with splash screen

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

Pierre Aumont - [@BeerCan2023](https://github.com/BeerCan2023)

## Acknowledgments

This project makes use of the following open-source libraries:

- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - For the graphical user interface
- [NumPy](https://numpy.org/) - For numerical computing
- [Matplotlib](https://matplotlib.org/) - For visualization components
- [Librosa](https://librosa.org/) - For audio analysis
- [SciPy](https://www.scipy.org/) - For scientific computing
- [SoundFile](https://pysoundfile.readthedocs.io/) - For audio file I/O
- [SoundDevice](https://python-sounddevice.readthedocs.io/) - For real-time audio playback

Special thanks to the open-source community for their invaluable contributions to these projects.
