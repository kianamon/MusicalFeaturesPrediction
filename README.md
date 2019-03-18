# Musical Features Prediction Using Machine Learning Algorithms

[Kiana Montazeri](https://github.com/kianamon)<sup>1</sup>, [Farnaz Ghashami](https://github.com/FarnazGhashami)<sup>1</sup>,
[Shideh Shams Amiri](https://github.com/shidehsh)<sup>1</sup>.<br><br>
<sup>1</sup>[Drexel University, Philadelphia, PA](https://drexel.edu/cci/academics/information-science-department/)

[FMA]:       https://freemusicarchive.org

Free Music Archive is a website which provides us with very diverse and thorough range of information regarding audio music files across all genres and types. [This data is pre-processed by Michaël Defferrard et al.](https://arxiv.org/pdf/1612.01840.pdf) to a metadata file which contains four datasets with information about:

* The index is the ID of the song, taken from the website, used as the name of the audio file.

* Per-track, per-album and per-artist metadata from the Free Music Archive website.

These datasets can be used to evaluate many tasks such as prediction of musical features such as mood or instruments used in the song, etc. These datasets are pre-processed by the same group that was previously mentioned.
We are converting the data to a suitable format for our purpose. We use visualization tools to explore the datasets and in the end we will build an analytical model for the musical files using machine learning algorithm.

Analyzing audio is of interest to not only musicians, but people who want to study relations with higher-level representations in musical pieces.
We are planning to use these datasets:

More information on the datasets is available:
> at <https://github.com/mdeff/fma>.

## Data

All metadata and features for all tracks are distributed in
**[fma_metadata.zip]** (342 MiB). The below tables can be used with [pandas] or
any other data analysis tool. 
* `tracks.csv`: per track metadata such as ID, title, artist, genres, tags and
  play counts, for all 106,574 tracks.
* `genres.csv`: all 163 genre IDs with their name and parent (used to infer the
  genre hierarchy and top-level genres).
* `features.csv`: common features extracted with [librosa].
* `echonest.csv`: audio features provided by [Echonest] for a subset of 13,129 tracks.

[pandas]:   http://pandas.pydata.org/
[librosa]:  https://librosa.github.io/librosa/
[echonest]: http://the.echonest.com/

## Code

The following notebooks and scripts, stored in this repository, have been developed for this project.
1. [Music]: The main code and general information can be found here. The map of the notebooks is described in this file as well.
### Genre Prediction:
2. [ArtistsInput]: Develops a model for predicting the top genre based on track information provided in `tracks.csv`.
3. [GenrePrediction]: Develops two models for predicting the top genre based on audio features provided in `features.csv` and applies the model to couple of randomly selected songs of our choosing to predict the closest genre.
### Artist's Popularity Prediction:
4. [ArtistPopularity]: Develops a linear regression model for predicting artist's general popularity based on audio and social features of the track.
### Shide Prediction:
5. [ArtistPopularity]: Develops .

[Music]:  https://nbviewer.jupyter.org/github/kianamon/MusicalFeaturesPrediction/blob/master/Music.ipynb
[ArtistsInput]: https://nbviewer.jupyter.org/github/kianamon/MusicalFeaturesPrediction/blob/master/ArtistsInput.ipynb
[GenrePrediction]:  https://nbviewer.jupyter.org/github/kianamon/MusicalFeaturesPrediction/blob/master/GenrePrediction.ipynb
[ArtistPopularity]: https://nbviewer.jupyter.org/github/kianamon/MusicalFeaturesPrediction/blob/master/ArtistPopularity.ipynb
[ArtistPopularity]: https://nbviewer.jupyter.org/github/kianamon/MusicalFeaturesPrediction/blob/master/ArtistPopularity.ipynb

## Implementing Genre Prediction
Genre prediction model can be used on any .mp3 formatted audio file. In order to apply the model to the audio file, please run [Application](https://nbviewer.jupyter.org/github/kianamon/MusicalFeaturesPrediction/blob/master/Application.ipynb) notebook with the correct file path. (The error faced is due to using the code for only one song at a time and can be ignored.) After ececution of this code, a .csv data file will be produced in data directory and can be used for applying the model.

## Requirements
In order to use the notebooks of this project, you will need the following:<br>

| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

`pandas`<br>
`librosa`<br>
`numpy`<br>
`matplotlib`<br>
`seaborn`<br>
`scikit-learn`<br>
`requests`<br>
`pydot`<br>
`tqdm`<br>
`jupyter`<br>
`python-dotenv`<br>
`yellowbrick`<br>
`pytorch`<br>
<br>
For more information please email: <km3436@drexel.edu>
