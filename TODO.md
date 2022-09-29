# Ideas for implementation with this code

## Flat field correction

- Guess the ideal mask dimensions by reducing the sinogram to the x axis and calculating the xmin and xmax for the masks;
    -Sum/mean through axis =0, then through axis = 1, find the baseline/background, find where the signal is higher than the background;
- Work on saving the the flat decomposition for each experiment.
- Flat field interpolation over time series
- Make the correction of the projections a class method within the base class


## Reconstruction

- Incorporate nabu as a reconstruction engine to improve speed (at least with FBP algorithm);
