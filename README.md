# ToFSim

Simulation Engine for Time-of-Flight Imaging. 

The simulation code assumes that there is not multi-path interference. 

The script `ToFSinglePixel.py` does the brightness measurements calculations, adds noise, and computes depths. If you want to simulate an entire scene you can modify the variable `depth` and change it to a list of depths associated to each point in the scene. Make sure that the depths of the scene are within the unambigous depth range of your functions (see `dMax` variable in the script). You can control the signal-to-noise-ratio (SNR) of the scene by varying: source average power (`pAveSourcePerPixel`), ambient average power (`pAveAmbientPerPixel`), mean albedo/reflectivity (`meanBeta`), integration/exposure time (`T`). 

The script `CalculateCodingSchemeMeanExpectedDepthError.py` shows how to calculate the mean expected depth error for a given coding scheme, under a set of SNR parameters. Figure 4 in [Practical Coding Function Design for Time-of-Flight Imaging](http://openaccess.thecvf.com/content_CVPR_2019/papers/Gutierrez-Barragan_Practical_Coding_Function_Design_for_Time-Of-Flight_Imaging_CVPR_2019_paper.pdf) was generated by running this type of simulation for a large number of SNR parameter combinations.

The script `VisualizeCodingScheme.py` plots the modulation, demodulation, and correlation functions for a given coding scheme.  

### References

If you use this code please make sure that you cite the following papers:

* *Practical Coding Function Design for Time-of-Flight Imaging*. Felipe Gutierrez-Barragan, Syed Azer Reza, Andreas Velten, Mohit Gupta. CVPR 2019.

* *What are the Optimal Coding Functions for Time-of-Flight Imaging?*. Mohit Gupta, S Nayar, A Velten, Eric Breitbach. ACM TOG, presented at SIGGRAPH 2018.

### Python Environment Setup

Follow the steps in this section to setup an anaconda virtual environment that contains all the required dependencies/libraries to run the code in this repository.

1. **Install miniconda**
2. **Create anaconda environment:** The following command creates an anaconda environment called `tofenv` with python 3.5.
```conda create -n tofenv python=3.5 ```

3. **Activate environment:** 
```source activate tofenv```

4. **Install libraries:** The code in this repository uses numpy, scipy, pandas, matplotlib, scikit-learn, and ipython (for development). To install all these dependencies run the following command:
```conda install numpy scipy matplotlib ipython pandas scikit-learn```

**Note:** If directly installing the packages with the above commands does not work it is probably because different versions of the libraries were installed. If this happened remove the environment and start over with the following steps.

1. Install miniconda and clone this repository.
2. Navigate to the folder containing this repos.
3. Use the `tofenv.yml` file to create the conda environment by running the following command: 
```conda env create -f tofenv.yml```. 
This command will setup the exact same conda environment we are currently using with all library versions being the same.

For more details on how to manage a conda environment take a look at this webpage: [Managing Conda Environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#sharing-an-environment).


