# ToFSim

Simulation Engine for Time-of-Flight Imaging. 

The simulation code assumes that there is not multi-path interference. 

Given a ground truth depth or depth map and the location of the ToF sensor with respect to the depth map, this code calculates the brightness measurements that are obtained with a given coding scheme. The `Decoding.py` file can take the brightness images and output the depth for each point. 

The script `ToFSinglePixel.py` does the brightness measurements calculations, adds noise, and computes depths. If you want to simulate an entire scene you can modify the variable `depth` and change it to a list of depths associated to each point in the scene. Make sure that the depths of the scene are within the unambigous depth range of your functions (see `dMax` variable in the script). You can control the SNR of the scene by varying: source average power (`pAveSourcePerPixel`), ambient average power (`pAveAmbientPerPixel`), mean albedo/reflectivity (`meanBeta`), integration/exposure time (`T`). 

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
