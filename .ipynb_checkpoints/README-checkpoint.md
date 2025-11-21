# UNSW-PANOPTES-PC-HPE
mmWave radar point cloud based human pose estimation

## Environment Setup

1. Download dataset pickle files from the oneDrive folder shared with you previously: `backup_data/UNSW-PANOPTES/pc_pose`. Store pickle files locally in the `data` folder (there is currently only one sample file located here).

2. Install [annaconda](https://www.anaconda.com/download)

3. Create a new conda environment: 

      `conda env create -f environment.yml`
4. Activate environment:
      
      `conda activate panoptes_pc_hpe`

5. Run `data_demo.ipynb` to learn how to use the dataset.


# All data of this project is prepared in the data folder, we can just run the code in the FinalVersionThesis