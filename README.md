# Pore space segmentation 
This is done with `main_segmentation_dataset_gen.ipynb` or `main.py` Here we segement the pore space using watershed segmentation. 

- The pore space is loaded
![image](https://github.com/cspurin/image_processing/assets/108369280/700c0e83-9ce7-4ecc-b45e-cf232a63d3d4)

- The data is filtered using a non-local means filter
![image](https://github.com/cspurin/image_processing/assets/108369280/04f43a9d-23c7-44e9-9739-4573e11c56d5)

- Then the image is segmented using a watershed segmentation. Values for this are chosen by the user
![image](https://github.com/cspurin/image_processing/assets/108369280/da1dd766-0fe5-411d-8243-25162564bc78)

## Notes
- There is a single_run function in the notebook which will do the whole pipline at once
- Also there are single_run | pore_tresh_sa | filter_sa functions for sensitivity analysis of Pore tresh parametwe & filter patch parameter.
