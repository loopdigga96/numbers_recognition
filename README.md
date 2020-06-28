## Tutorial
- create virtualenv with python 3.6
- install requirements from requirements.txt
- download [dataset](https://drive.google.com/file/d/1loObyNl2GiIsZr9Dp5SI1JeXnGUFrIkp/view?usp=sharing) and put it near ``main.py`` file


## Training tutorial
- ```python main.py --config_path config.json``` 

## Inference tutorial
- Download [checkpoint](https://drive.google.com/file/d/1MRK2LPdda-bt9_hwz3EaGbQcJXyx345X/view?usp=sharing)
- ```python infer.py --checkpoint path/to/checkpoint.pt --config_path config.json --data_path path/to/data.csv```
- result will be in submission.csv

