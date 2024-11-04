# chest_xray_torch_vision_app

#### Use Python 3.9 Version 

### Follow the steps to setup the env
* Create Conda env 
* activate conda env
* Run requirements file
* 
```sh
conda create -n torch-02 python=3.9
conda activate torch-02
pip install -r requirements.txt
```

### Train the model
```sh
python chest_xray_train_model_main.py
```

### Test the model
```sh
python chest_xray_train_model_main_test.py
```