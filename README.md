# How to run?

### STEPS:


### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n llmapp python=3.11 -y
```

```bash
conda activate llmapp
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

Now,
```bash
open up you local host and port
```
