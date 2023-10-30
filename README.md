# kg-schema-extraction

## What is this repo about?
This repository provides the complete code needed to execute `KGSEC` tool, a tool developed to tackle the schema extraction task of knowledge graphs. The repo has been developed as a part of the `stelar-eu` project and is purposed to support also the submission of a paper titled: 
```
KGSEC: A Modular Framework for Knowledge Graph Schema Extraction and Comparison
```
submitted at [*The Demonstrations Track of ICDE 2024*](https://icde2024.github.io/CFP_demos.html).


## How is the repo structured? 
This repo contains the following core directories: `back-end`, `front-end` and `notebooks`.
* `back-end`: contains the code for the `flask-api`, which powers the whole application. This module accesses the key functions that provide schema extraction functionality for the users and provides results through api-calls. 
* `front-end`: contains the code for a Python application developed in `Dash.js`. The module allows users to access the application's functionality through an easy-to-use UI, submit their requests to the back-end and analyze the results, with tables and visualizations. 
* `notebooks`: contains `.ipynb` files that can be used to explore the functions of the application and experiment with them in a more hands-on manner. 

## How to run this code?
The `front-end` of the application relies on the `back-end` to work. To run the `back-end` users need to navigate to the `back-end` folder 
```
$ cd back-end
```
and then execute the base `app.py` script that activates the server of the rest-api
```
$ python3 app.py
```
Then while this service is running, users should open a new terminal navigate to the `front-end` folder
```
$ cd front-end
```
and then execute the base `run.py` script that begins the browser application
```
$ python3 run.py
```

> 🚧 **Constant Updating**
> 
> This repo is still under construction.