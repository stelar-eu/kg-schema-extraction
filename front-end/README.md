# kg-schema-extraction

## What is this directory about?
This directory includes the `front-end` functionality of `KGSEC` tool. To run it the user should first make sure that all the dependencies of `requirements.txt` file are installed. Paths should be also configured in a `config.json` file, filled with data as explained in the `config_sample.json` file. The `back-end` module should be also running to handle the requests submitted by the `front-end`

To run the `front-end` users should run the following command:

```
$ python3 run.py
```
Then the api is accesible at endpoint: 
```
http://127.0.0.1:8050
```