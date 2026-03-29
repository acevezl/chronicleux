# Notes and reminders

## Launch .venv (Python Virtual Environment)
```
source .venv/bin/activate
```

Note to future Louie: I've created a .venv inside chroncileux only. 
I did not create a global .venv - So I can work on multiple projects
at a time.

So: 
- If in (base) first deactivate: 

```
conda deactivate
```

- If in (.venv) (base) deactivate both: 
```
deactivate
conda deactivate
```

## Run django server

```
cd ~/chronicleux/src
python manage.py runserver
```

## Making changes to the model
Whenever a change to the model is needed, run this:

```
python manage.py makemigrations
python manage.py migrate
```

And restart the server

## Watch NPM changes on files
 
 ```
 npm run watch:css
 ```