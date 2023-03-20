# Ranking system

## It provide a mechanism for ranking set of available spaces for making further reservation process

---

## Project setup

Project setup instruction here.

Clone the project

```bash
  git clone https://github.com/Nicola-Ibrahim/Ranking-system.git
```

Go to the project directory

```bash
  cd Ranking-system
```

In Powershell: install poetry for package management using

```bash
  (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

Add poetry to system environment

```bash
  setx path "%path%;C:\Users\{%user_name%}\AppData\Roaming\Python\Scripts"
```

Change the virtualenv directory to current directory

```bash
  poetry config virtualenvs.in-project true
```

Install dependencies using poetry

```bash
  poetry install
```

Activate the created environment

```bash
  .venv\Scripts\activate
```

Run the ranking algorithm

```bash
  make run-server
```
