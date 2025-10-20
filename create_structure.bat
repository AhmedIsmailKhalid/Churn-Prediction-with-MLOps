@echo off
REM ==========================================================
REM Create MLOps Churn Prediction folder structure (no root folder)
REM Run this script FROM the root directory (mlops-churn-prediction)
REM ==========================================================

echo Creating project structure...

REM -------------------- .github/workflows --------------------
mkdir .github
mkdir .github\workflows
echo. > .github\workflows\ci.yml
echo. > .github\workflows\train-deploy.yml

REM -------------------- src structure ------------------------
mkdir src
echo. > src\__init__.py

REM Features
mkdir src\features
echo. > src\features\__init__.py
mkdir src\features\feature_repo
echo. > src\features\feature_repo\__init__.py
echo. > src\features\feature_repo\feature_definitions.py
echo. > src\features\feature_repo\feature_store.yaml

REM Pipelines
mkdir src\pipelines
echo. > src\pipelines\__init__.py
echo. > src\pipelines\training.py
echo. > src\pipelines\inference.py
echo. > src\pipelines\data_validation.py

REM Models
mkdir src\models
echo. > src\models\__init__.py
echo. > src\models\train.py
echo. > src\models\evaluate.py
echo. > src\models\preprocessing.py

REM API
mkdir src\api
echo. > src\api\__init__.py
echo. > src\api\main.py
echo. > src\api\schemas.py
echo. > src\api\dependencies.py
echo. > src\api\Dockerfile
echo. > src\api\requirements.txt

REM Monitoring
mkdir src\monitoring
echo. > src\monitoring\__init__.py
echo. > src\monitoring\metrics.py
echo. > src\monitoring\drift_detection.py

REM -------------------- tests ------------------------
mkdir tests
echo. > tests\__init__.py
mkdir tests\unit
echo. > tests\unit\__init__.py
echo. > tests\unit\test_models.py
echo. > tests\unit\test_features.py
echo. > tests\unit\test_api.py

mkdir tests\integration
echo. > tests\integration\__init__.py
echo. > tests\integration\test_pipeline.py
echo. > tests\conftest.py

REM -------------------- data ------------------------
mkdir data
mkdir data\raw
echo. > data\raw\.gitkeep
mkdir data\processed
echo. > data\processed\.gitkeep
mkdir data\simulated
echo. > data\simulated\.gitkeep

REM -------------------- notebooks ------------------------
mkdir notebooks
echo. > notebooks\00-complete-walkthrough.ipynb
echo. > notebooks\01-eda.ipynb
echo. > notebooks\02-feature-engineering.ipynb

REM -------------------- monitoring ------------------------
mkdir monitoring
echo. > monitoring\prometheus.yml
mkdir monitoring\grafana
mkdir monitoring\grafana\dashboards
echo. > monitoring\grafana\dashboards\.gitkeep
mkdir monitoring\grafana\datasources
echo. > monitoring\grafana\datasources\prometheus.yaml

REM -------------------- docs ------------------------
mkdir docs
mkdir docs\adr
echo. > docs\adr\.gitkeep

REM -------------------- scripts ------------------------
mkdir scripts
echo. > scripts\__init__.py
echo. > scripts\generate_traffic.py
echo. > scripts\setup_feast.py
echo. > scripts\download_data.py
echo. > scripts\export_dashboards.py

REM -------------------- root files ------------------------
echo. > docker-compose.yml
echo. > docker-compose.demo.yml
echo. > .env.example
echo. > .gitignore
echo. > requirements.txt
echo. > pyproject.toml
echo. > Makefile
echo. > README.md

echo Project structure created successfully!
pause