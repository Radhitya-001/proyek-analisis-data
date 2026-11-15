# Dicoding Collection Dashboard

Proyek Analisis Data: **E-Commerce Public Dataset**

## Setup Environment - Anaconda

```
conda create --name main-ds python=3.11
conda activate main-ds
pip install -r requirements.txt
```

## Setup Environment - Shell/Terminal

```
mkdir proyek_analisis_data
cd proyek_analisis_data
pipenv install
pipenv shell
pip install -r requirements.txt
```

## Run steamlit app

```
streamlit run dashboard.py
```

## Menjalankan notebook

Buka `notebook.ipynb` dari folder `submission`, lalu jalankan seluruh shell.
Notebook akan menulis ulang `dashboard/main_data.csv` dan `dashboard/state_geo.csv`.
