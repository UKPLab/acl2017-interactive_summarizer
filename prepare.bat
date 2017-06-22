pip install virtualenv
virtualenv --system-site-packages --unzip-setuptools  .venv_windows
set "VIRTUAL_ENV=C:\Users\hatieke\Projects\ukp-thesis\casum_summarizer\.venv_windows"
set "PATH=%VIRTUAL_ENV%\Scripts;%PATH%"
pip install -r requirements.txt
