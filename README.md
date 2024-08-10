# MAL-BUD

### STEP 1. make .env file
```
GROQ_API_KEY=<YOUR API KEY>
SK_OPEN_API_KEY=<YOUR API KEY>
```
You can manage your API keys here: https://console.groq.com/playground , https://openapi.sk.com/mypage/dashboard

### STEP 2-0. install tkinter (if needed)
```shell
python3 -m tkinter
```
if nothing pops up with above command, then install tkinter
```shell
brew install tcl-tk
echo 'export PATH="/usr/local/opt/tcl-tk/bin:$PATH"' >> ~/.zshrc
echo 'export LDFLAGS="-L/usr/local/opt/tcl-tk/lib"' >> ~/.zshrc
echo 'export CPPFLAGS="-I/usr/local/opt/tcl-tk/include"' >> ~/.zshrc
source ~/.zshrc
```

### STEP 2-1. install dependencies
```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mac.txt
```
mac
```
python3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-windows.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
windows

### STEP 2-2. download pretrained model (if needed)
```shell
chmod +x download_models.sh
./download_models.sh
```
download model from hugging-face


### STEP 3. run LLM - TTS
```shell
python llm_tts.py
```

### STEP 4. install pre-commit before commit
```shell
pre-commit install
pre-commit run --all-files # if you want to check all files
```
