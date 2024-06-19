## Installation

### Step 1: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 2: Activate the Virtual Environment

```bash
venv\Scripts\activate
```

### Step 3:  Install Required Packages

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Collect Training Data

```bash
python collect_training_data.py
```

### Step 2: Run Classifier

```bash
python classifier.py
```

### Step 3: Recognize New Data
- Update the ids variable in the recognize_new.py file with an appropraite name assigned to each id according to the face samples collection. Here is an example.
```bash
ids = { 1: "Srawan", 2: "Shyam" }
```
- Then run the recognizer_new.py with a below command.

```bash
python recognize_new.py
```