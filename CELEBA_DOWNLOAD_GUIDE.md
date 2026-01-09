# CelebA Download Guide for Google Colab

Google Drive automatic downloads are rate-limited. Here are the best alternatives:

## ✅ RECOMMENDED: Kaggle (Fastest & Most Reliable)

### Method 1: Direct Download (Easiest)
1. Go to https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
2. Click **"Download"** (requires free Kaggle account)
3. Upload the downloaded `archive.zip` or `celeba-dataset.zip` to Colab Files panel
4. Run this in a Colab cell:

```python
!unzip -q /content/archive.zip -d /content/celeba
# OR
!unzip -q /content/celeba-dataset.zip -d /content/celeba
```

### Method 2: Kaggle API (More Automated)
1. Get your Kaggle API key:
   - Go to https://www.kaggle.com/settings
   - Click **"Create New API Token"**
   - Download `kaggle.json`

2. Upload `kaggle.json` to Colab

3. Run in Colab:
```python
!pip install kaggle
!mkdir -p ~/.kaggle
!cp /content/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d jessicali9530/celeba-dataset
!unzip -q celeba-dataset.zip -d /content/celeba
```

---

## Alternative: Google Drive (Manual)

### Direct Download Link:
- **Link**: https://drive.google.com/file/d/1xJs_8JB0HYXiaAmU8PTG9qbk0WJ2Wo1U/view?usp=sharing
- Download manually from browser if automatic download fails
- Upload to Colab and extract

### If you have CelebA in your Drive already:
```python
from google.colab import drive
drive.mount('/content/drive')
!unzip /content/drive/MyDrive/img_align_celeba.zip -d /content/celeba
```

### To download to your Drive first:
1. Go to: https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8
2. Find `img_align_celeba.zip`
3. Right-click → "Make a copy" to your Drive
4. Use the mount method above

---

## Alternative: Official Source (Academic)

If you have academic access:
1. Go to http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
2. Download from Baidu Drive or Google Drive (original source)
3. Upload to Colab

---

## Verify Dataset

After downloading, verify the structure:

```python
!ls -la /content/celeba/img_align_celeba/ | head
```

Expected output:
```
000001.jpg
000002.jpg
...
202599.jpg
```

Should have **202,599 images**.

---

## After Downloading

Once you have the dataset, just run the main script:

```python
# Paste COLAB_PASTE_HERE.py content here
# It will detect the dataset automatically
```

---

## Quick Test (5 Epochs)

To test the pipeline before running the full 20 epochs:

Find this line in the script:
```python
N_EPOCHS = 20  # Change to 5 for quick test
```

Change to:
```python
N_EPOCHS = 5  # Quick test (~15-20 min)
```

---

## Storage Note

CelebA requires:
- **Zipped**: ~1.4 GB
- **Extracted**: ~1.5 GB
- **Total**: ~3 GB

Colab free tier has ~75 GB disk space, so this should fit easily.

---

## Troubleshooting

### "No images found"
Check directory structure:
```python
!find /content/celeba -name "*.jpg" | head
```

Should show paths like:
```
/content/celeba/img_align_celeba/000001.jpg
```

### Wrong structure?
If images are in `/content/celeba/celeba/img_align_celeba/`:
```python
!mv /content/celeba/celeba/img_align_celeba /content/celeba/
!rm -rf /content/celeba/celeba
```

### Out of space?
Delete the zip after extraction:
```python
!rm /content/celeba.zip
!rm /content/archive.zip
```
