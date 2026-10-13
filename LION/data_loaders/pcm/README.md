# PCM Data Loading

The ground truth data is current in a structure like this:

```bash
pcm_data_dir
├── CIGS
│   ├── image0001.png
│   ├── image0002.png
│   ├── image0003.png
│   ├── image0004.png
|   ...
│   └── Thumbs.db
├── Organic
├── Perovskite
├── Reference
├── Silicone
```

The samples are saved as PNG, categorized into a few categories, with `Thumbs.db` files which I will probably ignore.
