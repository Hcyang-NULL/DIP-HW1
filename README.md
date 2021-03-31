## Homework1

#### Notice: 
* Please make sure your working directory is root directory
* My environment is MacOS, please modify the path symbols by yourself if you are using Windows
* All programs are set to show the result image in your screen instead of saving it (save code is commented)
* I have tested all the code, and it works in my conda environment on my Mac.

### Structure

```
.
├── Figures
│   ├── output
│   │   ├── city-final.png
│   │   ├── city-match.png
│   │   ├── fog_defog.png
│   │   ├── mc1-final.png
│   │   ├── mc1-match.png
│   │   ├── mc2-final.png
│   │   ├── mc2-match.png
│   │   ├── task1.png
│   │   ├── task2.png
│   │   ├── task3.png
│   │   ├── task4.png
│   │   ├── task5.png
│   │   ├── task6.png
│   │   ├── task7.png
│   │   ├── village-final.png
│   │   └── village-match.png
│   └── source
│       ├── Ex_ColorEnhance.png
│       ├── Mountains.png
│       ├── Starry_night.png
│       ├── Tam_clear.jpg
│       ├── city1.png
│       ├── city2.png
│       ├── cy_dst.png
│       ├── cy_src.png
│       ├── hats.bmp
│       ├── houses.bmp
│       ├── mc1-1.png
│       ├── mc1-2.png
│       ├── mc2-1.png
│       ├── mc2-2.png
│       ├── village1.jpg
│       └── village2.jpg
├── README.md
├── Source_code
│   ├── solution1.py
│   ├── solution2.py
│   ├── solution3.py
│   └── utils
│       └── SIFT.py
└── report.pdf
```

### 1. Problem-1: Defogging and Fogging

```shell
python Source_code/solution1.py
```

Line 128 is to save the plot image.

### 2. Problem-2: Stitching

```shell
python Source_code/solution2.py
```

We do not show the results because we use cv2.imwrite to save it to ./Figures/output. The names of figures are xxx-match.png and xxx-final.png

### 3. Problem-3: Color Transfer

```shell
python Source_code/solution3.py
```

Line 171 is to change the mode of plt.show and plt.savefig (default is plt.show)
