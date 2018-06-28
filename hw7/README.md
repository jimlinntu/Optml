# Enviroment
CSIE Workstation

# Preparation
1. Put `kddb` and `kddb.t` into `data/` folder
2. `cd src/`
3. Build the mex file
```
matlab -nodisplay -nosplash -r "LIBLINEAR_PATH = '<LIBLINEAR_PATH>'; run('build.m'); exit"
```
For example:
```
matlab -nodisplay -nosplash -r "LIBLINEAR_PATH = '~/tmp2/convex_hw7/liblinear'; run('build.m'); exit"
```

# How to train model
- Gradient Descent Method

```
matlab -nodisplay -nosplash -r "METHOD = 'gradient'; COST = 0.1; LR = 0.01; EPS = 0.01; XI = 0.1 ;run('gradient_descent_main.m'); exit"
```

- Newton Method

```
matlab -nodisplay -nosplash -r "METHOD = 'newton'; COST = 0.1; LR = 0.01; EPS = 0.01; XI = 0.1 ;run('gradient_descent_main.m'); exit"
```
