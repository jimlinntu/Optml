% Build libsvmread.c into matlab function
libsvmread_path = fullfile(LIBLINEAR_PATH, 'matlab', 'libsvmread.c');
mex(libsvmread_path);