function modified_label_vector = modify_label(label_vector)
    modified_label_vector = label_vector;
    modified_label_vector(modified_label_vector == 0) = -1;
