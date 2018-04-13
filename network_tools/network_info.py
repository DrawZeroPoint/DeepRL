

# param: kernel size, stride, padding
def get_blob_shape(in_h, in_w, ksp_list):
    result = []
    for p in ksp_list:
        out_h = (in_h + 2 * p[2] - p[0]) / p[1] + 1
        out_w = (in_w + 2 * p[2] - p[0]) / p[1] + 1
        shape = [out_h, out_w]
        result.append(shape)
        in_h = out_h
        in_w = out_w
    return result


def get_liner_input_dim(in_h, in_w, batch_size, ksp_list):
    result = get_blob_shape(in_h, in_w, ksp_list)
    return batch_size * result[-1][0] * result[-1][1]
