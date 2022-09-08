def compute_new_kernel_size(kernel_size, kernel_width):
    new_kernel_size = max(int(kernel_size * kernel_width), 1)
    # If kernel is even shape, round up to make it odd
    if new_kernel_size % 2 == 0:
        new_kernel_size += 1
    return new_kernel_size


def get_same_padding(kernel_size, stride, dilation) -> int:
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (dilation * (kernel_size - 1)) // 2


def get_asymtric_padding(kernel_size, stride, dilation, future_context):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")

    left_context = kernel_size - 1 - future_context
    right_context = future_context

    symmetric_padding = get_same_padding(kernel_size, stride, dilation)

    if kernel_size <= future_context:
        # kernel size is smaller than future context, equivalent to using entire context of kernel
        # simply return symmetric padding for this scenario
        print(
            f"Future context window is larger than the kernel size!\n"
            f"Left context = {left_context} | Right context = greater than {right_context} | "
            f"Kernel size = {kernel_size}\n"
            f"Switching to symmetric padding (left context = right context = {symmetric_padding})"
        )
        return symmetric_padding

    if left_context < symmetric_padding:
        print(
            f"Future context window is larger than half the kernel size!\n"
            f"Conv layer therefore uses more future information than past to compute its output!\n"
            f"Left context = {left_context} | Right context = {right_context} | "
            f"Kernel size = {kernel_size}"
        )

    if dilation > 1:
        left_context = dilation * kernel_size - 1 - dilation * future_context
        right_context = dilation * future_context
        return left_context, right_context

    return left_context, right_context