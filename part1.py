import torch


def my_1dfilter(signal: torch.FloatTensor,
                kernel: torch.FloatTensor) -> torch.FloatTensor:
    """Filters the signal by the kernel.

    output = signal * kernel where * denotes the cross-correlation function.
    Cross correlation is similar to the convolution operation with difference
    being that in cross-correlation we do not flip the sign of the kernel.

    Reference: 
    - https://mathworld.wolfram.com/Cross-Correlation.html
    - https://mathworld.wolfram.com/Convolution.html

    Note:
    1. The shape of the output should be the same as signal.
    2. You may use zero padding as required. Please do not use any other 
       padding scheme for this function.
    3. Take special care that your function performs the cross-correlation 
       operation as defined even on inputs which are asymmetric.

    Args:
        signal (torch.FloatTensor): input signal. Shape=(N,)
        kernel (torch.FloatTensor): kernel to filter with. Shape=(K,)

    Returns:
        torch.FloatTensor: filtered signal. Shape=(N,)
    """
    filtered_signal = torch.FloatTensor()

    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    
    # cross-correlation 

    # Goal is to pad the signal on both sides 
    size_k = list(kernel.shape)[0]
    size_s = list(signal.shape)[0]
    padding = size_k // 2

    signal_padding = torch.nn.ConstantPad1d((padding, padding), 0)
    signal_padded = signal_padding(signal)

    list_of_tensors = []
    iter_max = size_s
    for i in range(0, iter_max):
        list_of_tensors.append(float(torch.matmul(signal_padded[i:i+size_k], kernel)))
    
    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return torch.FloatTensor(list_of_tensors)
