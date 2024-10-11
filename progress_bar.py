def print_progress_bar(epoch, total_epochs, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        epoch        - Required : current epoch (Int)
        total_epochs - Required : total epochs (Int)
        prefix       - Optional : prefix string (Str)
        suffix       - Optional : suffix string (Str)
        decimals     - Optional : positive number of decimals in percent complete (Int)
        length       - Optional : character length of bar (Int)
        fill         - Optional : bar fill character (Str)
        print_end    - Optional : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * ((epoch + 1) / float(total_epochs)))
    filled_length = int(length * (epoch + 1) // total_epochs)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if epoch == total_epochs - 1: 
        print()