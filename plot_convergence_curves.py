import csv
import numpy as np
import matplotlib.pyplot as plt

def read_validation_csv(csvfile):
    data = {}
    with open(csvfile) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['data'].split("/")[1]
            iter = int(row["iter"])
            if name not in data:
                data[name] = []
            data[name].append((iter, float(row["loss"]), row["file"]))

    return data

def read_training_csv(csvfile):
    data = []
    with open(csvfile) as f:
        iter = -1
        reader = csv.DictReader(f)
        for row in reader:
            s = row["step"]
            if s:
                iter0 = int(s)
                if iter0 > iter:
                    iter = iter0
                    time = float(row["time/total"])
                    if iter == 0:
                        batch_size = int(row["samples"])
            l = row["loss"]
            if l:
                loss = float(l)
                data.append((iter, loss))

    return data, batch_size, time/iter

if __name__ == "__main__":

    import os

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="folder where lies validation_results.csv and metrics.csv", default=".", nargs="?")
    parser.add_argument("--segment_length", help="Number of tokens in each sequence", type=int, default=2048)
    args = parser.parse_args()

    validation_file = None
    training_file = None
    for root, dirs, files in os.walk(args.folder):
        if "validation_results.csv" in files:
            validation_file = os.path.join(root, "validation_results.csv")
        if "metrics.csv" in files:
            if training_file is None:
                training_file = os.path.join(root, "metrics.csv")
            else:
                # Look at modification times
                if os.path.getmtime(os.path.join(root, "metrics.csv")) > os.path.getmtime(training_file):
                    training_file = os.path.join(root, "metrics.csv")
    
    assert validation_file is not None, f"Could not find validation_results.csv in {args.folder}"
    assert training_file is not None, f"Could not find metrics.csv in {args.folder}"

    conv_validation = read_validation_csv(validation_file)
    conv_training, batch_size, factor_time = read_training_csv(training_file)

    fig, ax = plt.subplots(4, 1, gridspec_kw={'height_ratios': [10, 0.2, 0.2, 0.2]})
    # plt.suptitle("Claire-7b v0.02 (batch size= 12 sequences)")
    
    plt.subplot(4, 1, 1)
    x, y = zip(*sorted(conv_training))
    plt.plot(x, y, label="(Training)", alpha=0.5)
    max_x = max(x)
    valids = None
    x_valids = None
    for name in sorted(conv_validation.keys(), key = lambda name: (-len(conv_validation[name]), name)):
        x, y, files = zip(*sorted(conv_validation[name]))
        plt.plot(x, y, label=name)
        if valids is None:
            valids = [[yi] for yi in y]
            x_valids = x
        else:
            for i, yi in enumerate(y):
                i = x_valids.index(x[i])
                valids[i].append(yi)

    mean_valids = [np.median(v) for v in valids]
    best_valid = mean_valids.index(min(mean_valids))
    print("Best loss:", files[best_valid])
    plt.axvline(x=x[best_valid], color='r', linestyle=':', label=f"Best ({files[best_valid]})")
    
    plt.ylabel("Loss")
    plt.legend()

    # a, b = plt.xlim()
    plt.xlim(0, max_x)
    xticks, _ = plt.xticks()
    xticks_batches = np.arange(0, max_x, 1000)
    xticks_batches_string = [f"{int(x/1000)}k" for x in xticks_batches]
    xticks_batches_string[-1] += " batches"
    plt.xticks(xticks_batches, xticks_batches_string)
    # plt.xlabel("Batches", loc='right')

    plt.subplot(4, 1, 2)
    plt.xlim(0, max_x)
    xticks_samples = np.arange(0, max_x, 5000/(batch_size))
    xticks_samples_string = [f"{int(x*batch_size/1000)}k" for x in xticks_samples]
    xticks_samples_string[-1] += " sequences"
    plt.xticks(xticks_samples, xticks_samples_string)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    # plt.xlabel("Samples", loc='right')

    plt.subplot(4, 1, 3)
    plt.xlim(0, max_x)
    xticks_tokens = np.arange(0, max_x, 10000000/(batch_size * args.segment_length))
    xticks_tokens_string = [f"{int(x*batch_size*args.segment_length/1000000)}M" for x in xticks_tokens]
    xticks_tokens_string[-1] += " tokens"
    plt.xticks(xticks_tokens, xticks_tokens_string)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    # plt.xlabel("Tokens", loc='right')

    plt.subplot(4, 1, 4)
    plt.xlim(0, max_x)
    xticks_times = np.arange(0, max_x, 3600 / factor_time)
    xticks_times_string = [f"{int(x*factor_time/3600)}h" for x in xticks_times]
    xticks_times_string[-1] += " hours"
    plt.xticks(xticks_times, xticks_times_string)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    # plt.xlabel("Time", loc='right')

    plt.show()
