import csv
import json
import numpy as np
import matplotlib.pyplot as plt

def read_validation_csv(csvfile):
    if csvfile is None:
        return {}
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
    parser.add_argument("folders", help="folder where lies validation_results.csv and metrics.csv", default=".", nargs="+")
    parser.add_argument("--segment_length", help="Number of tokens in each sequence", type=int, default=2048)
    parser.add_argument("--max_iter", help="Maximum number of iterations", type=int, default=None)
    args = parser.parse_args()

    num_expes = len(args.folders)
    num_columns = num_expes # 1

    # fig = plt.figure()
    # spec = plt.gridspec.GridSpec(ncols=num_columns, nrows=4,
    #     height_ratios=[10, 0.2, 0.2, 0.2],
    #     width_ratios=[1/num_columns] * num_columns,
    #     #wspace=0.5, hspace=0.5
    # )
        
    fig, axes = plt.subplots(nrows=4, ncols=num_columns, gridspec_kw={
        'height_ratios': [10, 0.2, 0.2, 0.2], # list(zip(*([[10, 0.2, 0.2, 0.2]] * num_columns)))}
        'width_ratios': [1/num_columns] * num_columns,
    })
    if num_columns == 1:
        axes = [[ax] for ax in axes]
    # plt.suptitle("Claire-7b v0.02 (batch size= 12 sequences)")

    hparams = []
    for folder in args.folders:
        hparams_file = os.path.join(folder, "hparams.json")
        if os.path.isfile(hparams_file):
            hparams.append(json.load(open(hparams_file)))
        else:
            hparams = [{}] * len(args.folders)
            break

    # Only retain different hyperparameters
    ignore_keys = ["out_dir", "save_interval", "eval_interval", "max_checkpoints", "gradient_accumulation_iters"]
    for i, hparam in enumerate(hparams):
        for key in list(hparam.keys()):
            all_same = all([(hparam[key] == hother[key]) if (key in hother and key not in ignore_keys) else True for hother in hparams])
            if all_same:
                del hparam[key]

    min_loss = 0
    max_loss = 0

    for iexpe, folder in enumerate(args.folders):
        icolumn = min(iexpe, num_columns-1)

        max_x = 0

        validation_file = None
        training_file = None
        for root, dirs, files in os.walk(folder):
            if "validation_results.csv" in files:
                validation_file = os.path.join(root, "validation_results.csv")
            if "metrics.csv" in files:
                if training_file is None:
                    training_file = os.path.join(root, "metrics.csv")
                else:
                    # Look at modification times
                    if os.path.getmtime(os.path.join(root, "metrics.csv")) > os.path.getmtime(training_file):
                        training_file = os.path.join(root, "metrics.csv")
        
        # assert validation_file is not None, f"Could not find validation_results.csv in {args.folder}"
        assert training_file is not None, f"Could not find metrics.csv in {args.folder}"

        conv_validation = read_validation_csv(validation_file)
        conv_training, batch_size, factor_time = read_training_csv(training_file)
        
        ax = axes[0][icolumn]
        if hparams[iexpe]:
            title = ", ".join([f"{k}={v}" for k, v in hparams[iexpe].items()])
            ax.set_title(title)

        # plt.subplot(4, icolumn+1, 1)
        x, y = zip(*sorted(conv_training))
        ax.plot(x, y, label="(Training)", alpha=0.5)
        max_x = max(max_x, max(x))
        if args.max_iter and (max_x > args.max_iter or num_columns > 1):
            max_x = args.max_iter

        if conv_validation:
            valids = None
            x_valids = None
            for name in sorted(conv_validation.keys(), key = lambda name: (-len(conv_validation[name]), name)):
                x, y, files = zip(*sorted(conv_validation[name]))
                ax.plot(x, y, label=name)
                if valids is None:
                    valids = [[yi] for yi in y]
                    x_valids = x
                else:
                    for i, yi in enumerate(y):
                        i = x_valids.index(x[i])
                        valids[i].append(yi)

            mean_valids = [np.median(v) for v in valids]
            best_valid = mean_valids.index(min(mean_valids))
            print("Best loss:", os.path.join(folder, files[best_valid]))
            ax.axvline(x=x[best_valid], color='r', linestyle=':', label=f"Best ({files[best_valid]})")
            
        ymin, ymax = ax.get_ylim()
        max_loss = max(max_loss, ymax)
        ax.set_ylabel("Loss")
        ax.legend()

        for iax, (label, factor, step, unit) in enumerate([
            ("batches",     1,                                1000,     ""),
            ("sequences",   batch_size,                       5000,     "k"),
            ("tokens",      batch_size * args.segment_length, 10000000, "M"),
            ("time",        factor_time,                      3600,     "h"),
        ]):
            factor2 = {"k": 1000, "M": 1000000, "h": 3600}.get(unit, 1)
            _zero = "0" if iax == 0 else ""
            ax = axes[iax][icolumn]
            ax.set_xlim(0, max_x)
            xticks = np.arange(0, max_x+1, step / factor)
            xticks_string = [f"{int(round(x*factor/factor2))}{unit}" if x > 0 else _zero for x in xticks]
            xticks_string[-1] += " " + label
            ax.set_xticks(xticks, xticks_string)

    for icolumn in range(num_columns):
        ax = axes[0][icolumn]
        ax.set_ylim(0, max_loss)

    plt.show()
