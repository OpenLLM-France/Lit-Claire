import csv
import json
import numpy as np
import matplotlib.pyplot as plt

def read_validation_csv(csvfile):
    print(f"Reading {csvfile}")
    if csvfile is None:
        return {}
    data = {}
    with open(csvfile) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['data'].split("/")[1]
            iter = int(row["iter"]) + 1
            if name not in data:
                data[name] = []
            data[name].append((iter, float(row["loss"]), row["file"]))

    return data

def read_training_csv(csvfile, folder="."):
    print(f"Reading {csvfile}")
    data = []
    valid_data = []
    valid_time_delta = 0
    batch_size = 1
    with open(csvfile) as f:
        iter = 0
        reader = csv.DictReader(f)
        for row in reader:
            s = row["step"]
            if s:
                iter0 = int(s) + 1
                if iter0 > iter:
                    iter = iter0
                    time = float(row["time/total"])
                    if iter == 1:
                        batch_size = int(row["samples"])
            l = row["loss"]
            if l:
                loss = float(l)
                data.append((iter, loss))
            l = row.get("val_loss")
            if l:
                loss = float(l)
                valid_data.append((iter, loss, os.path.join(folder, f"iter-{iter-1:06d}-ckpt.pth")))
                valid_time_delta += float(row["val_time"])

    if valid_time_delta:
        print(f"Time spent to validate: {valid_time_delta:2f} sec")

    return data, valid_data, batch_size, (time-valid_time_delta)/iter, valid_time_delta

def format_dataset_name(name):
    name = name.replace("Politics", "Débats politiques")
    name = name.replace("AssembleeNationale", "Assemblée Nationale")
    name = name.replace("Theatre", "Théâtre")
    name = name.replace("Meetings", "Réunions")
    if name == "Validation":
        return "Validation (online)"
    return name # "Validation: " + name

def name_order(name):
    if name == "Validation":
        return (1e10, name)
    if name == "Meetings":
        return (2, name)
    if name == "Politics":
        return (3, name)
    if name == "AssembleeNationale":
        return (4, name)
    if name == "Theatre":
        return (5, name)
    return (0, name)

if __name__ == "__main__":

    import os

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", help="folder where lies validation_results.csv and metrics.csv", default=".", nargs="+")
    parser.add_argument("--segment_length", help="Number of tokens in each sequence", type=int, default=2048)
    parser.add_argument("--max_iter", help="Maximum number of iterations", type=int, default=None)
    parser.add_argument("--max_loss", help="Maximum loss to plot", type=float, default=None)
    parser.add_argument("--min_loss", help="Minimum loss to plot", type=float, default=None)
    args = parser.parse_args()

    COLOR_TRAIN = "cornflowerblue"
    COLOR_VALID = "gray"
    COLOR_BEST = "b"
    COLORS_VALID_OFFLINE = [
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "olive",
        "cyan",
        "magenta",
        "yellow",
        "black",
    ]

    num_expes = len(args.folders)
    num_columns = num_expes # 1

    fig, axes = plt.subplots(nrows=5, ncols=num_columns, gridspec_kw={
        'height_ratios': [10, 0.2, 0.2, 0.2, 0.2],
        'width_ratios': [1/num_columns] * num_columns,
    }, facecolor=(1,1,1,0)) # transparent)
    if num_columns == 1:
        axes = [[ax] for ax in axes]
    # plt.suptitle("Claire-7b v0.02 (batch size= 12 sequences)")

    hparams = []
    batch_sizes = []
    devices = []
    for folder in args.folders:
        hparams_file = os.path.join(folder, "hparams.json")
        if os.path.isfile(hparams_file):
            hparams.append(json.load(open(hparams_file)))
            batch_sizes.append(hparams[-1]["micro_batch_size"])
            devices.append(hparams[-1]["devices"])
        else:
            hparams = [{}] * len(args.folders)
            batch_sizes.append(None)
            devices.append(1)
            break

    # Only retain different hyperparameters
    ignore_keys = [
        "out_dir",
        "save_interval", "eval_interval", "log_interval",
        "enable_validation",
        "gradient_accumulation_iters",
        "max_checkpoints", "num_epochs", "early_stopping",
    ]
    for i, hparam in enumerate(hparams):
        for key in list(hparam.keys()):
            all_same = all([(hparam[key] == hother[key]) if (key in hother and key not in ignore_keys) else True for hother in hparams])
            if all_same:
                del hparam[key]

    min_loss = args.min_loss if args.min_loss else 1e10
    max_loss = args.max_loss if args.max_loss else 0

    for iexpe, folder in enumerate(args.folders):
        icolumn = min(iexpe, num_columns-1)

        max_x = 0

        validation_file = None
        training_file = None
        for root, dirs, files in os.walk(folder):
            for filename in files:
                if filename.startswith("validation_results") and filename.endswith(".csv"):
                    if validation_file is None:
                        validation_file = os.path.join(root, filename)
                    else:
                        # Look at modification times
                        if os.path.getmtime(os.path.join(root, filename)) > os.path.getmtime(validation_file):
                            validation_file = os.path.join(root, filename)
                if filename == "metrics.csv":
                    if training_file is None:
                        training_file = os.path.join(root, filename)
                    else:
                        # Look at modification times
                        if os.path.getmtime(os.path.join(root, filename)) > os.path.getmtime(training_file):
                            training_file = os.path.join(root, filename)
        
        # assert validation_file is not None, f"Could not find validation_results.csv in {args.folder}"
        assert training_file is not None, f"Could not find metrics.csv in {args.folder}"

        conv_training, valid_data, batch_size, factor_time, valid_time_delta = read_training_csv(training_file, folder=folder)
        if batch_sizes[iexpe] is not None:
            assert batch_size == batch_sizes[iexpe], f"Batch size mismatch: {batch_size} != {batch_sizes[iexpe]}"
        conv_validation = {}
        if valid_data:
            conv_validation["Validation"] = valid_data
        else:
            conv_validation["Validation"] = []
        conv_validation.update(read_validation_csv(validation_file))
        
        ax = axes[0][icolumn]
        if hparams[iexpe]:
            title = ",\n".join([f"{k}={v}" for k, v in hparams[iexpe].items()])
            ax.set_title(title, fontsize=9)
        elif num_columns > 1:
            ax.set_title(os.path.basename(folder))

        x, y = zip(*sorted(conv_training))
        ax.plot(x, y, label="Training (online)", alpha=0.5, color=COLOR_TRAIN)
        max_x = max(max_x, max(x))
        if args.max_iter and (max_x > args.max_iter or num_columns > 1):
            max_x = args.max_iter

        best_x = None
        if conv_validation:
            x_valids = None
            for name in sorted(conv_validation.keys(), key = lambda name: -len(conv_validation[name])):
                x, y, files = zip(*sorted(conv_validation[name]))
                x_valids = x
                ckpt_files = files
                break
            valids = [[] for _ in x_valids]

            sorted_names = sorted(conv_validation.keys(), key = lambda name: name_order(name))

            for ivalid, name in enumerate(sorted_names):
                empty = not conv_validation[name]
                if not empty:
                    x, y, _ = zip(*sorted(conv_validation[name]))
                    ax.plot(x, y, label=format_dataset_name(name),
                        marker="+" if name != "Validation" else None,
                        linewidth = 2 if name == "Validation" else 1,
                        color = COLORS_VALID_OFFLINE[ivalid % len(COLORS_VALID_OFFLINE)] if name != "Validation" else COLOR_VALID,
                    )
                else:
                    ax.plot([], [], label=None)
                # Exclude online validation if there is offline validation to compute best results
                if len(conv_validation) == 1 or name != "Validation":
                    for i, yi in enumerate(y):
                        i = x_valids.index(x[i])
                        valids[i].append(yi)

            mean_valids = [np.median(v) if v else 1e10 for v in valids]
            best_valid = mean_valids.index(min(mean_valids))
            print("Best loss:", os.path.join(folder, files[best_valid]))
            best_x = x_valids[best_valid]
            ax.axvline(x=best_x, color=COLOR_BEST, linestyle=':') #, label=f"Best ({files[best_valid]})")
            ax.text(best_x, 0.5, f"{os.path.basename(files[best_valid])}", color=COLOR_BEST, fontsize=9, rotation=90, ha="right", va="center", transform=ax.get_xaxis_transform())
            for ivalid, name in enumerate(sorted_names):
                if name == "Validation" and len(conv_validation) > 1:
                    continue
                values = conv_validation[name]
                x, y, files = zip(*sorted(values))
                if best_x not in x:
                    continue
                i = x.index(best_x)
                ax.axhline(y=y[i], color=COLORS_VALID_OFFLINE[ivalid % len(COLORS_VALID_OFFLINE)], linestyle=':')
                ax.text(-0.05, y[i], f"{y[i]:.2f}", color=COLORS_VALID_OFFLINE[ivalid % len(COLORS_VALID_OFFLINE)], fontsize=9, ha="right", va="center", transform=ax.get_yaxis_transform())
            
        ymin, ymax = ax.get_ylim()
        if not args.max_loss:
            max_loss = max(max_loss, ymax)
        if not args.min_loss:
            min_loss = min(min_loss, ymin)
        if icolumn == 0:
            ax.set_ylabel("Loss")
        ax.legend()

        for iax, (label, factor, step, unit) in enumerate([
            ("batches",         1,                                1000,     "k"),
            ("sequences",       batch_size,                       5000,     "k"),
            ("tokens",          batch_size * args.segment_length, 10000000, "M"),
            ("training",        factor_time,                      3600,     "hrs"),
            ("GPU",             factor_time*devices[iexpe],       3600*devices[iexpe], "hrs"),
        ]):
            factor2 = {"k": 1000, "M": 1000000, "hrs": 3600}.get(unit, 1)
            _zero = "0" if iax == 0 else ""
            ax = axes[iax][icolumn]
            ax.set_xlim(0, max_x)
            xticks = np.arange(0, max_x+1, step / factor)
            _unit = "" # unit if unit != "hrs" else ""
            xticks_string = [f"{int(round(x*factor/factor2))}{_unit}" if x > 0 else _zero for x in xticks]
            # xticks_string[-1] += " " + label
            ax.set_xticks(xticks, xticks_string)
            xlabel = (unit + " " + label).strip()
            if icolumn == 0:
                ax.set_xlabel(xlabel, fontsize=9, ha="right")
                ax.xaxis.set_label_coords(-0.05, 0)
            elif iexpe == num_columns-1:
                ax.set_xlabel(xlabel, fontsize=9, ha="left")
                ax.xaxis.set_label_coords( 1.05, 0)
            if iax > 0:
                # Remove upper, left and right axis
                for spine in 'top', 'right', 'left':
                    ax.spines[spine].set_visible(False)
                ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

                # Plot best
                if best_x is not None:
                    ax.axvline(x=best_x, color=COLOR_BEST)



    for icolumn in range(num_columns):
        ax = axes[0][icolumn]
        ax.set_ylim(min_loss, max_loss)

    plt.show()
