import glob

def clean_labels(label_dir, max_cls=43):
    for f in glob.glob(label_dir + "/*.txt"):
        valid = []
        with open(f) as file:
            for line in file:
                cls = int(line.split()[0])
                if cls < max_cls:
                    valid.append(line)
        with open(f, "w") as file:
            file.writelines(valid)

clean_labels("train/labels", 43)
clean_labels("valid/labels", 43)

print("Label cleanup done ✔")
