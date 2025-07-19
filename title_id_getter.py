import os
import glob
from tqdm import tqdm


def getter(file_path):
    title, idd, typ = "", "", ""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("&title"):
                title = line.split("=")[1]
                if title.find("[SD]") != -1:
                    title = title.replace("[SD]", "")
                    typ = "std"
                elif title.find("[DX]") != -1:
                    title = title.replace("[DX]", "")
                    typ = "dx"
                elif title.find("[宴]") != -1:
                    title = title.replace("[宴]", "")
                    typ = "utage"
            if line.startswith("&shortid"):
                idd = line.split("=")[1]
    return title.strip(), idd.strip(), typ.strip()


def work():
    data = []
    # 查找所有maidata.txt路径
    maidat_paths = glob.glob("./Maichart-Converts/**/*/maidata.txt", recursive=True)
    target_dirs = {os.path.abspath(os.path.dirname(p)) for p in maidat_paths}

    with tqdm(target_dirs, unit="dir", desc="处理进度") as pbar:
        for input_dir in pbar:
            pbar.set_postfix(file=os.path.basename(input_dir))
            file_path = os.path.join(input_dir, "maidata.txt")
            title, idd, typ = getter(file_path)
            data.append((idd, title, typ))
            # print(input_dir)

    with open("title_id.csv", "w", encoding="utf-8") as f:
        for idd, title, typ in data:
            f.write(f"{idd},{title},{typ}\n")


if __name__ == "__main__":
    work()
