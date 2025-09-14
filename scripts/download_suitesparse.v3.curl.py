import os
import sys
import argparse
# import ssgetpy
import requests
import time
import subprocess

# https://sparse.tamu.edu/files/ssstats.csv


def download_list():
    url = "https://sparse.tamu.edu/files/ssstats.csv"
    file_name = "ssstats.csv"
    response = requests.get(url)
    with open(file_name, "wb") as fout:
        fout.write(response.content)
        print(F"Downloaded {file_name}")

    return file_name


def get_total_num_matrices(file_name: str):
    with open(file_name) as fin:
        line = fin.readline()
        # number of matrices is the first line
        number = int(line)
        line = fin.readline()
        print(F"Totally {number} matrices. Updated on {line}.")

        names = []
        for mtx_ind in range(number):
            line = fin.readline()
            columns = line.split(',')
            type = columns[0]
            name = columns[1]
            names.append((type, name))
        return number, names

    return (None, None)


def download_all(output_dir: str,
                #  num_matrices: int,
                 mtx_names: list):
    """Download all matrices from SuiteSparse

    Args:
        output_dir (str): save target directory
        num_matrices (int): number of matrices up to now
    """

    # # Matrix ID is starting from 1, not 0
    # for mtx_ind in range(1, num_matrices + 1):
    #     result = ssgetpy.search(mtx_ind)
    #     # Download the matrix into MM format under output_dir and extract it.
    #     # Reference: https://github.com/drdarshan/ssgetpy/blob/master/demo.ipynb
    #     print(F"\nDownloading matrix [{mtx_ind}] {result} under {output_dir} ...")
    #     result.download(format="MM", destpath=output_dir, extract=True)

    num_matrices = len(mtx_names)
    # Change working directory
    os.chdir(output_dir)

    for ind, (type, name) in enumerate(mtx_names):
        mtx_file = os.path.join(name, F"{name}.mtx")
        if os.path.isfile(mtx_file):
            print(F"\n[{ind}/{num_matrices}] {mtx_file} exists. Skipped.", flush=True)
            continue
        print(F"\n[{ind}/{num_matrices}] Downloading {name} ...", flush=True)

        url = F"https://suitesparse-collection-website.herokuapp.com/MM/{type}/{name}.tar.gz"
        cmd = F"curl -O -L {url}"
        subprocess.run(cmd, shell=True)

        cmd = F"tar xvf {name}.tar.gz"
        subprocess.run(cmd, shell=True)

        cmd = F"rm -rf {name}.tar.gz"
        subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download all SuiteSparse matrices")
    parser.add_argument("--dir", "-d", type=str, help="download target directory")
    if len(sys.argv) < 2:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()
    output_dir = args.dir

    file_name = download_list()
    num_matrices, mtx_names = get_total_num_matrices(file_name)
    if num_matrices == None or mtx_names == None:
        print(F"Error: failed to get matrix names.", file=sys.stderr)

    start_time = time.perf_counter()
    # download_all(output_dir, num_matrices, mtx_names)
    download_all(output_dir, mtx_names)
    end_time = time.perf_counter()

    print(F"Downloaded {num_matrices} matrices in total.")
    print(F"Execution_time(s): {end_time - start_time}")
