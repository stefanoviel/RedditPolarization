import os


def get_slurm_job_report_as_str(file_path: str):
    with open(file_path, 'r') as f:
        return f.read()


def analyze_slurm_jobs_reports(path: str = '/Users/andrea/Desktop/Temp/slurm_jobs/'):
    files = os.listdir(path)
    files = [f for f in files if f.endswith('.out')]
    bad_files = []
    for f in files:
        s = get_slurm_job_report_as_str(file_path=f'{path}/{f}')
        if 'Finished converting' not in s:
            file = s.split('\n')[1].split(' ')[8].split('/')[-1]
            print(file)
            print(s)
            bad_files.append(file)

    print(sorted(bad_files))
    print(sorted([b.split('.')[0] for b in bad_files]))


if __name__ == '__main__':
    analyze_slurm_jobs_reports()