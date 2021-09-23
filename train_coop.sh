#! /bin/bash

#PBS -A "MModul-GMAS"
#PBS -l select=1:ncpus=1:mem=64gb:ngpus=1
#PBS -l pmem=64gb
#PBS -l walltime=47:59:00
#PBS -q CUDA
#PBS -o Coop_Keyword.out

set -e
export LOGFILE=$PBS_O_WORKDIR/$PBS_JOBNAME"."$PBS_JOBID".log"

SCRATCHDIR=/scratch_gs/$USER/$PBS_JOBID
mkdir -p "$SCRATCHDIR"
mkdir -p "$SCRATCHDIR/bert-base-cased"
mkdir -p "$SCRATCHDIR/gpt2"

cp -r /gpfs/project/$USER/cooplio/* $SCRATCHDIR/.
cp -r /gpfs/project/$USER/models/gpt2/* $SCRATCHDIR/gpt2/.
cp -r /gpfs/project/$USER/models/bert-base-cased/* $SCRATCHDIR/bert-base-cased/.

cd $PBS_O_WORKDIR

echo "$PBS_JOBID ($PBS_JOBNAME) @ `hostname` at `date` in "$RUNDIR" START" > $LOGFILE
echo "`date +"%d.%m.%Y-%T"`" >> $LOGFILE

module load Miniconda/3
module load Python/3.6.5
unset PYTHONPATH
unset PYTHONHOME

cp -r $PBS_O_WORKDIR/* $SCRATCHDIR/.
cd $SCRATCHDIR
rm $PBS_JOBNAME"."$PBS_JOBID".log"

echo >> $LOGFILE
qstat -f $PBS_JOBID >> $LOGFILE

conda create -c http://conda.repo.test.hhu.de/bioconda -c http://conda.repo.test.hhu.de/main -c http://conda.repo.test.hhu.de/conda-forge --override-channels --name my_environment python=3.6

conda activate my_environment


conda install -c http://conda.repo.test.hhu.de/bioconda -c http://conda.repo.test.hhu.de/main -c http://conda.repo.test.hhu.de/conda-forge --override-channels jsonnet


python3 -m pip install --user --upgrade -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de pip

echo "load packages" >>  $LOGFILE
pip install --user --pre -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de /software/pytorch/torch-1.8.0.dev20201102+cu110-cp36-cp36m-linux_x86_64.whl /software/pytorch/torchvision-0.9.0.dev20201102+cu110-cp36-cp36m-linux_x86_64.whl
pip install --user -q -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de transformers==3.4.0
pip install --user -q -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de sentencepiece
pip install --user -q -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de click
pip install --user -q -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de py-rouge
pip install --user -q -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de tqdm
# pip install --user -q -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de jsonnet
pip install --user -q -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de pandas
pip install --user -q -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de allennlp==1.2.0
pip install --user -q -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de tensorboard
pip install --user -q -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de requests
pip install --user -q -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de nltk
pip install --user -q -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de huggingface_hub



# conda install -c http://conda.repo.test.hhu.de/bioconda -c http://conda.repo.test.hhu.de/main -c http://conda.repo.test.hhu.de/conda-forge --override-channels torch==1.7.1
# conda install -c http://conda.repo.test.hhu.de/bioconda -c http://conda.repo.test.hhu.de/main -c http://conda.repo.test.hhu.de/conda-forge --override-channels transformers==3.4.0
# conda install -c http://conda.repo.test.hhu.de/bioconda -c http://conda.repo.test.hhu.de/main -c http://conda.repo.test.hhu.de/conda-forge --override-channels entencepiece
# conda install -c http://conda.repo.test.hhu.de/bioconda -c http://conda.repo.test.hhu.de/main -c http://conda.repo.test.hhu.de/conda-forge --override-channels click
# conda install -c http://conda.repo.test.hhu.de/bioconda -c http://conda.repo.test.hhu.de/main -c http://conda.repo.test.hhu.de/conda-forge --override-channels py-rouge
# conda install -c http://conda.repo.test.hhu.de/bioconda -c http://conda.repo.test.hhu.de/main -c http://conda.repo.test.hhu.de/conda-forge --override-channels tqdm
# conda install -c http://conda.repo.test.hhu.de/bioconda -c http://conda.repo.test.hhu.de/main -c http://conda.repo.test.hhu.de/conda-forge --override-channels pandas
# conda install -c http://conda.repo.test.hhu.de/bioconda -c http://conda.repo.test.hhu.de/main -c http://conda.repo.test.hhu.de/conda-forge --override-channels allennlp==1.2.0
# conda install -c http://conda.repo.test.hhu.de/bioconda -c http://conda.repo.test.hhu.de/main -c http://conda.repo.test.hhu.de/conda-forge --override-channels tensorboard
# conda install -c http://conda.repo.test.hhu.de/bioconda -c http://conda.repo.test.hhu.de/main -c http://conda.repo.test.hhu.de/conda-forge --override-channels requests
# conda install -c http://conda.repo.test.hhu.de/bioconda -c http://conda.repo.test.hhu.de/main -c http://conda.repo.test.hhu.de/conda-forge --override-channels nltk
# conda install -c http://conda.repo.test.hhu.de/bioconda -c http://conda.repo.test.hhu.de/main -c http://conda.repo.test.hhu.de/conda-forge --override-channels huggingface_hub

echo "installed optimus coop vae dependencies" >> $LOGFILE

echo "starting training" >>  $LOGFILE

python train.py config/optimus/amznKeyword.jsonnet -s log/optimus/amzn/ex1 >>  $LOGFILE

mkdir -p "/gpfs/project/$USER/cooplio/output"
mkdir -p "/gpfs/project/$USER/cooplio/output/log"
cp -r $SCRATCHDIR/log/* /gpfs/project/$USER/cooplio/output/log/.


echo "$PBS_JOBID ($PBS_JOBNAME) @ `hostname` at `date` in "$RUNDIR" END" >> $LOGFILE
echo "`date +"%d.%m.%Y-%T"`" >> $LOGFILE