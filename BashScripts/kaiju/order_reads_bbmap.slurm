#!/bin/sh
#SBATCH --job-name=order_reads			# Job name
#SBATCH --account=group_name			# your own sponser or account
#SBATCH --qos=group_name                # your own sponser or account
#SBATCH --mail-type=ALL               	# Mail events
#SBATCH --mail-user=name@email.com      # Where to send email
#SBATCH --ntasks=1                    	# Run on a single machine (node)
#SBATCH --cpus-per-task 30		  		# Run on a several CPU
#SBATCH --mem-per-cpu=1gb	            # Memory limit
#SBATCH --time=01:00:00               	# Time: hrs:min:sec
#SBATCH --output=output_%j.out		   	# Output and error log  

pwd; hostname; date 

# This code is required only if the reads are not in same order in paired (forward and reverse) fastq files

# load bbmap
module load bbmap

mkdir bbmap_ordered

cd /path/to/fastq/files

for f in $(ls *.fastq.gz | sed -e 's/_1.fastq.gz//' -e 's/_2.fastq.gz//' | sort -u)
do

repair.sh in1=${f}_1.fastq.gz in2=${f}_2.fastq.gz out1= bbmap_ordered/${f}ORDERED_1.fastq.gz out2=bbmap_ordered/${f}ORDERED_2.fastq.gz outs=bbmap_ordered/${f}ORDERED_singleton.fastq.gz repair

done

cd bbmap_ordered

rm *ORDERED_singleton.fastq.gz

# Now, fastq files from bbmap_ordered folder must be given as an input to kaiju

date
